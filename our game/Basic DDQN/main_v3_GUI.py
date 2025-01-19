import numpy as np
import random
import time
import csv
import os
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers.legacy import Adam
import tkinter as tk
from tkinter import messagebox
# this is V4 with fixed expectimax

metrics_file = "training_metrics_v5.csv"
if not os.path.exists(metrics_file):
    with open(metrics_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header row
        writer.writerow(["Episode", "Score", "Invalid Moves", "Highest Tile", "Total Reward", "Epsilon"])

def log_metrics(episode, score, invalid_moves, highest_tile, total_reward, epsilon):
    with open(metrics_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        # Write a row for the current episode
        writer.writerow([episode, score, invalid_moves, highest_tile, total_reward, epsilon])


class Game2048:
    def __init__(self):
        self.score = 0
        self.invalidMoves = 0
        self.is_valid_move = True
        self.highest_tile = 0
        self.consecutive_invalid_moves = 0
        self.board = self.reset()


    def reset(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.invalidMoves = 0
        self.highest_tile = 0
        self.is_valid_move = True
        self.consecutive_invalid_moves = 0
        self.add_new_tile()
        self.add_new_tile()
        return self.get_board()

    def add_new_tile(self):
        empty_positions = np.argwhere(self.board == 0)
        if empty_positions.size > 0:
            row, col = random.choice(empty_positions)
            self.board[row, col] = 2 if random.random() < 0.9 else 4

    def move_left(self):
        changed = False
        for i in range(4):
            row = self.board[i][self.board[i] != 0]
            merged_row = []
            skip = False
            for j in range(len(row)):
                if skip:
                    skip = False
                    continue
                if j + 1 < len(row) and row[j] == row[j + 1]:
                    merged_row.append(row[j] * 2)
                    self.score += row[j] * 2
                    skip = True
                    changed = True
                else:
                    merged_row.append(row[j])
            merged_row += [0] * (4 - len(merged_row))
            if not np.array_equal(self.board[i], merged_row):
                changed = True
            self.board[i] = merged_row
        if changed:
            self.add_new_tile()
            self.highest_tile = np.max(self.board)
            self.is_valid_move = True
            self.consecutive_invalid_moves = 0
        else:
            self.invalidMoves += 1
            self.consecutive_invalid_moves += 1
            self.is_valid_move = False


    def move_right(self):
        self.board = np.fliplr(self.board)
        self.move_left()
        self.board = np.fliplr(self.board)

    def move_up(self):
        self.board = np.rot90(self.board)
        self.move_left()
        self.board = np.rot90(self.board, -1)

    def move_down(self):
        self.board = np.rot90(self.board)
        self.move_right()
        self.board = np.rot90(self.board, -1)

    def get_board(self):
        return self.board

    def get_score(self):
        return self.score

    def get_invalid_moves(self):
        return self.invalidMoves

    def get_highest_tile(self):
        return self.highest_tile

    def is_move_valid(self):
        return self.is_valid_move

    def can_move(self):
        return( np.any(self.board == 0) or np.any(
            self.board[:-1, :] == self.board[1:, :]
        ) or np.any(self.board[:, :-1] == self.board[:, 1:]) )

    def get_reward(self, prev_board):
        zero_old = np.count_nonzero(prev_board == 0)
        zero_new = np.count_nonzero(self.board == 0)
        if not self.is_valid_move:
            return -10*self.consecutive_invalid_moves
        elif not self.can_move():
            return -100
        elif self.highest_tile > np.max(prev_board):
            return 100
        elif zero_new > zero_old:
            return 50
        elif zero_new == zero_old:
            return 10
        else:
            return 0


def preprocess_state(board):
    return np.log2(np.maximum(board, 1)).flatten() / 11.0  # Normalize to [0, 1]

def build_ddqn_model(input_size=16, output_size=4, hidden_sizes=[256, 256]):
    model = Sequential()
    model.add(Input(shape=(input_size,)))  # Input layer
    for hidden_size in hidden_sizes:
        model.add(Dense(hidden_size, activation="relu"))  # Hidden layers
    model.add(Dense(output_size, activation="linear"))  # Output layer
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model

def is_valid_action(game, action):
    """
    Return True if taking 'action' changes the board
    (i.e., simulated_game.is_valid_move is True).
    """
    # 1. Copy the current game
    temp_game = Game2048()
    temp_game.board = game.board.copy()
    temp_game.invalidMoves = game.invalidMoves
    temp_game.is_valid_move = game.is_valid_move


    # 2. Apply the move
    if action == 0:
        temp_game.move_left()
    elif action == 1:
        temp_game.move_right()
    elif action == 2:
        temp_game.move_up()
    elif action == 3:
        temp_game.move_down()

    # 3. Check if it was valid
    return temp_game.is_valid_move


def get_valid_actions(game):
    valid_actions = []
    for action in range(4):  # 0=left,1=right,2=up,3=down
        if is_valid_action(game, action):
            valid_actions.append(action)
    return valid_actions


def evaluate_board(board):
    """
    Simple heuristic:
      - reward empty tiles
      - reward having a high max tile
    """
    empty_tiles = np.count_nonzero(board == 0)
    max_tile = np.max(board)
    return 10 * empty_tiles + max_tile


def depth_limited_expectimax_search(
        game,
        depth,
        max_depth,
        threshold,
        is_agent=True,
        current_best=float('-inf')
):
    """
    Depth-limited Expectimax with threshold-based pruning.

    :param game: An instance of Game2048 (simulated).
    :param depth: Current search depth.
    :param max_depth: Maximum depth to explore.
    :param threshold: Pruning threshold (heuristic-based).
    :param is_agent: True if it's the agent's turn; False for environment's turn.
    :param current_best: Tracks the best score found so far for pruning.
    :return: (value, move) for agent's turn or just value for environment's turn.
    """
    # 1. Termination condition
    if depth == max_depth or not game.can_move():
        # Return the heuristic value of the board
        return evaluate_board(game.get_board())

    if is_agent:
        # Agent’s turn: choose the move that maximizes expected value
        best_value = float('-inf')
        best_move = None

        # Enumerate all possible moves
        for action in [0, 1, 2, 3]:  # Left, Right, Up, Down
            # Simulate the move
            simulated_game = Game2048()
            simulated_game.board = game.board.copy()
            simulated_game.score = game.score
            simulated_game.invalidMoves = game.invalidMoves
            simulated_game.is_valid_move = game.is_valid_move
            simulated_game.consecutive_invalid_moves = game.consecutive_invalid_moves
            simulated_game.highest_tile = game.highest_tile


            if action == 0:
                simulated_game.move_left()
            elif action == 1:
                simulated_game.move_right()
            elif action == 2:
                simulated_game.move_up()
            elif action == 3:
                simulated_game.move_down()

            if not simulated_game.is_move_valid():
                # Invalid move: give a large penalty or skip
                continue
            else:
                # Recursively compute the environment’s response (expectation)
                value = depth_limited_expectimax_search(
                    simulated_game,
                    depth + 1,
                    max_depth,
                    threshold,
                    is_agent=False,
                    current_best=best_value
                )

            # Threshold pruning: if our current branch is already better than
            # a large positive threshold, we might skip other branches
            # or if it’s too low compared to best_value, skip further exploration.
            if value > best_value:
                best_value = value
                best_move = action

            # Prune if the value is far below the current_best (for top-level calls)
            if (best_value - current_best) > threshold:
                break

        if depth == 0:
            # At the root node, return both the best value and the best move
            return best_value, best_move
        else:
            # Return just the best value deeper in the tree
            return best_value

    else:
        # Environment's turn: average (expectation) over all possible tile placements
        empty_positions = np.argwhere(game.board == 0)
        if empty_positions.size == 0:
            # No empty spots, treat as terminal
            return evaluate_board(game.get_board())

        value_sum = 0.0
        total_prob = 0.0
        # For each empty position, tile can be 2 (90% chance) or 4 (10% chance)
        for (r, c) in empty_positions:
            for tile_value, prob in [(2, 0.9), (4, 0.1)]:
                simulated_game = Game2048()
                simulated_game.board = game.board.copy()
                simulated_game.score = game.score
                simulated_game.board[r, c] = tile_value
                simulated_game.invalidMoves = game.invalidMoves
                simulated_game.is_valid_move = game.is_valid_move
                simulated_game.consecutive_invalid_moves = game.consecutive_invalid_moves
                simulated_game.highest_tile = game.highest_tile


                value = depth_limited_expectimax_search(
                    simulated_game,
                    depth + 1,
                    max_depth,
                    threshold,
                    is_agent=True,
                    current_best=current_best
                )
                value_sum += value * prob
                total_prob += prob

                # Optional: If the environment’s moves push the value
                # far below the threshold, you could do an early break here as well.
                # if (value_sum/total_prob - current_best) < -threshold:
                #     break

        expected_value = value_sum / total_prob
        return expected_value


class PrioritizedReplayBuffer:
    def __init__(self, max_size=20000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.max_size = max_size
        self.buffer = []
        self.priorities = []
        self.alpha = alpha  # Priority exponent
        self.beta = beta  # Importance-sampling exponent
        self.beta_increment = beta_increment
        self.epsilon = 1e-6  # Small constant to avoid zero priority

    def add(self, experience, td_error):
        priority = (abs(td_error) + self.epsilon) ** self.alpha
        self.buffer.append(experience)
        self.priorities.append(priority)

        # Remove oldest experience if buffer exceeds max size
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
            self.priorities.pop(0)

    def sample(self, batch_size):
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()  # Normalize priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        # Importance-sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights

        # Increment beta for stability
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Get sampled experiences
        samples = [self.buffer[idx] for idx in indices]
        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for i, td_error in zip(indices, td_errors):
            self.priorities[i] = (abs(td_error) + self.epsilon) ** self.alpha



class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.main_model = build_ddqn_model(state_size, action_size)
        self.target_model = build_ddqn_model(state_size, action_size)
        self.update_target_model()
        self.replay_buffer = PrioritizedReplayBuffer()
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.997
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.use_expectimax = False

    def update_target_model(self):
        self.target_model.set_weights(self.main_model.get_weights())


    def train(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return

        # Sample from prioritized replay buffer
        batch, indices, weights = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        # Predict Q-values
        q_values = self.main_model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)

        # Compute TD errors and targets
        td_errors = []
        for i in range(self.batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.max(next_q_values[i])
            td_errors.append(target - q_values[i, actions[i]])
            q_values[i, actions[i]] = target

        # Update the model
        self.main_model.fit(states, q_values, sample_weight=weights, verbose=0, batch_size=self.batch_size)

        # Update priorities in the replay buffer
        self.replay_buffer.update_priorities(indices, td_errors)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act_with_expectimax(self, game, max_depth=3, threshold=200):
        """
        Use the depth-limited Expectimax with pruning to choose an action.
        :param game: Current Game2048 instance.
        :param max_depth: Max depth for the search.
        :param threshold: Pruning threshold.
        :return: best_action chosen by Expectimax.
        """
        # We'll call the search function from the root (depth=0).
        # The function returns (best_value, best_move).
        best_value, best_move = depth_limited_expectimax_search(
            game,
            depth=0,
            max_depth=max_depth,
            threshold=threshold,
            is_agent=True,
            current_best=float('-inf')
        )
        if best_move is None:
            # Fallback if somehow no valid move is found
            return random.randint(0, 3)
        return best_move

    def act(self, state, game=None):
        # Example: if we're using Expectimax in certain episodes
        if self.use_expectimax and game is not None:
            return self.act_with_expectimax(game, max_depth=3, threshold=200)

        # Standard DDQN-based action
        if game is not None:
            valid_actions = get_valid_actions(game)
        else:
            valid_actions = [0,1,2,3]
        if not valid_actions:
            return random.randint(0, self.action_size - 1)
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        else:
            # Evaluate Q-values for each valid action
            q_values = self.main_model.predict(state[np.newaxis, :], verbose=0)[0]

            # Set invalid actions to a very negative value
            masked_q_values = np.full(self.action_size, -999999.0)
            for a in valid_actions:
                masked_q_values[a] = q_values[a]

            # Argmax of the masked Q-values
            return np.argmax(masked_q_values)


def train_ddqn(agent, episodes=500, update_target_every=10):
    game = Game2048()
    print("Starting training loop...")
    for episode in range(1, episodes + 1):
        start_time = time.time()
        state = preprocess_state(game.reset())
        score = 0
        total_reward = 0
        done = False
        step_count = 0

        agent.use_expectimax = (episode % 10 == 0)

        while not done and step_count < 1000:
            action = agent.act(state, game=game)
            prev_board = game.get_board().copy()
            if action == 0:
                game.move_left()
            elif action == 1:
                game.move_right()
            elif action == 2:
                game.move_up()
            elif action == 3:
                game.move_down()

            next_state = preprocess_state(game.get_board())
            reward = game.get_reward(prev_board)

            done = not game.can_move()


            # Calculate initial TD error
            q_values = agent.main_model.predict(state[np.newaxis, :], verbose=0)
            target = reward
            if not done:
                target += agent.gamma * np.max(agent.target_model.predict(next_state[np.newaxis, :], verbose=0))
            td_error = target - q_values[0][action]

            # Add experience to PER buffer
            agent.replay_buffer.add((state, action, reward, next_state, done), td_error)
            state = next_state
            step_count += 1
            total_reward += reward

            # Train the agent
            agent.train()
        # Update target network periodically
        if episode % update_target_every == 0:
            agent.update_target_model()

        #log to csv
        score = game.get_score()
        invalid_moves = game.get_invalid_moves()
        highest_tile = game.get_highest_tile()

        log_metrics(episode, score, invalid_moves, highest_tile, total_reward, agent.epsilon)
        print( f"Episode {episode}/{episodes}, Score: {score}, Invalid-Moves:{invalid_moves}, Highest-tile: {game.get_highest_tile()} Reward: {total_reward}, time: {time.time() - start_time:.2f} seconds")
        if episode % 100 == 0:
            agent.main_model.save(f"ddqn_model_v5_{episode}.h5")




class GameGUI:
    def __init__(self, master, agent=None):
        self.master = master
        self.master.title("2048 Game")
        self.game = Game2048()
        self.agent = agent  # The trained agent
        self.canvas = tk.Canvas(master, width=400, height=400, bg="lightgrey")
        self.canvas.pack()
        self.IM_label = tk.Label(master, text="Invalid Moves: 0", font=("Arial", 24))
        self.IM_label.pack()
        self.score_label = tk.Label(master, text="Score: 0", font=("Arial", 24))
        self.score_label.pack()
        self.draw_board()

        # Buttons to control agent play
        self.play_button = tk.Button(master, text="Play Agent", command=self.play_agent)
        self.play_button.pack()
        self.reset_button = tk.Button(master, text="Reset", command=self.reset_game)
        self.reset_button.pack()

    def draw_board(self):
        self.canvas.delete("all")
        for i in range(4):
            for j in range(4):
                value = self.game.get_board()[i, j]
                x1, y1 = j * 100, i * 100
                x2, y2 = x1 + 100, y1 + 100
                color = self.get_color(value)
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")
                if value != 0:
                    self.canvas.create_text(
                        x1 + 50, y1 + 50, text=str(value), font=("Arial", 24)
                    )
        self.update_score_and_IM()

    def update_score_and_IM(self):
        self.score_label.config(text=f"Score: {self.game.get_score()}")
        self.IM_label.config(text=f"Invalid Moves: {self.game.get_invalid_moves()}")

    def get_color(self, value):
        colors = {
            0: "lightgray",
            2: "#edd68c",
            4: "#edc18c",
            8: "#ab7e6c",
            16: "#f59563",
            32: "#f67c5f",
            64: "#a84d36",
            128: "#5f6631",
            256: "#315166",
            512: "#4b3166",
            1024: "#643166",
            2048: "#316644",
        }
        return colors.get(value, "#3c3a32")

    def play_agent(self):
        """Automate the game with the trained agent."""
        if self.agent is None:
            messagebox.showerror("Error", "No trained agent loaded!")
            return

        def agent_play():
            if not self.game.can_move():
                messagebox.showinfo("Game Over", f"Final Score: {self.game.get_score()}")
                return

            # Get the current state
            state = preprocess_state(self.game.get_board())

            # Agent decides an action
            action = self.agent.act(state)

            # Perform the action
            if action == 0:
                self.game.move_left()
            elif action == 1:
                self.game.move_right()
            elif action == 2:
                self.game.move_up()
            elif action == 3:
                self.game.move_down()

            # Update the board
            self.draw_board()

            # Schedule the next step
            self.master.after(200, agent_play)  # Delay for smoother gameplay

        agent_play()

    def reset_game(self):
        self.game.reset()
        self.draw_board()


if __name__ == '__main__':
    root = tk.Tk()
    trained_agent = DDQNAgent(state_size=16, action_size=4)  # Ensure state_size matches your game's input
    trained_agent.main_model.load_weights("trained_ddqn_model_v.h5")  # Load the trained model weights
    app = GameGUI(root, agent=trained_agent)
    root.mainloop()





