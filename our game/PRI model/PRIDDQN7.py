import time
import numpy as np
import random
import csv
import os
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
import Game2048 as Game
from tensorflow.keras.optimizers import Adam

#incorporated restart algorithm per episode if it reaches 100+ moves
# epsilon decay only on the first 5 episodes
# restart 10 times max



metrics_file = "training_metrics_PRI7.3.csv" # replaced reward with heuristic
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


def is_valid_action(game, action):
    """
    Return True if taking 'action' changes the board
    (i.e., simulated_game.is_valid_move is True).
    """
    # 1. Copy the current game
    temp_game = Game.Game2048()
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


class QNetwork:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Input(shape= self.state_size),
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def predict(self, state):
        return self.model.predict(state[np.newaxis, :], verbose=0)

    def update(self, state, target):
        state = np.array(state).reshape(-1, self.state_size)  # Ensure proper batch dimension
        target = np.array(target).reshape(-1, self.action_size)  # Ensure proper batch dimension
        self.model.fit(state, target, epochs=1, verbose=0)

class Agent:
    def __init__(self, state_size, action_size, memory_size, batch_size, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.q_network = QNetwork(state_size, action_size)
        self.target_q_network = QNetwork(state_size, action_size)
        self.epsilon = 0.8

    def act(self, state, game=None):

        if game is not None:
            valid_actions = get_valid_actions(game)
        else:
            valid_actions = [0,1,2,3]

        if np.random.rand() < self.epsilon:
            return random.choice(valid_actions)

        q_values = self.q_network.predict(state)
        masked_q_values = np.full(self.action_size, -999999.0)
        for a in valid_actions:
            masked_q_values[a] = q_values[0][a]
        return np.argmax(masked_q_values)

    def remember(self, state, action, reward, next_state, done):
        q_current = self.q_network.predict(state)[0]
        q_next = 0 if done else self.gamma * np.max(self.target_q_network.predict(next_state)[0])
        priority = abs(reward + q_next - q_current)
        self.memory.append((state, action, reward, next_state, done, priority))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        # Prepare arrays for batch training
        states = []
        targets = []

        # We also want next_states for computing Q-values of next states
        next_states = []

        for (state, action, reward, next_state, done, _) in minibatch:
            states.append(state)
            next_states.append(next_state)

        # Convert to numpy arrays
        states = np.array(states).reshape(-1, self.state_size)
        next_states = np.array(next_states).reshape(-1, self.state_size)

        # Predict Q-values for all states in the minibatch (one call)
        q_curr_batch = self.q_network.model.predict(states, verbose=0)
        # Predict Q-values for next_states from the target network (one call)
        q_next_batch = self.target_q_network.model.predict(next_states, verbose=0)

        for i, (state, action, reward, next_state, done, _) in enumerate(minibatch):
            # q_curr_batch[i] is the Q-values for the i-th state in the minibatch
            if done:
                q_curr_batch[i][action] = reward
            else:
                q_curr_batch[i][action] = reward + self.gamma * np.max(q_next_batch[i])

        # Finally, do a single gradient update for the entire batch
        self.q_network.model.fit(states, q_curr_batch, epochs=1, verbose=0)


    def update_target_network(self):
        self.target_q_network.model.set_weights(self.q_network.model.get_weights())


def train_agent_with_selfplay_restart(agent, game, max_restart, max_steps, e,  L = 50):
    tstart = 0
    tend = float('inf')
    restarts_done = 0
    start_time = time.time()

    while (tend - tstart) > L and restarts_done < max_restart:
        restarts_done += 1
        total_reward = 0
        transitions = []
        if restarts_done == 1:
            # brand new game
            state = game.reset()
            t = 0
        else:
            # Jump to state that was recorded at tstart
            game.restore_game_from_history(tstart)
            state = game.preprocess_state()
            t = tstart


        for step in range(max_steps):
            # Q-learning logic:
            prev_board = game.get_board().copy()
            action = agent.act(state, game)

            # Apply action
            if action == 0:
                game.move_left()
            elif action == 1:
                game.move_right()
            elif action == 2:
                game.move_up()
            elif action == 3:
                game.move_down()

            next_state = game.preprocess_state()
            reward = game.get_reward(prev_board)
            done = not game.can_move() or step == max_steps - 1

            transitions.append((state, action, reward, next_state, done))
            state = next_state
            t += 1
            total_reward += reward

            if done:
                break
            if e < 5 :
                agent.epsilon = max(0.1, pow(0.8, step))

        tend = t

        # Do your BackwardLearning
        for (s, a, r, s_next, dn) in transitions:
            agent.remember(s, a, r, s_next, dn)
        agent.replay()
        agent.update_target_network()

        score = game.get_score()
        invalid_moves = game.get_invalid_moves()
        highest_tile = game.get_highest_tile()
        if restarts_done == 1:
            log_metrics(e, score, invalid_moves, highest_tile, total_reward, agent.epsilon)

        print(
            f"Restart {restarts_done}, tstart={tstart}, tend={tend}, Score={score}, "
            f"Highest={highest_tile}, Invalid={invalid_moves}, Reward={total_reward}, Time={time.time()-start_time:.2f}"
        )
        # If we want to "restart" from mid-point:
        if (tend - tstart) > L:
            tstart = (tstart + tend) // 2


def train_agent(agent, episodes, max_steps):
    game = Game.Game2048()

    print("Agent created")
    for e in range(1, episodes+1):
        print(f"Episode {e}/{episodes} ")
        train_agent_with_selfplay_restart(agent, game, 10, max_steps, e)
        if e % 200 == 0:
            agent.q_network.model.save(f"ddqn_model_PRI7.3_{e}.h5")


if __name__ == '__main__':
    state_size = 16
    action_size = 4
    memory_size = 10000
    batch_size = 64
    episodes = 500
    max_steps = 1000

    agent = Agent(state_size, action_size, memory_size, batch_size)
    train_agent(agent, episodes, max_steps)
    agent.q_network.model.save("trained_ddqn_model_PRI7.3.h5")
