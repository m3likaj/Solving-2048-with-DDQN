import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import os
import csv
from time import time

# Directions for 2048
LEFT, UP, RIGHT, DOWN = (0, -1), (-1, 0), (0, 1), (1, 0)
directions = [LEFT, UP, RIGHT, DOWN]

metrics_file = "training_metrics_4.csv"
if not os.path.exists(metrics_file):
    with open(metrics_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Score", "Invalid Moves", "Highest Tile", "Total Reward", "Epsilon"])

def log_metrics(episode, score, invalid_moves, highest_tile, total_reward, epsilon):
    with open(metrics_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([episode, score, invalid_moves, highest_tile, total_reward, epsilon])

class QNetwork:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
        return model

    def predict(self, state):
        return self.model.predict(state, verbose=0)

    def update(self, state, target):
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
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99

    def act(self, state, board):
        valid_actions = self.get_valid_actions(board)
        if np.random.rand() < self.epsilon:
            return random.choice(valid_actions)
        q_values = self.q_network.predict(state)
        masked_q_values = np.full(self.action_size, -np.inf)
        for action in valid_actions:
            masked_q_values[action] = q_values[0][action]
        return np.argmax(masked_q_values)

    def get_valid_actions(self, board):
        valid_actions = []
        for action_idx, direction in enumerate(directions):
            temp_board = Board(board.boardSize)
            temp_board.board = [row[:] for row in board.board]
            _, moved = temp_board.move(direction, addNextTile=False)
            if moved:
                valid_actions.append(action_idx)
        return valid_actions

    def remember(self, state, action, reward, next_state, done):
        q_current = self.q_network.predict(state)[0][action]
        q_next = 0 if done else self.gamma * np.max(self.target_q_network.predict(next_state)[0])
        priority = abs(reward + q_next - q_current)
        self.memory.append((state, action, reward, next_state, done, priority))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        priorities = np.array([m[5] for m in self.memory])
        probabilities = priorities / sum(priorities)
        indices = np.random.choice(len(self.memory), self.batch_size, p=probabilities)
        minibatch = [self.memory[i] for i in indices]

        states = np.array([m[0] for m in minibatch]).reshape(-1, self.state_size)
        next_states = np.array([m[3] for m in minibatch]).reshape(-1, self.state_size)
        targets = self.q_network.predict(states)
        q_next = self.target_q_network.predict(next_states)

        for i, (state, action, reward, next_state, done, _) in enumerate(minibatch):
            targets[i][action] = reward if done else reward + self.gamma * np.max(q_next[i])

        self.q_network.model.fit(states, targets, epochs=1, verbose=0)

    def update_target_network(self):
        self.target_q_network.model.set_weights(self.q_network.model.get_weights())

def train_agent(board_size, memory_size, batch_size, episodes, max_steps):
    state_size = board_size * board_size
    action_size = 4
    agent = Agent(state_size, action_size, memory_size, batch_size)

    for episode in range(1, episodes + 1):
        board = Board(board_size)
        state = np.array(board.board).flatten().reshape(1, -1)
        total_reward = 0
        invalid_moves = 0
        start_time = time()

        for step in range(max_steps):
            action = agent.act(state, board)
            direction = directions[action]
            prev_board = [row[:] for row in board.board]
            reward, moved = board.move(direction)

            if not moved:
                invalid_moves += 1
                reward = -5
            else:
                max_tile = max(max(row) for row in board.board)
                if max_tile >= 256:
                    reward += 50
                if max_tile >= 512:
                    reward += 100
                if max_tile >= 1024:
                    reward += 200

            next_state = np.array(board.board).flatten().reshape(1, -1)
            done = board.checkLoss()

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break

            if step % 10 == 0:
                agent.replay()
                agent.update_target_network()

        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        highest_tile = max(max(row) for row in board.board)
        log_metrics(episode, board.score, invalid_moves, highest_tile, total_reward, agent.epsilon)

        print(f"Episode {episode}/{episodes}, Score: {board.score}, Invalid Moves: {invalid_moves}, "
              f"Highest Tile: {highest_tile}, Total Reward: {total_reward}, "
              f"Time: {time() - start_time:.2f}s, Epsilon: {agent.epsilon:.2f}")

        if episode % 200 == 0:
            agent.q_network.model.save(f"ddqn_model_4_{episode}.h5")

    return agent

if __name__ == "__main__":
    # Training parameters
    board_size = 4
    memory_size = 10000
    batch_size = 64
    episodes = 500
    max_steps = 1000

    trained_agent = train_agent(board_size, memory_size, batch_size, episodes, max_steps)
    trained_agent.q_network.model.save("trained_ddqn_model_4.h5")
