# gui_2048.py

import tkinter as tk
import numpy as np
from PRIDDQN9 import *
import Game2048 as Game
from tensorflow.keras.models import load_model
from Expectiminimax import *
import multiprocessing as mp

class Game2048GUI(tk.Tk):
    def __init__(self, agent):
        super().__init__()
        self.title("2048 - Agent Play")
        self.agent = agent

        # Create the game environment
        self.game = Game.Game2048()

        # Main frame for the 4x4 grid
        self.main_frame = tk.Frame(self, bg="#bbada0")
        self.main_frame.pack(padx=20, pady=20)

        self.tile_labels = []
        for row in range(4):
            row_labels = []
            for col in range(4):
                label = tk.Label(
                    self.main_frame,
                    text='',
                    width=4,
                    height=2,
                    font=("Helvetica", 24, "bold"),
                    bg="#cdc1b4",
                    fg="#776e65",
                    borderwidth=2,
                    relief="groove"
                )
                label.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
                row_labels.append(label)
            self.tile_labels.append(row_labels)

        # Score label
        self.score_label = tk.Label(self, text="Score: 0", font=("Helvetica", 16))
        self.score_label.pack()

        # Start button
        self.start_button = tk.Button(self, text="Start", command=self.start_game)
        self.start_button.pack(pady=10)

    def start_game(self):
        """Reset the game and start playing automatically."""
        self.game.reset()
        self.update_ui()
        self.after(500, self.step)  # call step after 500 ms

    def step(self):
        """Perform one agent move, update UI, and schedule next step."""
        if self.game.can_move():
            state = self.game.preprocess_state()
            valid_actions = get_valid_actions(self.game)

            # Use the agent to choose action
            zeroes= np.count_nonzero(self.game.get_board() == 0)
            if zeroes > 4 :
                action = self.agent.act(state, self.game)
            else:
                MODEL_PATH = "trained_ddqn_model_PRI7.h5"
                action = self.agent.act(state, self.game)
            print(f"Action: {action}")
            # Apply the action
            if action == 0:
                self.game.move_left()
            elif action == 1:
                self.game.move_right()
            elif action == 2:
                self.game.move_up()
            elif action == 3:
                self.game.move_down()

            # Update UI
            self.update_ui()
            # Schedule next move
            self.after(200, self.step)
        else:
            # Game over
            final_score = self.game.get_score()
            self.score_label.config(text=f"Game Over! Final Score: {final_score}")

    def update_ui(self):
        """Refresh the board tiles and score."""
        board = self.game.get_board()
        for row in range(4):
            for col in range(4):
                val = board[row][col]
                tile_label = self.tile_labels[row][col]
                if val == 0:
                    tile_label.config(text="", bg="#cdc1b4")
                else:
                    tile_label.config(
                        text=str(val),
                        bg=self.get_tile_color(val),
                        fg="#f9f6f2" if val > 4 else "#776e65"
                    )
        # Update score
        current_score = self.game.get_score()
        self.score_label.config(text=f"Score: {current_score}")

    @staticmethod
    def get_tile_color(value):
        """Return a background color based on tile value."""
        colors = {
            0:    "#cdc1b4",
            2:    "#eee4da",
            4:    "#ede0c8",
            8:    "#f2b179",
            16:   "#f59563",
            32:   "#f67c5f",
            64:   "#f65e3b",
            128:  "#edcf72",
            256:  "#edcc61",
            512:  "#edc850",
            1024: "#edc53f",
            2048: "#edc22e"
        }
        return colors.get(value, "#3c3a32")


def main():
    # Create an Agent with the same hyperparameters used in training
    state_size = 16
    action_size = 4
    memory_size = 10000
    batch_size = 64
    agent = Agent(state_size, action_size, memory_size, batch_size)

    # Load the trained model weights
    MODEL_PATH = "trained_ddqn_model_PRI3.h5"
    agent.q_network.model.load_weights(MODEL_PATH)
    agent.update_target_network()

    # Set epsilon to 0 so the agent plays greedily
    agent.epsilon = 0.0

    # Create and run GUI
    app = Game2048GUI(agent)
    app.mainloop()

def main2():
    # Create an Agent with the same hyperparameters used in training
    state_size = 16
    action_size = 4
    memory_size = 10000
    batch_size = 64
    agent = Agent(state_size, action_size, memory_size, batch_size)

    # Load the trained model weights
    MODEL_PATH = "trained_ddqn_model_PRI7.h5"
    agent.q_network.model.load_weights(MODEL_PATH)
    agent.update_target_network()

    # Set epsilon to 0 so the agent plays greedily
    agent.epsilon = 0.0

    # Create and run GUI
    app = Game2048GUI(agent)
    app.mainloop()


if __name__ == "__main__":
    main2()
