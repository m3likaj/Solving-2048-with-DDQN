import numpy as np
import random

class Game2048:
    def __init__(self):
        self.score = 0
        self.invalidMoves = 0
        self.is_valid_move = True
        self.highest_tile = 0
        self.consecutive_invalid_moves = 0
        self.history = []  # Store snapshots of game state
        self.board = self.reset()


    def reset(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.invalidMoves = 0
        self.highest_tile = 0
        self.is_valid_move = True
        self.consecutive_invalid_moves = 0
        self.history.clear()
        self.record_history()
        self.add_new_tile()
        self.add_new_tile()
        return self.board

    def record_history(self):
        """
        Store a snapshot of the current board, score, invalidMoves, etc.
        into self.history so we can restore later.
        """
        snapshot = {
            'board': np.copy(self.board),
            'score': self.score,
            'invalidMoves': self.invalidMoves,
            'is_valid_move': self.is_valid_move,
             'rng_state': np.random.get_state()
        }
        self.history.append(snapshot)

    def restore_game_from_history(self, t_index):
        """
        Restore the game state from self.history[t_index].
        This will let you jump to a specific time-step in your self-play.
        """
        if t_index < 0 or t_index >= len(self.history):
            raise ValueError(f"t_index {t_index} out of range (0..{len(self.history) - 1})")

        snapshot = self.history[t_index]
        self.board = np.copy(snapshot['board'])
        self.score = snapshot['score']
        self.invalidMoves = snapshot['invalidMoves']
        self.is_valid_move = snapshot['is_valid_move']
        np.random.set_state(snapshot['rng_state'])

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
        self.record_history()

    def move_right(self):
        self.board = np.fliplr(self.board)
        self.move_left()
        self.board = np.fliplr(self.board)
        self.record_history()

    def move_up(self):
        self.board = np.rot90(self.board)
        self.move_left()
        self.board = np.rot90(self.board, -1)
        self.record_history()

    def move_down(self):
        self.board = np.rot90(self.board)
        self.move_right()
        self.board = np.rot90(self.board, -1)
        self.record_history()

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
            return -50
        elif self.highest_tile > np.max(prev_board):
            return 20
        elif zero_new > zero_old:
            return 30
        elif zero_new == zero_old:
            return 10
        else:
            return 0

    def preprocess_state(self):
        return np.log2(np.maximum(self.board, 1)).flatten() / 11.0  # Normalize to [0, 1]
