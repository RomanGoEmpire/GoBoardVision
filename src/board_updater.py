import numpy as np

from src.settings import ROWS, COLS


class BoardUpdater:

    def __init__(self):
        self.initial_board = np.zeros((ROWS, COLS))
        self.last_board = np.zeros((ROWS, COLS))
        self.recording_board = self.initial_board.copy()
        self.next_board = self.initial_board.copy()

    def clear_recording_board(self):
        self.recording_board = self.initial_board.copy()

    def update_last_board(self):
        self.last_board = self.next_board.copy()

    def update_next_board(self):
        self.next_board = self.recording_board.copy()

    def get_board_differences(self):
        return self.last_board - self.next_board

    def count_black_on_board(self):
        return np.count_nonzero(self.recording_board == 1)

    def count_white_on_board(self):
        return np.count_nonzero(self.recording_board == 2)

    def get_changed_position(self):
        return np.column_stack(np.where(self.last_board != self.next_board))
