import numpy as np


class GameEvaluator:

    def __init__(self, board_updater):
        self.board_updater = board_updater
        self.black_stones = 0
        self.black_captured = 0
        self.white_stones = 0
        self.white_captured = 0
        self.is_blacks_turn = True
        self.game_history = []
        self.frame_counter = 0

    def get_added_and_removed(self, positions):
        added = 0
        removed = 0
        for position in positions:
            value = self.get_change_at_position(position)
            if value < 0:
                color = "black" if value == -1 else "white"
                print(f'added {color} to {position}')
                added += 1
            else:
                color = "black" if value == 1 else "white"
                print(f'removed {color} from {position}')
                removed += 1
        return added, removed

    def is_valid_board(self, changes):
        if not np.array_equal(self.board_updater.recording_board, self.board_updater.next_board):
            self.frame_counter = 0
            self.board_updater.update_next_board()
            return False

        if self.frame_counter < 10:
            self.frame_counter += 1
            return False

        if len(changes) == 1:
            is_valid_black_turn = self.is_blacks_turn and \
                                  self.board_updater.count_black_on_board() + self.black_captured == self.black_stones + 1 and \
                                  self.board_updater.count_white_on_board() + self.white_captured == self.white_stones
            is_valid_white_turn = not self.is_blacks_turn and \
                                  self.board_updater.count_white_on_board() + self.white_captured == self.white_stones + 1 and \
                                  self.board_updater.count_black_on_board() + self.black_captured == self.black_stones
            return is_valid_black_turn or is_valid_white_turn

        if len(changes) > 1:
            added, removed = self.get_added_and_removed(changes)
            if added == 1:
                is_valid_black_turn = self.is_blacks_turn and \
                                      self.board_updater.count_black_on_board() + self.black_captured == self.black_stones + 1 and \
                                      self.board_updater.count_white_on_board() + self.white_captured + removed == self.white_stones
                is_valid_white_turn = not self.is_blacks_turn and \
                                      self.board_updater.count_white_on_board() + self.white_captured == self.white_stones + 1 and \
                                      self.board_updater.count_black_on_board() + self.black_captured + removed == self.black_stones
                return is_valid_black_turn or is_valid_white_turn
        return False

    def get_change_at_position(self, position):
        difference_board = self.board_updater.get_board_differences()
        return difference_board[position[0]][position[1]]

    def update_board(self, positions):
        _, removed = self.get_added_and_removed(positions)
        if self.is_blacks_turn:
            self.black_stones += 1
            self.white_captured += removed
            self.is_blacks_turn = False
        else:
            self.white_stones += 1
            self.black_captured += removed
            self.is_blacks_turn = True
        self.save_last_move(positions)
        self.board_updater.update_last_board()

    def get_last_move_position(self, positions):
        for position in positions:
            value = self.get_change_at_position(position)
            if value < 0:
                return tuple(position)
        return RuntimeError('No new added stone found')

    def save_last_move(self, positions):
        position = self.get_last_move_position(positions)
        color = 'W'
        # since moved got already changed in update board we have to invert the boolean statement
        if not self.is_blacks_turn:
            color = 'B'
        self.game_history.append([color, position])

    def print_last_move(self):
        print(self.game_history[-1])
