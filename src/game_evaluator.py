import numpy as np


def hand():
    return False


def is_arr_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)


class GameEvaluator:

    def __init__(self, board_updater):
        self.board_updater = board_updater
        self.game_history = []
        self.black_stones = 0
        self.black_captured = 0
        self.white_stones = 0
        self.white_captured = 0
        self.is_blacks_turn = True
        self.frame_counter = 0

    def get_change_at_position(self, position):
        difference_board = self.board_updater.get_board_differences()
        return difference_board[position[0]][position[1]]

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
        if hand():
            return False
        if not np.array_equal(self.board_updater.recording_board, self.board_updater.next_board):
            self.frame_counter = 0
            self.board_updater.update_next_board()
            return False

        if self.frame_counter < 10:
            self.frame_counter += 1
            return False
        black_valid = self.board_updater.count_black_on_board() + self.black_captured == self.black_stones + 1
        white_valid = self.board_updater.count_white_on_board() + self.white_captured == self.white_stones + 1

        if len(changes) == 1:
            is_valid_black_turn = self.is_blacks_turn and black_valid and \
                                  self.board_updater.count_white_on_board() + self.white_captured == self.white_stones
            is_valid_white_turn = not self.is_blacks_turn and white_valid and \
                                  self.board_updater.count_black_on_board() + self.black_captured == self.black_stones
            return is_valid_black_turn or is_valid_white_turn

        if len(changes) > 1:
            added, removed = self.get_added_and_removed(changes)
            if added == 1 and self.remove_is_valid(changes):
                is_valid_black_turn = self.is_blacks_turn and black_valid and \
                                      self.board_updater.count_white_on_board() + \
                                      self.white_captured + removed == self.white_stones
                is_valid_white_turn = not self.is_blacks_turn and white_valid and \
                                      self.board_updater.count_black_on_board() + \
                                      self.black_captured + removed == self.black_stones
                return is_valid_black_turn or is_valid_white_turn
        return False

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

    def get_removed_positions(self, positions):
        removed_position = []
        for position in positions:
            value = self.get_change_at_position(position)
            if value > 0:
                removed_position.append(position)
        return removed_position

    def save_last_move(self, positions):
        position = self.get_last_move_position(positions)
        color = 'W'
        # since moved got already changed in update board we have to invert the boolean statement
        if not self.is_blacks_turn:
            color = 'B'
        self.game_history.append([color, position])

    def remove_is_valid(self, positions):
        x, y = self.get_last_move_position(positions)
        print(x,y)
        board = self.board_updater.last_board.copy()
        color = '1' if self.is_blacks_turn else '2'
        opponent = '2' if self.is_blacks_turn else '1'
        board[x][y] = color
        removed_positions = self.get_removed_positions(positions)
        print(board)
        print(removed_positions)
        for move in removed_positions:
            if not self.captured(opponent, color, move, board, removed_positions):
                return False
        return True

    def print_last_move(self):
        print(self.game_history[-1])

    def captured(self, opponent, color, move, board, position):
        x = move[0]
        y = move[1]

        left_color_and_removed = board[x - 1][y] == color and not is_arr_in_list(np.array([x - 1, y]), position)
        right_color_and_removed = board[x + 1][y] == color and not is_arr_in_list(np.array([x + 1, y]), position)
        top_color_and_removed = board[x][y - 1] == color and not is_arr_in_list(np.array([x, y - 1]), position)
        bottom_color_and_removed = board[x][y + 1] == color and not is_arr_in_list(np.array([x, y + 1]), position)

        print(board[x - 1][y] == color)

        # Check left neighbor if not at left edge
        if x > 1 and (board[x - 1][y] == 0 or left_color_and_removed):
            return False
            # Check right neighbor if not at right edge
        if x < 19 and (board[x + 1][y] == 0 or right_color_and_removed):
            return False
            # Check top neighbor if not at top edge
        if y > 1 and (board[x][y - 1] == 0 or top_color_and_removed):
            return False
            # Check bottom neighbor if not at bottom edge
        if y < 19 and (board[x][y + 1] == 0 or bottom_color_and_removed):
            return False
            # If none of the above conditions are met, return False
        return True
