import cv2
import numpy as np

from slider import Slider
from board_updater import BoardUpdater
from settings import WINDOW_SIZE, load_points_from_file, save_points_to_file
from visualizer import Visualizer

# black = 1, white = 2
frame_counter = 0

black_stones = 0
black_captured = 0
white_stones = 0
white_captured = 0
is_blacks_turn = True

game_history = []


def get_added_and_removed(positions):
    added = 0
    removed = 0
    for position in positions:
        value = get_change_at_position(position)
        if value < 0:
            color = "black" if value == -1 else "white"
            print(f'added {color} to {position}')
            added += 1
        else:
            color = "black" if value == 1 else "white"
            print(f'removed {color} from {position}')
            removed += 1
    return added, removed


def is_valid_board(changes):
    global frame_counter
    if not np.array_equal(board_updater.recording_board, board_updater.next_board):
        frame_counter = 0
        board_updater.update_next_board()
        return False

    if frame_counter < 10:
        frame_counter += 1
        return False

    if len(changes) == 1:
        is_valid_black_turn = is_blacks_turn and \
                              board_updater.count_black_on_board() + black_captured == black_stones + 1 and \
                              board_updater.count_white_on_board() + white_captured == white_stones
        is_valid_white_turn = not is_blacks_turn and \
                              board_updater.count_white_on_board() + white_captured == white_stones + 1 and \
                              board_updater.count_black_on_board() + black_captured == black_stones
        return is_valid_black_turn or is_valid_white_turn

    if len(changes) > 1:
        added, removed = get_added_and_removed(changes)
        if added == 1:
            is_valid_black_turn = is_blacks_turn and \
                                  board_updater.count_black_on_board() + black_captured == black_stones + 1 and \
                                  board_updater.count_white_on_board() + white_captured + removed == white_stones
            is_valid_white_turn = not is_blacks_turn and \
                                  board_updater.count_white_on_board() + white_captured == white_stones + 1 and \
                                  board_updater.count_black_on_board() + black_captured + removed == black_stones
            return is_valid_black_turn or is_valid_white_turn
    return False


def get_change_at_position(position):
    difference_board = board_updater.get_board_differences()
    return difference_board[position[0]][position[1]]


def update_board(positions):
    global black_stones, black_captured, white_stones, white_captured, is_blacks_turn, last_board
    _, removed = get_added_and_removed(positions)
    if is_blacks_turn:
        black_stones += 1
        white_captured += removed
        is_blacks_turn = False
    else:
        white_stones += 1
        black_captured += removed
        is_blacks_turn = True
    save_last_move(positions)
    board_updater.update_last_board()


def get_last_move_position(positions):
    for position in positions:
        value = get_change_at_position(position)
        if value < 0:
            return tuple(position)
    return RuntimeError('No new added stone found')


def save_last_move(positions):
    global game_history
    position = get_last_move_position(positions)
    color = 'W'
    # since moved got already changed in update board we have to invert the boolean statement
    if not is_blacks_turn:
        color = 'B'
    game_history.append([color, position])


def print_last_move():
    print(game_history[-1])


def window_for_black_white(white, i_white, black, i_black):
    # Resize the images to the same size (optional)
    white = cv2.resize(white, (0, 0), None, 0.5, 0.5)
    i_white = cv2.resize(i_white, (0, 0), None, 0.5, 0.5)
    black = cv2.resize(black, (0, 0), None, 0.5, 0.5)
    i_black = cv2.resize(i_black, (0, 0), None, 0.5, 0.5)
    top_row = cv2.hconcat((white, black))
    bottom_row = cv2.hconcat((i_white, i_black))
    grid = cv2.vconcat((top_row, bottom_row))
    return grid


if __name__ == '__main__':
    load_points_from_file()

    slider = Slider()
    board_updater = BoardUpdater()
    visualizer = Visualizer(slider, board_updater)

    cap = visualizer.initialize_cam()

    # Set the mouse callback function
    cv2.namedWindow('camera')
    cv2.setMouseCallback('camera', visualizer.move_point)

    analyzing = False

    transformed = None
    white = None
    black = None
    identified_black = None
    identified_white = None
    align = None
    final_board = None
    while True:
        board_updater.clear_recording_board()
        ret, frame = cap.read()
        visualizer.draw_rectangle_with_corner_points(frame)

        if visualizer.M is not None:
            size = cv2.getTrackbarPos('size', 'Slider')
            transformed = cv2.warpPerspective(frame, visualizer.M, (WINDOW_SIZE, WINDOW_SIZE))
            white = visualizer.get_white(transformed)
            black = visualizer.get_black(transformed)
            align = visualizer.draw_green_point_grid(transformed, size)
            identified_black = visualizer.color_identifed_grid_greyscale(black, True, size)
            identified_white = visualizer.color_identifed_grid_greyscale(white, False, size)

            changes = board_updater.get_changed_position()
            if analyzing and is_valid_board(changes):
                print(changes)
                update_board(changes)
                print_last_move()
                final_board = visualizer.drawn_board(transformed)
            # cv2.imshow('transformed', transformed)
            grid = window_for_black_white(white, identified_white, black, identified_black)
            cv2.imshow('alinger', align)
            cv2.imshow('combined', grid)
            if final_board is not None:
                cv2.imshow("Final", final_board)

        cv2.imshow('camera', frame)

        # Check for key presses
        key = cv2.waitKey(1)

        if key == ord('s'):
            if analyzing:
                print('stop analysis')
                print(game_history)
                print(f"white: {white_stones}, captured: {black_captured}")
                print(f"black: {black_stones}, captured: {white_captured}")
            else:
                print('start analysis')
            analyzing = not analyzing
        if key == ord('r'):
            POINTS = [[200, 200], [200, 400], [400, 400], [400, 200]]
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    save_points_to_file()
