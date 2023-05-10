import cv2
import win32gui

from src.game_evaluator import GameEvaluator
from src.sgf_converter import SGFConverter
from src.slider import Slider
from src.board_updater import BoardUpdater
from src.settings import WINDOW_SIZE
from src.visualizer import Visualizer
from src.window_capture import WindowsCapture


def main():
    slider = Slider()
    updater = BoardUpdater()
    v = Visualizer(slider, updater)
    evaluator = GameEvaluator(updater)
    capture = WindowsCapture()

    converter = SGFConverter(evaluator.game_history)
    # cap = v.initialize_cam() TODO: Make a switch if you want to get the camera or window

    # Set the mouse callback function
    cv2.namedWindow('camera')
    cv2.setMouseCallback('camera', v.move_point)

    analyzing = False

    transformed = None
    white = None
    black = None
    identified_black = None
    identified_white = None
    align = None
    final_board = None

    while True:
        updater.clear_recording_board()
        # ret, frame = cap.read() TODO: Make a switch if you want to get the camera or window
        frame = capture.get_screenshot()
        v.draw_rectangle_with_corner_points(frame)

        if v.M is not None:
            size = cv2.getTrackbarPos('size', 'Slider')
            transformed = cv2.warpPerspective(frame, v.M, (WINDOW_SIZE, WINDOW_SIZE))
            white = v.get_white(transformed)
            black = v.get_black(transformed)
            align = v.draw_green_point_grid(transformed, size)
            identified_black = v.color_identified_grid_greyscale(black, True, size)
            identified_white = v.color_identified_grid_greyscale(white, False, size)

            changes = updater.get_changed_position()
            if analyzing and evaluator.is_valid_board(changes):
                evaluator.update_board(changes)
                evaluator.print_last_move()
                final_board = v.drawn_board(transformed)
            # cv2.imshow('transformed', transformed)
            grid = v.window_for_black_white(white, identified_white, black, identified_black)
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
                print(evaluator.game_history)
                print(f"white: {evaluator.white_stones}, captured: {evaluator.black_captured}")
                print(f"black: {evaluator.black_stones}, captured: {evaluator.white_captured}")
                converter.create_file('game.sgf')
            else:
                print('start analysis')
            analyzing = not analyzing
        if key == ord('r'):
            POINTS = [[200, 200], [200, 400], [400, 400], [400, 200]]
        if key == ord('q'):
            break

    # cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture = WindowsCapture()
    # print(capture.get_window_names())
    main()
