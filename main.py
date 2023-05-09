import cv2

from game_evaluator import GameEvaluator
from slider import Slider
from board_updater import BoardUpdater
from settings import WINDOW_SIZE
from visualizer import Visualizer

if __name__ == '__main__':

    slider = Slider()
    board_updater = BoardUpdater()
    visualizer = Visualizer(slider, board_updater)
    board_evaluator = GameEvaluator(board_updater)

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
            if analyzing and board_evaluator.is_valid_board(changes):
                print(changes)
                board_evaluator.update_board(changes)
                board_evaluator.print_last_move()
                final_board = visualizer.drawn_board(transformed)
            # cv2.imshow('transformed', transformed)
            grid = visualizer.window_for_black_white(white, identified_white, black, identified_black)
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
                print(board_evaluator.game_history)
                print(f"white: {board_evaluator.white_stones}, captured: {board_evaluator.black_captured}")
                print(f"black: {board_evaluator.black_stones}, captured: {board_evaluator.white_captured}")
            else:
                print('start analysis')
            analyzing = not analyzing
        if key == ord('r'):
            POINTS = [[200, 200], [200, 400], [400, 400], [400, 200]]
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
