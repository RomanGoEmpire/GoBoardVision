import cv2
import numpy as np

WINDOW_SIZE = 400
ROWS, COLS = 19, 19
FILE = 'points.txt'


def save_points_to_file():
    with open(FILE, 'w') as file:
        for point in POINTS:
            file.write(f'{point[0]},{point[1]}\n')
    print('saved points')


def load_points_from_file():
    points = []
    with open(FILE, 'r') as file:
        for line in file:
            x, y = line.strip().split(',')
            points.append((int(x), int(y)))
    return points


POINTS = load_points_from_file()

POINT_TRANSFORMED = [[0, 0], [WINDOW_SIZE, 0], [WINDOW_SIZE, WINDOW_SIZE], [0, WINDOW_SIZE]]
MAX_SIZE_TILE = int(WINDOW_SIZE / 20)
HALF_MAX_SIZE = int(MAX_SIZE_TILE / 2)
left_star_point = 3 * MAX_SIZE_TILE
middle_star_point = 9 * MAX_SIZE_TILE
right_star_point = 15 * MAX_SIZE_TILE
color_circle = (0, 255, 0)
color_line = (255, 0, 0)

color_brown = (135, 184, 222)  # RGB value for brown
color_black = (0, 0, 0)  # RGB value for black
color_white = (255, 255, 255)  # RGB value for white

selected_point = None
dragging = False
M = None

max_value = 255
max_alpha = 40
max_beta = 100

size = 1
white_threshold = 254
black_threshold = 254
white_alpha_value = 15
white_beta_value = 0
black_alpha_value = 15
black_beta_value = 0

# black = 1, white = 2
initial_board = np.zeros((ROWS, COLS))
last_board = np.zeros((ROWS, COLS))
recording_board = initial_board.copy()
next_board = initial_board.copy()
frame_counter = 0

black_stones = 0
black_captured = 0
white_stones = 0
white_captured = 0
is_blacks_turn = True

game_history = []


def initialize_cam():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920 / 2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080 / 2)
    return cap


def move_point(event, x, y, flags, param):
    # Declare global variables
    global POINTS, selected_point, dragging, M
    # Check if the left mouse button is clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # Loop through points and check if the click is near a point
        for i, (px, py) in enumerate(POINTS):
            if abs(x - px) < 10 and abs(y - py) < 10:
                selected_point = i  # Set the selected point index
                dragging = True  # Enable dragging
                break
    # Check if the mouse is moving and dragging is enabled
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        # Update the selected point's coordinates
        POINTS[selected_point] = (x, y)
        # Check if there are 4 points
        if len(POINTS) == 4:
            src = np.float32(POINTS)  # Convert points to float32
            dst = np.float32(POINT_TRANSFORMED)  # Convert transformed points to float32
            M = cv2.getPerspectiveTransform(src, dst)  # Calculate the perspective transform matrix
    # Check if the left mouse button is released
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False  # Disable dragging


def draw_rectangle_with_corner_points():
    for i in range(len(POINTS)):
        cv2.line(frame, POINTS[i], POINTS[(i + 1) % len(POINTS)], color_line, 1)
    for px, py in POINTS:
        cv2.circle(frame, (px, py), 5, color_circle, -1)


def on_threshold(val):
    # do something with the trackbar value
    pass


def calculate_positions():
    all_sizes = np.zeros((MAX_SIZE_TILE + 1, ROWS, COLS, 2), dtype='int16')
    h, w = WINDOW_SIZE, WINDOW_SIZE
    dy, dx = int((h - (h / 20)) / ROWS), int((w - (w / 20)) / COLS)
    for i in range(MAX_SIZE_TILE + 1):
        offset = HALF_MAX_SIZE - int(round(i / 2))
        for j in range(ROWS):
            for k in range(COLS):
                all_sizes[i][j][k][0] = k * dx + HALF_MAX_SIZE + offset
                all_sizes[i][j][k][1] = j * dy + HALF_MAX_SIZE + offset
    return all_sizes


ALL_GRID_POINTS = calculate_positions()


def create_all_slider():
    slider = cv2.namedWindow('Slider')
    cv2.resizeWindow("Slider", 400, 400)
    cv2.createTrackbar('white_alpha', 'Slider', white_alpha_value, max_alpha, on_threshold)
    cv2.createTrackbar('white_beta', 'Slider', white_beta_value, max_beta, on_threshold)
    cv2.createTrackbar('black_alpha', 'Slider', black_alpha_value, max_alpha, on_threshold)
    cv2.createTrackbar('black_beta', 'Slider', black_beta_value, max_beta, on_threshold)
    cv2.createTrackbar('size', 'Slider', size, MAX_SIZE_TILE, on_threshold)
    return slider


def get_white(image):
    new_image = image.copy()
    gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    sharpened = cv2.addWeighted(gray, 1.5, gray_blur, -0.5, 0)
    # Adjust the brightness and contrast
    alpha = cv2.getTrackbarPos('white_alpha', 'Slider') / 10
    beta = cv2.getTrackbarPos('white_beta', 'Slider')
    adjusted_img = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=beta)
    ret, thresh = cv2.threshold(adjusted_img, white_threshold, 255, cv2.THRESH_BINARY)
    thresh = np.invert(thresh)
    adjusted_img[thresh == 255] = 0
    return adjusted_img


def get_black(image):
    new_image = image.copy()
    gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sharpened = cv2.addWeighted(gray, 1.5, gray_blur, -0.5, 0)

    # Adjust the brightness and contrast
    alpha = cv2.getTrackbarPos('black_alpha', 'Slider') / 10
    beta = cv2.getTrackbarPos('black_beta', 'Slider')
    adjusted_img = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=beta)
    ret, thresh = cv2.threshold(adjusted_img, black_threshold, 255, cv2.THRESH_BINARY)
    thresh = np.invert(thresh)
    adjusted_img[thresh == 255] = 0
    return adjusted_img


def draw_green_point_grid(img):
    new_image = img.copy()
    for row in range(ROWS):
        for col in range(COLS):
            x = ALL_GRID_POINTS[size][row][col][0]
            y = ALL_GRID_POINTS[size][row][col][1]
            cv2.rectangle(new_image, (x, y), (x + size, y + size), color=(0, 255, 0), thickness=1)
    return new_image


def color_identifed_grid_greyscale(img, isBlack):
    color = color_circle
    new_img = np.zeros_like(img)

    cv2.rectangle(new_img, (0, 0), (WINDOW_SIZE, WINDOW_SIZE), (97, 147, 177), -1)

    for row in range(ROWS):
        for col in range(COLS):
            x = ALL_GRID_POINTS[size][row][col][0]
            y = ALL_GRID_POINTS[size][row][col][1]
            max_x = ALL_GRID_POINTS[MAX_SIZE_TILE][row][col][0]
            max_y = ALL_GRID_POINTS[MAX_SIZE_TILE][row][col][1]

            if size != 0:
                square = img[y:y + size, x:x + size]
                average_color = np.mean(np.mean(square, axis=0), axis=0)
                color = average_color.astype(int).tolist()
                color = closest_color(color)

                if color == 255:
                    recording_board[row][col] = 1 if isBlack else 2

            cv2.rectangle(new_img, (max_x, max_y), (max_x + MAX_SIZE_TILE, max_y + MAX_SIZE_TILE), color=color,
                          thickness=-1)
            cv2.rectangle(new_img, (max_x, max_y), (max_x + MAX_SIZE_TILE, max_y + MAX_SIZE_TILE), color=127,
                          thickness=1)
    return new_img


def closest_color(color):
    return 255 if color >= 127 else 0


def get_board_grid(img):
    for i in range(ROWS + 1):
        distance = i * MAX_SIZE_TILE
        cv2.line(img, (MAX_SIZE_TILE, distance), (WINDOW_SIZE - MAX_SIZE_TILE, distance),
                 color_black, 1)
        cv2.line(img, (distance, MAX_SIZE_TILE), (distance, WINDOW_SIZE - MAX_SIZE_TILE),
                 color_black, 1)
        if i == 3:
            cv2.circle(img, (left_star_point + MAX_SIZE_TILE, left_star_point + MAX_SIZE_TILE), 3, color_black, -1)
            cv2.circle(img, (left_star_point + MAX_SIZE_TILE, middle_star_point + MAX_SIZE_TILE), 3, color_black, -1)
            cv2.circle(img, (left_star_point + MAX_SIZE_TILE, right_star_point + MAX_SIZE_TILE), 3, color_black, -1)
        if i == 9:
            cv2.circle(img, (middle_star_point + MAX_SIZE_TILE, left_star_point + MAX_SIZE_TILE), 3, color_black, -1)
            cv2.circle(img, (middle_star_point + MAX_SIZE_TILE, middle_star_point + MAX_SIZE_TILE), 3, color_black,
                       -1)
            cv2.circle(img, (middle_star_point + MAX_SIZE_TILE, right_star_point + MAX_SIZE_TILE), 3, color_black,
                       -1)
        if i == 15:
            cv2.circle(img, (right_star_point + MAX_SIZE_TILE, left_star_point + MAX_SIZE_TILE), 3, color_black, -1)
            cv2.circle(img, (right_star_point + MAX_SIZE_TILE, middle_star_point + MAX_SIZE_TILE), 3, color_black,
                       -1)
            cv2.circle(img, (right_star_point + MAX_SIZE_TILE, right_star_point + MAX_SIZE_TILE), 3, color_black,
                       -1)

    return img


def drawn_board(img):
    new_img = np.zeros_like(img)

    cv2.rectangle(new_img, (0, 0), (WINDOW_SIZE, WINDOW_SIZE), color_brown, -1)

    new_img = get_board_grid(new_img)

    for row in range(19):
        for col in range(19):
            if recording_board[row][col] == 1:
                board_color = color_black
            elif recording_board[row][col] == 2:
                board_color = color_white
            else:
                continue
            max_x = ALL_GRID_POINTS[MAX_SIZE_TILE][row][col][0]
            max_y = ALL_GRID_POINTS[MAX_SIZE_TILE][row][col][1]
            cv2.circle(new_img, (max_x + HALF_MAX_SIZE, max_y + HALF_MAX_SIZE), HALF_MAX_SIZE, board_color, -1)
    return new_img


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
    global frame_counter, next_board

    if not np.array_equal(recording_board, next_board):
        frame_counter = 0
        next_board = recording_board.copy()
        return False

    if frame_counter < 10:
        frame_counter += 1
        return False

    if len(changes) == 1:
        is_valid_black_turn = is_blacks_turn and \
                              count_black_on_board() + black_captured == black_stones + 1 and \
                              count_white_on_board() + white_captured == white_stones
        is_valid_white_turn = not is_blacks_turn and \
                              count_white_on_board() + white_captured == white_stones + 1 and \
                              count_black_on_board() + black_captured == black_stones
        return is_valid_black_turn or is_valid_white_turn

    if len(changes) > 1:
        added, removed = get_added_and_removed(changes)
        if added == 1:
            is_valid_black_turn = is_blacks_turn and \
                                  count_black_on_board() + black_captured == black_stones + 1 and \
                                  count_white_on_board() + white_captured + removed == white_stones
            is_valid_white_turn = not is_blacks_turn and \
                                  count_white_on_board() + white_captured == white_stones + 1 and \
                                  count_black_on_board() + black_captured + removed == black_stones
            return is_valid_black_turn or is_valid_white_turn
    return False


def get_board_differences():
    return last_board - next_board


def get_change_at_position(position):
    difference_board = get_board_differences()
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
    last_board = next_board.copy()


def count_black_on_board():
    return np.count_nonzero(recording_board == 1)


def count_white_on_board():
    return np.count_nonzero(recording_board == 2)


def get_changed_position():
    return np.column_stack(np.where(last_board != next_board))


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
    # img1 = cv2.resize(img1, (0, 0), None, 0.5, 0.5)
    # img2 = cv2.resize(img2, (0, 0), None, 0.5, 0.5)
    # img3 = cv2.resize(img3, (0, 0), None, 0.5, 0.5)
    # img4 = cv2.resize(img4, (0, 0), None, 0.5, 0.5)

    # Concatenate the images into a 2x2 grid
    top_row = cv2.hconcat((white, black))
    bottom_row = cv2.hconcat((i_white, i_black))
    grid = cv2.vconcat((top_row, bottom_row))
    return grid


if __name__ == '__main__':

    cap = initialize_cam()
    slider = create_all_slider()

    # Set the mouse callback function
    cv2.namedWindow('camera')
    cv2.setMouseCallback('camera', move_point)

    analyzing = False

    transformed = None
    white = None
    black = None
    identified_black = None
    identified_white = None
    align = None
    final_board = None
    while True:
        recording_board = initial_board.copy()
        ret, frame = cap.read()
        draw_rectangle_with_corner_points()

        if M is not None:
            size = cv2.getTrackbarPos('size', 'Slider')
            transformed = cv2.warpPerspective(frame, M, (WINDOW_SIZE, WINDOW_SIZE))
            white = get_white(transformed)
            black = get_black(transformed)
            align = draw_green_point_grid(transformed)
            identified_black = color_identifed_grid_greyscale(black, True)
            identified_white = color_identifed_grid_greyscale(white, False)

            changes = get_changed_position()
            if analyzing and is_valid_board(changes):
                print(changes)
                update_board(changes)
                print_last_move()
                final_board = drawn_board(transformed)
            # cv2.imshow('transformed', transformed)
            cv2.imshow('align', align)

            grid = window_for_black_white(white, identified_white, black, identified_black)
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

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    save_points_to_file()
