import cv2
import numpy as np

WINDOW_SIZE = 400
ROWS, COLS = 19, 19
POINTS = [(200, 200), (800, 200), (800, 450), (200, 450)]
POINT_TRANSFORMED = [[0, 0], [WINDOW_SIZE, 0], [WINDOW_SIZE, WINDOW_SIZE], [0, WINDOW_SIZE]]

color_circle = (0, 255, 0)
color_line = (255, 0, 0)

color_brown = (135, 184, 222)  # RGB value for brown
color_black = (0, 0, 0)  # RGB value for black
color_white = (255, 255, 255)  # RGB value for white

selected_point = None
dragging = False
M = None

MAX_SIZE_TILE = int(WINDOW_SIZE / 20)
print(MAX_SIZE_TILE)
HALF_MAX_SIZE = int(MAX_SIZE_TILE / 2)

max_value = 255
max_sharpen = 30
max_alpha = 40
max_beta = 100

size = 1
white_threshold = 254
black_threshold = 254
white_sharpen = 15
white_alpha_value = 15
white_beta_value = 0
black_sharpen = 15
black_alpha_value = 15
black_beta_value = 0

# black = 1, white = 2
initial_board = np.zeros((ROWS, COLS), dtype=np.uint8)
last_board = np.zeros((ROWS, COLS), dtype=np.uint8)
recording_board = initial_board.copy()
next_board = initial_board.copy()
frame_counter = 0

black_stones = 0
white_stones = 0
is_blacks_turn = True


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
    cv2.namedWindow('White Image', )
    cv2.resizeWindow("White Image", 400, 200)
    cv2.createTrackbar('threshold_white', 'White Image', white_threshold, max_value, on_threshold)
    cv2.createTrackbar('sharpen', 'White Image', white_sharpen, max_sharpen, on_threshold)
    cv2.createTrackbar('alpha', 'White Image', white_alpha_value, max_alpha, on_threshold)
    cv2.createTrackbar('beta', 'White Image', white_beta_value, max_beta, on_threshold)

    cv2.namedWindow('Black Image')
    cv2.resizeWindow("Black Image", 400, 200)
    cv2.createTrackbar('threshold_black', 'Black Image', black_threshold, max_value, on_threshold)
    cv2.createTrackbar('sharpen', 'Black Image', black_sharpen, max_sharpen, on_threshold)
    cv2.createTrackbar('alpha', 'Black Image', black_alpha_value, max_alpha, on_threshold)
    cv2.createTrackbar('beta', 'Black Image', black_beta_value, max_beta, on_threshold)

    cv2.namedWindow('Grid Image')
    cv2.resizeWindow("Grid Image", 400, 200)
    cv2.createTrackbar('size', 'Grid Image', size, MAX_SIZE_TILE, on_threshold)


def get_white(image):
    new_image = image.copy()
    gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    sharpen = cv2.getTrackbarPos('sharpen', 'White Image') / 10
    sharpened = cv2.addWeighted(gray, sharpen, gray_blur, -0.5, 0)

    # Adjust the brightness and contrast
    alpha = cv2.getTrackbarPos('alpha', 'White Image') / 10
    beta = cv2.getTrackbarPos('beta', 'White Image')
    adjusted_img = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=beta)
    low = cv2.getTrackbarPos('threshold_white', 'White Image')
    ret, thresh = cv2.threshold(adjusted_img, low, 255, cv2.THRESH_BINARY)
    thresh = np.invert(thresh)
    adjusted_img[thresh == 255] = 0
    return adjusted_img


def get_black(image):
    new_image = image.copy()
    gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sharpen = cv2.getTrackbarPos('sharpen', 'Black Image') / 10
    sharpened = cv2.addWeighted(gray, sharpen, gray_blur, -0.5, 0)

    # Adjust the brightness and contrast
    alpha = cv2.getTrackbarPos('alpha', 'Black Image') / 10
    beta = cv2.getTrackbarPos('beta', 'Black Image')
    adjusted_img = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=beta)

    low = cv2.getTrackbarPos('threshold_black', 'Black Image')

    ret, thresh = cv2.threshold(adjusted_img, low, 255, cv2.THRESH_BINARY)
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


def is_valid_board():
    global frame_counter, next_board
    if not np.array_equal(recording_board, next_board):
        frame_counter = 0
        next_board = recording_board.copy()
        return False
    if frame_counter < 10:
        frame_counter += 1
        return False
    return frame_counter == 10 and (is_blacks_turn and count_black() == black_stones + 1) or (
            not is_blacks_turn and count_white() == white_stones + 1)


def update_board():
    global black_stones, white_stones, is_blacks_turn, last_board
    if is_blacks_turn:
        black_stones += 1
        is_blacks_turn = False
    else:
        white_stones += 1
        is_blacks_turn = True
    get_last_move()
    last_board = next_board.copy()


def count_black():
    return np.count_nonzero(recording_board == 1)


def count_white():
    return np.count_nonzero(recording_board == 2)


def get_last_move():
    row, col = np.where(last_board != next_board)
    print(row - 1, col -1)


if __name__ == '__main__':

    cap = initialize_cam()
    create_all_slider()

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
            transformed = cv2.warpPerspective(frame, M, (WINDOW_SIZE, WINDOW_SIZE))
            white = get_white(transformed)
            black = get_black(transformed)
            align = draw_green_point_grid(transformed)
            identified_black = color_identifed_grid_greyscale(black, True)
            identified_white = color_identifed_grid_greyscale(white, False)

            size = cv2.getTrackbarPos('size', 'Grid Image')
            if analyzing and is_valid_board():
                update_board()
                final_board = drawn_board(transformed)
            # cv2.imshow('transformed', transformed)
            cv2.imshow('aligner', align)
            cv2.imshow('identified black', identified_black)
            cv2.imshow('identified white', identified_white)
            cv2.imshow('white', white)
            cv2.imshow('black', black)
            if final_board is not None:
                cv2.imshow("Final", final_board)

        cv2.imshow('camera', frame)

        # Check for key presses
        key = cv2.waitKey(1)

        if key == ord('s'):
            if analyzing:
                print('stop analysis')
            else:
                print('start analysis')
            analyzing = not analyzing

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
