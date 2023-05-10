import numpy as np

FILE = '../points.txt'

WINDOW_SIZE = 400
ROWS, COLS = 19, 19

MAX_SIZE_TILE = int(WINDOW_SIZE / 20)
HALF_MAX_SIZE = int(MAX_SIZE_TILE / 2)

POINTS = [[200, 200], [400, 200], [400, 400], [200, 400]]

COLOR_WHITE = (255, 255, 255)  # RGB value for white
COLOR_BLACK = (0, 0, 0)  # RGB value for black
COLOR_BROWN = (135, 184, 222)  # RGB value for brown


def save_points_to_file():
    with open(FILE, 'w') as file:
        for point in POINTS:
            file.write(f'{point[0]},{point[1]}\n')
    print('saved points')


def load_points_from_file():
    global POINTS
    POINTS = []
    with open(FILE, 'r') as file:
        for line in file:
            x, y = line.strip().split(',')
            POINTS.append((int(x), int(y)))


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
