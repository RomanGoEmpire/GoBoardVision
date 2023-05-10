import numpy as np


class Settings:

    def __init__(self):
        self.FILE = 'points.txt'

        self.WINDOW_SIZE = 400
        self.ROWS, self.COLS = 19, 19

        self.MAX_SIZE_TILE = int(self.WINDOW_SIZE / 20)
        self.HALF_MAX_SIZE = int(self.MAX_SIZE_TILE / 2)

        self.POINTS = [[200, 200], [400, 200], [400, 400], [200, 400]]

        self.COLOR_WHITE = (255, 255, 255)  # RGB value for white
        self.COLOR_BLACK = (0, 0, 0)  # RGB value for black
        self.COLOR_BROWN = (135, 184, 222)  # RGB value for brown
        self.ALL_GRID_POINTS = None

    def initialize(self):
        self.ALL_GRID_POINTS = self.calculate_positions()
        self.load_points_from_file()

    def save_points_to_file(self):
        with open(self.FILE, 'w') as file:
            for point in self.POINTS:
                file.write(f'{point[0]},{point[1]}\n')
        print('saved points')

    def load_points_from_file(self):
        self.POINTS = []
        with open(self.FILE, 'r') as file:
            for line in file:
                x, y = line.strip().split(',')
                self.POINTS.append((int(x), int(y)))

    def calculate_positions(self):
        all_sizes = np.zeros((self.MAX_SIZE_TILE + 1, self.ROWS, self.COLS, 2), dtype='int16')
        h, w = self.WINDOW_SIZE, self.WINDOW_SIZE
        dy, dx = int((h - (h / 20)) / self.ROWS), int((w - (w / 20)) / self.COLS)
        for i in range(self.MAX_SIZE_TILE + 1):
            offset = self.HALF_MAX_SIZE - int(round(i / 2))
            for j in range(self.ROWS):
                for k in range(self.COLS):
                    all_sizes[i][j][k][0] = k * dx + self.HALF_MAX_SIZE + offset
                    all_sizes[i][j][k][1] = j * dy + self.HALF_MAX_SIZE + offset
        return all_sizes
