import cv2
import numpy as np


class Visualizer:

    def __init__(self, slider, board_updater, settings):
        self.board_updater = board_updater
        self.slider = slider
        self.s = settings
        self.points_transformed = [[0, 0], [self.s.WINDOW_SIZE, 0], [self.s.WINDOW_SIZE, self.s.WINDOW_SIZE],
                                   [0, self.s.WINDOW_SIZE]]

        self.selected_point = None
        self.dragging = False

        self.M = self.setM()

        self.color_circle = (0, 255, 0)
        self.color_line = (255, 0, 0)

        self.left_star_point = 3 * self.s.MAX_SIZE_TILE
        self.middle_star_point = 9 * self.s.MAX_SIZE_TILE
        self.right_star_point = 15 * self.s.MAX_SIZE_TILE

    @staticmethod
    def initialize_cam():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920 / 2)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080 / 2)
        return cap

    def get_white(self, image):
        new_image = image.copy()
        gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        sharpened = cv2.addWeighted(gray, 1.5, gray_blur, -0.5, 0)
        # Adjust the brightness and contrast
        alpha = cv2.getTrackbarPos('white_alpha', 'Slider') / 10
        beta = cv2.getTrackbarPos('white_beta', 'Slider')
        adjusted_img = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=beta)
        ret, thresh = cv2.threshold(adjusted_img, self.slider.white_threshold, 255, cv2.THRESH_BINARY)
        thresh = np.invert(thresh)
        adjusted_img[thresh == 255] = 0
        return adjusted_img

    def get_black(self, image):
        new_image = image.copy()
        gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        sharpened = cv2.addWeighted(gray, 1.5, gray_blur, -0.5, 0)

        # Adjust the brightness and contrast
        alpha = cv2.getTrackbarPos('black_alpha', 'Slider') / 10
        beta = cv2.getTrackbarPos('black_beta', 'Slider')
        adjusted_img = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=beta)
        ret, thresh = cv2.threshold(adjusted_img, self.slider.black_threshold, 255, cv2.THRESH_BINARY)
        thresh = np.invert(thresh)
        adjusted_img[thresh == 255] = 0
        return adjusted_img

    @staticmethod
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

    def move_point(self, event, x, y, flags, param):
        # Declare global variables
        # Check if the left mouse button is clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            # Loop through points and check if the click is near a point
            for i, (px, py) in enumerate(self.s.POINTS):
                if abs(x - px) < 10 and abs(y - py) < 10:
                    self.selected_point = i  # Set the selected point index
                    self.dragging = True  # Enable dragging
                    break
        # Check if the mouse is moving and dragging is enabled
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            # Update the selected point's coordinates
            self.s.POINTS[self.selected_point] = (x, y)
            src = np.float32(self.s.POINTS)  # Convert points to float32
            dst = np.float32(self.points_transformed)  # Convert transformed points to float32
            self.M = cv2.getPerspectiveTransform(src, dst)  # Calculate the perspective transform matrix
        # Check if the left mouse button is released
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False  # Disable dragging

    def draw_rectangle_with_corner_points(self, frame):
        for i in range(len(self.s.POINTS)):
            cv2.line(frame, self.s.POINTS[i], self.s.POINTS[(i + 1) % len(self.s.POINTS)], self.color_line, 1)
        for px, py in self.s.POINTS:
            cv2.circle(frame, (px, py), 3, self.color_circle, -1)

    def draw_green_point_grid(self, img, size):
        new_image = img.copy()
        for row in range(self.s.COLS):
            for col in range(self.s.COLS):
                x = self.s.ALL_GRID_POINTS[size][row][col][0]
                y = self.s.ALL_GRID_POINTS[size][row][col][1]
                cv2.rectangle(new_image, (x, y), (x + size, y + size), color=(0, 255, 0), thickness=1)
        return new_image

    def color_identified_grid_greyscale(self, img, is_black, size):
        color = self.color_circle
        new_img = np.zeros_like(img)

        cv2.rectangle(new_img, (0, 0), (self.s.WINDOW_SIZE, self.s.WINDOW_SIZE), (97, 147, 177), -1)

        for row in range(self.s.COLS):
            for col in range(self.s.COLS):
                x = self.s.ALL_GRID_POINTS[size][row][col][0]
                y = self.s.ALL_GRID_POINTS[size][row][col][1]
                max_x = self.s.ALL_GRID_POINTS[self.s.MAX_SIZE_TILE][row][col][0]
                max_y = self.s.ALL_GRID_POINTS[self.s.MAX_SIZE_TILE][row][col][1]

                if size != 0:
                    square = img[y:y + size, x:x + size]
                    average_color = np.mean(np.mean(square, axis=0), axis=0)
                    color = average_color.astype(int).tolist()
                    color = self.closest_color(color)

                    if color == 255:
                        self.board_updater.recording_board[row][col] = 1 if is_black else 2

                cv2.rectangle(new_img, (max_x, max_y), (max_x + self.s.MAX_SIZE_TILE, max_y + self.s.MAX_SIZE_TILE),
                              color=color,
                              thickness=-1)
                cv2.rectangle(new_img, (max_x, max_y), (max_x + self.s.MAX_SIZE_TILE, max_y + self.s.MAX_SIZE_TILE),
                              color=127,
                              thickness=1)
        return new_img

    @staticmethod
    def closest_color(color):
        return 255 if color >= 127 else 0

    def get_board_grid(self, img):
        for i in range(self.s.COLS + 1):
            distance = i * self.s.MAX_SIZE_TILE
            cv2.line(img, (self.s.MAX_SIZE_TILE, distance), (self.s.WINDOW_SIZE - self.s.MAX_SIZE_TILE, distance),
                     self.s.COLOR_BLACK, 1)
            cv2.line(img, (distance, self.s.MAX_SIZE_TILE), (distance, self.s.WINDOW_SIZE - self.s.MAX_SIZE_TILE),
                     self.s.COLOR_BLACK, 1)
            if i == 3:
                cv2.circle(img,
                           (self.left_star_point + self.s.MAX_SIZE_TILE, self.left_star_point + self.s.MAX_SIZE_TILE),
                           3,
                           self.s.COLOR_BLACK, -1)
                cv2.circle(img,
                           (self.left_star_point + self.s.MAX_SIZE_TILE, self.middle_star_point + self.s.MAX_SIZE_TILE),
                           3,
                           self.s.COLOR_BLACK,
                           -1)
                cv2.circle(img,
                           (self.left_star_point + self.s.MAX_SIZE_TILE, self.right_star_point + self.s.MAX_SIZE_TILE),
                           3,
                           self.s.COLOR_BLACK, -1)
            if i == 9:
                cv2.circle(img,
                           (self.middle_star_point + self.s.MAX_SIZE_TILE, self.left_star_point + self.s.MAX_SIZE_TILE),
                           3,
                           self.s.COLOR_BLACK,
                           -1)
                cv2.circle(img, (
                self.middle_star_point + self.s.MAX_SIZE_TILE, self.middle_star_point + self.s.MAX_SIZE_TILE), 3,
                           self.s.COLOR_BLACK,
                           -1)
                cv2.circle(img, (
                self.middle_star_point + self.s.MAX_SIZE_TILE, self.right_star_point + self.s.MAX_SIZE_TILE), 3,
                           self.s.COLOR_BLACK,
                           -1)
            if i == 15:
                cv2.circle(img,
                           (self.right_star_point + self.s.MAX_SIZE_TILE, self.left_star_point + self.s.MAX_SIZE_TILE),
                           3,
                           self.s.COLOR_BLACK, -1)
                cv2.circle(img, (
                self.right_star_point + self.s.MAX_SIZE_TILE, self.middle_star_point + self.s.MAX_SIZE_TILE), 3,
                           self.s.COLOR_BLACK,
                           -1)
                cv2.circle(img,
                           (self.right_star_point + self.s.MAX_SIZE_TILE, self.right_star_point + self.s.MAX_SIZE_TILE),
                           3,
                           self.s.COLOR_BLACK,
                           -1)
        return img

    def drawn_board(self, img):
        new_img = np.zeros_like(img)

        cv2.rectangle(new_img, (0, 0), (self.s.WINDOW_SIZE, self.s.WINDOW_SIZE), self.s.COLOR_BROWN, -1)

        new_img = self.get_board_grid(new_img)

        for row in range(19):
            for col in range(19):
                if self.board_updater.recording_board[row][col] == 1:
                    board_color = self.s.COLOR_BLACK
                elif self.board_updater.recording_board[row][col] == 2:
                    board_color = self.s.COLOR_WHITE
                else:
                    continue
                max_x = self.s.ALL_GRID_POINTS[self.s.MAX_SIZE_TILE][row][col][0]
                max_y = self.s.ALL_GRID_POINTS[self.s.MAX_SIZE_TILE][row][col][1]
                cv2.circle(new_img, (max_x + self.s.HALF_MAX_SIZE, max_y + self.s.HALF_MAX_SIZE), self.s.HALF_MAX_SIZE,
                           board_color, -1)
        return new_img

    def setM(self):
        src = np.float32(self.s.POINTS)  # Convert points to float32
        dst = np.float32(self.points_transformed)  # Convert transformed points to float32
        M = cv2.getPerspectiveTransform(src, dst)  # Calculate the perspective transform matrix
        return M
