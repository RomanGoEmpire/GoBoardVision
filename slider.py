import cv2

from settings import MAX_SIZE_TILE


class Slider:

    def __init__(self):
        self.max_value = 255
        self.max_alpha = 40
        self.max_beta = 100

        self.size = 1
        self.white_threshold = 254
        self.black_threshold = 254
        self.white_alpha_value = 15
        self.white_beta_value = 0
        self.black_alpha_value = 15
        self.black_beta_value = 0

        self.slider = self.create_all_slider()

    def on_threshold(sef, val):
        # do something with the trackbar value
        pass

    def create_all_slider(self):
        slider = cv2.namedWindow('Slider')
        cv2.resizeWindow("Slider", 400, 400)
        cv2.createTrackbar('white_alpha', 'Slider', self.white_alpha_value, self.max_alpha, self.on_threshold)
        cv2.createTrackbar('white_beta', 'Slider', self.white_beta_value, self.max_beta, self.on_threshold)
        cv2.createTrackbar('black_alpha', 'Slider', self.black_alpha_value, self.max_alpha, self.on_threshold)
        cv2.createTrackbar('black_beta', 'Slider', self.black_beta_value, self.max_beta, self.on_threshold)
        cv2.createTrackbar('size', 'Slider', self.size, MAX_SIZE_TILE, self.on_threshold)
        return slider
