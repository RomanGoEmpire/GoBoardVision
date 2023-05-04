import cv2
import numpy as np

max_value = 255
max_sharpen = 30
max_alpha = 40
max_beta = 100

white_threshold = 254
black_threshold = 254
white_sharpen = 15
white_alpha_value = 15
white_beta_value = 0
black_sharpen = 15
black_alpha_value = 15
black_beta_value = 0


def on_threshold():
    # do something with the trackbar value
    pass


cv2.namedWindow('White Image')
cv2.createTrackbar('threshold_white', 'White Image', white_threshold, max_value, on_threshold)
cv2.createTrackbar('sharpen', 'White Image', white_sharpen, max_sharpen, on_threshold)
cv2.createTrackbar('alpha', 'White Image', white_alpha_value, max_alpha, on_threshold)
cv2.createTrackbar('beta', 'White Image', white_beta_value, max_beta, on_threshold)

cv2.namedWindow('Black Image')
cv2.createTrackbar('threshold_black', 'Black Image', black_threshold, max_value, on_threshold)
cv2.createTrackbar('sharpen', 'Black Image', black_sharpen, max_sharpen, on_threshold)
cv2.createTrackbar('alpha', 'Black Image', black_alpha_value, max_alpha, on_threshold)
cv2.createTrackbar('beta', 'Black Image', black_beta_value, max_beta, on_threshold)


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


while True:
    image = cv2.imread('board.png')
    white = get_white(image)
    black = get_black(image)
    # combined = np.hstack((image,adjusted_img))
    cv2.imshow('Image', image)
    cv2.imshow('white', white)
    cv2.imshow('black', black)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cv2.destroyAllWindows()
