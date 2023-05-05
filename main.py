import cv2
import numpy as np

circle_color = (0, 255, 0)
line_color = (255, 0, 0)
grid_shape = (19, 19)

points = [(200, 200), (800, 200), (800, 450), (200, 450)]
transformed_size = 400
points_transformed = [[0, 0], [transformed_size, 0], [transformed_size, transformed_size], [0, transformed_size]]
size = 1
selected_point = None
dragging = False
M = None

max_size = int(transformed_size / 20)
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


def initialize_cam():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920 / 2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080 / 2)
    return cap


def move_point(event, x, y, flags, param):
    global points, selected_point, dragging, M
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the mouse click is inside one of the points
        for i, (px, py) in enumerate(points):
            if abs(x - px) < 10 and abs(y - py) < 10:
                selected_point = i
                dragging = True
                break
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        # Move the selected point
        points[selected_point] = (x, y)
        # Compute the perspective transformation matrix
        if len(points) == 4:
            src = np.float32(points)
            dst = np.float32(points_transformed)
            M = cv2.getPerspectiveTransform(src, dst)
    elif event == cv2.EVENT_LBUTTONUP:
        # Stop dragging
        dragging = False


def draw_rectangle_with_corner_points():
    for i in range(len(points)):
        cv2.line(frame, points[i], points[(i + 1) % len(points)], line_color, 2)
    for px, py in points:
        cv2.circle(frame, (px, py), 3, circle_color, -1)


def on_threshold(val):
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

cv2.namedWindow('Grid')
cv2.createTrackbar('size', 'Grid', size, max_size, on_threshold)


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


def draw_green_grid(img):
    new_image = img.copy()
    h, w, _ = img.shape
    offset = 15 - int(round(size / 2))
    rows, cols = grid_shape
    dy, dx = int((h - (h / 20)) / rows), int((w - (w / 20)) / cols)

    for i in range(rows):
        for j in range(cols):
            x = j * dx + 15 + offset
            y = i * dy + 15 + offset
            cv2.rectangle(new_image, (x, y), (x + size, y + size), color=(0, 255, 0), thickness=1)
    return new_image


def color_identifed_grid(img):
    h, w, _ = img.shape
    rows, cols = grid_shape
    dy, dx = int((h - (h / 20)) / rows), int((w - (w / 20)) / cols)

    # create a new image with the same dimensions as the original image
    new_img = np.zeros_like(img)
    # Draw a rectangle on the image with blue color and thickness 10
    cv2.rectangle(new_img, (0, 0), (h, w), (97, 147, 177), 1000)

    for i in range(rows):
        for j in range(cols):
            x = j * dx + 15
            y = i * dy + 15
            # calculate average color of image inside square
            square = img[y:y + size, x:x + size, :]
            average_color = np.mean(np.mean(square, axis=0), axis=0)

            # set rectangle color to average color
            color = tuple(average_color.astype(int).tolist())
            # fill rectangle with new color
            cv2.rectangle(new_img, (x, y), (x + size, y + size), color=color, thickness=-1)
    return new_img


if __name__ == '__main__':

    cap = initialize_cam()

    # Set the mouse callback function
    cv2.namedWindow('camera')
    cv2.setMouseCallback('camera', move_point)
    transformed = None
    while True:

        ret, frame = cap.read()

        if M is not None:
            transformed = cv2.warpPerspective(frame, M, (transformed_size, transformed_size))
            white = get_white(transformed)
            black = get_black(transformed)
            size = cv2.getTrackbarPos('size', 'Grid')
            align = draw_green_grid(transformed)
            identified = color_identifed_grid(transformed)
            #identified_black = color_identifed_grid(black)
            #identified_white = color_identifed_grid(white)
            # cv2.imshow('transformed', transformed)
            cv2.imshow('aligner', align)
            cv2.imshow('identified', identified)
            #cv2.imshow('identified black', identified_black)
            #cv2.imshow('identified white', identified_white)
            cv2.imshow('white', white)
            cv2.imshow('black', black)

        draw_rectangle_with_corner_points()

        cv2.imshow('camera', frame)

        # Check for key presses
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
