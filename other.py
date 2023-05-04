import cv2
import numpy as np


# Define the mouse callback function
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
            dst = np.float32([[0, 0], [600, 0], [600, 600], [0, 600]])
            M = cv2.getPerspectiveTransform(src, dst)
    elif event == cv2.EVENT_LBUTTONUP:
        # Stop dragging
        dragging = False


def draw_green_grid(img, grid_shape, size):
    h, w, _ = img.shape
    rows, cols = grid_shape
    dy, dx = (h - (h / 20)) / rows, (w - (w / 20)) / cols

    # create a new image with the same dimensions as the original image

    for i in range(rows):
        for j in range(cols):
            x = int(round(j * dx)) + 16
            y = int(round(i * dy)) + 16
            cv2.rectangle(img, (x, y), (x + size, y + size), color=(0, 255, 0), thickness=1)
    return img


def closest_color(input_color):
    brown = (97, 147, 177)
    white = (255, 255, 255)
    black = (0, 0, 0)
    colors = [brown, white, black]

    # Calculate the difference between the input color and each predefined color
    color_diffs = []
    for color in colors:
        r_diff = abs(input_color[0] - color[0])
        g_diff = abs(input_color[1] - color[1])
        b_diff = abs(input_color[2] - color[2])
        color_diffs.append((r_diff + g_diff + b_diff))

    # Return the predefined color with the smallest difference
    closest_color_index = color_diffs.index(min(color_diffs))
    return colors[closest_color_index]


def draw_grid(img, grid_shape, size):
    h, w, _ = img.shape
    rows, cols = grid_shape
    dy, dx = (h - (h / 20)) / rows, (w - (w / 20)) / cols

    # create a new image with the same dimensions as the original image
    new_img = np.zeros_like(img)
    # Draw a rectangle on the image with blue color and thickness 10
    cv2.rectangle(new_img, (0, 0), (h, w), (97, 147, 177), 1000)

    for i in range(rows):
        for j in range(cols):
            x = int(round(j * dx)) + 16
            y = int(round(i * dy)) + 16
            # calculate average color of image inside square
            square = img[y:y + size, x:x + size, :]
            average_color = np.mean(np.mean(square, axis=0), axis=0)

            # set rectangle color to average color
            color = tuple(average_color.astype(int).tolist())
            color = closest_color(color)

            # fill rectangle with new color
            cv2.rectangle(new_img, (x, y), (x + size, y + size), color=color, thickness=-1)
    return new_img


# Initialize the camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920 / 2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080 / 2)


def on_trackbar(val):
    # do something with the trackbar value
    pass


cv2.namedWindow('size')
cv2.createTrackbar('size', 'size', 0, 30, on_trackbar)

# Initialize the points and the transformation matrix
points = [(200, 200), (800, 200), (800, 450), (200, 450)]
selected_point = None
dragging = False
M = None

# Set the mouse callback function
cv2.namedWindow('camera')
cv2.setMouseCallback('camera', move_point)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Apply the perspective transformation on the frame
    if M is not None:
        transformed = cv2.warpPerspective(frame, M, (600, 600))
        size = cv2.getTrackbarPos('size', 'size')
        grid = draw_grid(transformed, (19, 19), size)
        draw_green_grid(transformed, (19, 19), size)
        # apply Canny edge detection with the current threshold values
        cv2.imshow('transformed', transformed)
        cv2.imshow('grid', grid)

    # Draw the lines between the points
    for i in range(len(points)):
        cv2.line(frame, points[i], points[(i + 1) % len(points)], (255, 0, 0), 2)

    # Draw the points on the frame
    for px, py in points:
        cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)

    # Display the frame on the screen
    cv2.imshow('camera', frame)

    # Check for key presses
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the camera and close the windows
cap.release()
cv2.destroyAllWindows()
