import cv2
import numpy as np

points = [(200, 200), (800, 200), (800, 450), (200, 450)]
points_transformed = [[0, 0], [600, 0], [600, 600], [0, 600]]
selected_point = None
dragging = False
M = None


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
        cv2.line(frame, points[i], points[(i + 1) % len(points)], (255, 0, 0), 2)
    for px, py in points:
        cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)


if __name__ == '__main__':

    cap = initialize_cam()

    # Set the mouse callback function
    cv2.namedWindow('camera')
    cv2.setMouseCallback('camera', move_point)
    transformed = None
    while True:

        ret, frame = cap.read()

        if M is not None:
            transformed = cv2.warpPerspective(frame, M, (600, 600))
            cv2.imshow('transformed', transformed)

        draw_rectangle_with_corner_points()

        cv2.imshow('camera', frame)

        # Check for key presses
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
