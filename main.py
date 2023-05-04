import cv2


# Define the mouse callback function
def move_point(event, x, y, flags, param):
    global points, selected_point, dragging
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
        # Redraw the lines between the points
        for i in range(len(points)):
            cv2.line(frame, points[i], points[(i + 1) % len(points)], (255, 0, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        # Stop dragging
        dragging = False


# Initialize the camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920 / 2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080 / 2)

# Initialize the points
points = [(50, 50), (50, 200), (200, 50), (200, 200)]
selected_point = None
dragging = False

# Set the mouse callback function
cv2.namedWindow('camera')
cv2.setMouseCallback('camera', move_point)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Draw the lines between the points
    for i in range(len(points)):
        cv2.line(frame, points[i], points[(i + 1) % len(points)], (255, 0, 0), 2)

    # Draw the points on the frame
    for px, py in points:
        cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow('camera', frame)

    # Check for key presses
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
