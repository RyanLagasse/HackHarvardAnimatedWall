import cv2
import numpy as np

# Define the size of the grid
GRID_SIZE = 25
THRESHOLD = 150

# Initialize the camera
cap = cv2.VideoCapture(0)

# Get the width and height of the camera frame
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate the number of rows and columns in the grid
num_rows = height // GRID_SIZE
num_cols = width // GRID_SIZE

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the frame
    gray = cv2.resize(gray, (num_cols * GRID_SIZE, num_rows * GRID_SIZE))

    # Create an array to store the color of each square
    squares = np.zeros((num_rows, num_cols, 3))

    # Compute the average color of each square
    for i in range(num_rows):
        for j in range(num_cols):
            x = j * GRID_SIZE
            y = i * GRID_SIZE
            square = gray[y:y+GRID_SIZE, x:x+GRID_SIZE]
            avg_color = np.mean(square)

            # Check if the average color is dark
            if avg_color < THRESHOLD:
                # Set a shade of silver
                color = [192, 192, 192]
            else:
                # Set a shade of green
                color = [0, 255, 0]

            squares[i, j] = color

    # Display the grid of circles
    output = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(num_rows):
        for j in range(num_cols):
            x = j * GRID_SIZE + GRID_SIZE // 2
            y = i * GRID_SIZE + GRID_SIZE // 2
            color = tuple(map(int, squares[i, j]))
            cv2.circle(output, (x, y), GRID_SIZE // 2, color, -1)

    # Display the output
    cv2.imshow('output', output)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()