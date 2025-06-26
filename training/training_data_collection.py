import cv2
import mediapipe as mp
import numpy as np
import random
import time
from screeninfo import get_monitors
from utility import utils_
import pandas as pd


# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Get screen dimensions
# Get the monitor information
monitor = get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height

# Initialize the list to store calibration data
calibration_data = []

# Define the number of points to collect
num_points = 10

def show_full_screen_with_dot():
    cap = cv2.VideoCapture(0)

    for _ in range(num_points):
        # Generate random screen coordinates for the dot
        dot_x = random.randint(0, screen_width)
        dot_y = random.randint(0, screen_height)
        dot_position = (dot_x, dot_y)

        # Create a black image
        image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

        # Draw a white dot at the random position
        cv2.circle(image, dot_position, 80, (255, 0, 255), -1)

        # Display the image in full screen
        cv2.namedWindow('Calibration', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        key = cv2.waitKey(1)
        cv2.imshow('Calibration', image)

        # Wait for a short time to allow the user to look at the dot
        time.sleep(1.5)

        # Capture the face landmarks
        success, frame = cap.read()
        if not success:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = np.array(utils_.data_extractor(results.multi_face_landmarks))
            # Store the landmarks and the corresponding screen coordinates
            calibration_data.append((landmarks, np.array((dot_x, dot_y))))

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def show_full_screen_with_moving_dot(duration=120):
    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # Calibration data list

    # Define the end time
    end_time = time.time() + duration
    start_x = random.randint(0, screen_width - 1)
    start_y = random.randint(0, screen_height - 1)
    while time.time() < end_time:
        # Generate random  end points for the dot
        end_x = random.randint(0, screen_width - 1)
        end_y = random.randint(0, screen_height - 1)

        # Move the dot from start to end
        num_steps = 100  # Number of steps for the dot to move
        for step in range(num_steps):
            # Calculate the current position of the dot
            alpha = step / num_steps
            dot_x = int(start_x * (1 - alpha) + end_x * alpha)
            dot_y = int(start_y * (1 - alpha) + end_y * alpha)
            dot_position = (dot_x, dot_y)

            # Create a black image
            image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

            # Draw a magenta dot at the current position
            cv2.circle(image, dot_position, 20, (255, 0, 255), -1)

            # Display the image in full screen
            cv2.namedWindow('Calibration', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('Calibration', image)

            # Capture the face landmarks
            success, frame = cap.read()
            if not success:
                print("Failed to capture image.")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                landmarks = np.array(utils_.data_extractor(results.multi_face_landmarks))
                # Store the landmarks and the corresponding screen coordinates
                calibration_data.append((landmarks, np.array((dot_x, dot_y))))

            # Wait for a short time to make the movement visible
            if cv2.waitKey(1) & 0xFF == 27:
                return
        # Ensure the loop runs smoothly with consistent timing
        start_x = end_x
        start_y = end_y
        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

#show_full_screen_with_dot()
show_full_screen_with_moving_dot()
data = []
for landmarks, screen_point in calibration_data:
    # print(landmarks.shape, screen_point.shape)
    # Flatten the landmarks matrix
    flattened_landmarks = landmarks.flatten()
    # Append screen coordinates
    row = np.append(flattened_landmarks, screen_point)
    data.append(row)
# Define column names
num_landmarks = 478
columns = [f'landmark_{i}_x' for i in range(num_landmarks)] + \
          [f'landmark_{i}_y' for i in range(num_landmarks)] + \
          [f'landmark_{i}_z' for i in range(num_landmarks)] + \
          ['screen_x', 'screen_y']

# Create DataFrame
df = pd.DataFrame(data, columns=columns)

# Save DataFrame to CSV
df.to_csv('test_training_data.csv', index=False)
print('Training_data_saved...')
# Save the collected data for later use
#np.save('calibration_data.npy', calibration_data)