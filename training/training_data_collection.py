import cv2
import mediapipe as mp
import numpy as np
import random
import time
from screeninfo import get_monitors
import pandas as pd


def data_extractor(coordinates):
    """Extracts landmark data from MediaPipe's result object."""
    for face_landmarks in coordinates:
        keypoints = []
        # Loop through all landmarks and print their 3D coordinates
        for landmark in face_landmarks.landmark:
            x = landmark.x
            y = landmark.y
            z = landmark.z
            xyz = [x, y, z]
            keypoints.append(xyz)
        return keypoints
    return []


# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5)

# Get screen dimensions
monitor = get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height

# Initialize the list to store calibration data
calibration_data = []


def show_full_screen_with_moving_dot(duration=120):
    """Shows a moving dot and collects landmark data."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    end_time = time.time() + duration
    start_x = random.randint(0, screen_width - 1)
    start_y = random.randint(0, screen_height - 1)

    print(f"Starting data collection for {duration} seconds. Please follow the dot. Press 'ESC' to stop early.")

    while time.time() < end_time:
        end_x = random.randint(0, screen_width - 1)
        end_y = random.randint(0, screen_height - 1)

        num_steps = 100  # Number of steps for the dot to move
        for step in range(num_steps):
            if time.time() >= end_time:
                break

            alpha = step / num_steps
            dot_x = int(start_x * (1 - alpha) + end_x * alpha)
            dot_y = int(start_y * (1 - alpha) + end_y * alpha)
            dot_position = (dot_x, dot_y)

            image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            cv2.circle(image, dot_position, 20, (255, 0, 255), -1)

            cv2.namedWindow('Calibration', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('Calibration', image)

            success, frame = cap.read()
            if not success:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                landmarks = np.array(data_extractor(results.multi_face_landmarks))  # <-- USES THE NEW LOCAL FUNCTION
                if landmarks.size > 0:
                    calibration_data.append((landmarks, np.array((dot_x, dot_y))))

            if cv2.waitKey(1) & 0xFF == 27:
                end_time = time.time()  # End the outer loop
                break

        start_x, start_y = end_x, end_y

    cap.release()
    cv2.destroyAllWindows()


# --- Main script execution ---
show_full_screen_with_moving_dot()

if not calibration_data:
    print("No data collected. Exiting.")
else:
    data = []
    for landmarks, screen_point in calibration_data:
        flattened_landmarks = landmarks.flatten()
        row = np.append(flattened_landmarks, screen_point)
        data.append(row)

    # Define column names
    # MediaPipe Face Mesh has 478 landmarks
    num_landmarks = 478
    landmark_cols = []
    for i in range(num_landmarks):
        landmark_cols.append(f'landmark_{i}_x')
        landmark_cols.append(f'landmark_{i}_y')
        landmark_cols.append(f'landmark_{i}_z')

    columns = landmark_cols + ['screen_x', 'screen_y']

    # Re-create the flattened data with the correct number of columns
    data_for_df = []
    for landmarks, screen_point in calibration_data:
        row = list(landmarks.flatten()) + list(screen_point)
        data_for_df.append(row)

    df = pd.DataFrame(data_for_df, columns=columns)

    # Save DataFrame to CSV
    df.to_csv('training_data.csv', index=False)
    print(f"\nTraining data saved to 'training_data.csv' with {len(df)} samples.")
