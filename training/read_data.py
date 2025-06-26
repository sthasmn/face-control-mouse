import pandas as pd
import numpy as np
from utility import configuration as config

# Prepare sequences
def create_sequences(data, targets, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        end_idx = i + time_steps
        X.append(data[i:end_idx])
        y.append(targets[end_idx])
    return np.array(X), np.array(y)


def load_from_csv(csv_data_path):
    # Load DataFrame from CSV
    df = pd.read_csv(csv_data_path)

    # Reconstruct calibration data
    calibration_data = []
    train_data = []
    target_data = []
    for _, row in df.iterrows():
        # Extract flattened landmarks and reshape back to (468, 3)
        landmarks = np.array([
            row[f'landmark_{i}_x'] for i in range(478)] +
            [row[f'landmark_{i}_y'] for i in range(478)] +
            [row[f'landmark_{i}_z'] for i in range(478)]
        )#.reshape(478, 3)
        # Extract screen coordinates
        screen_point = np.array((row['screen_x'], row['screen_y']))
        train_data.append(landmarks)
        target_data.append(screen_point)
        #calibration_data.append((landmarks, screen_point))
    time_steps = config.time_steps  # Example time_steps
    X, y = create_sequences(train_data, target_data, time_steps)
    return X, y

# Use example
# X, y = load_from_csv("training_data.csv")
# print(X.shape)
# print(y.shape)