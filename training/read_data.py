import pandas as pd
import numpy as np


# Prepare sequences
def create_sequences(data, targets, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        end_idx = i + time_steps
        X.append(data[i:end_idx])
        y.append(targets[end_idx])
    return np.array(X), np.array(y)


def load_from_csv(csv_data_path, time_steps):
    df = pd.read_csv(csv_data_path)

    train_data = []
    target_data = []

    # Dynamically find the number of landmarks from the column names
    landmark_cols = [col for col in df.columns if 'landmark' in col]
    num_landmarks = len(landmark_cols) // 3

    for _, row in df.iterrows():
        # --- CORRECTED THIS PART ---
        # Extract flattened landmarks directly, which is what the model expects.
        # The .reshape() part was a bug I introduced, it has been removed.
        landmarks = np.array([
                                 row[f'landmark_{i}_x'] for i in range(num_landmarks)] +
                             [row[f'landmark_{i}_y'] for i in range(num_landmarks)] +
                             [row[f'landmark_{i}_z'] for i in range(num_landmarks)]
                             )

        screen_point = np.array((row['screen_x'], row['screen_y']))
        train_data.append(landmarks)
        target_data.append(screen_point)

    X, y = create_sequences(train_data, target_data, time_steps)
    return X, y
