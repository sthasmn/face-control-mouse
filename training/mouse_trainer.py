from screeninfo import get_monitors
from utility import configuration as confg
from Training.read_data import load_from_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten, LSTM
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
response = input("Do you want take new data? (y/n): ").strip().lower()
if response == 'y':
    print("Proceeding with the operation...")
    import training_data_collection
    # Add the code to proceed here
elif response == 'n':
    print("Model will be trained with old data.")
else:
    print("Invalid input. proceeding with old data.")


print("model training will start shortly.")
# Load the collected calibration data

X, y = load_from_csv("test_training_data.csv")
time_steps = confg.time_steps
# X = data[:, :-2]  # Features: flattened landmarks
# y = data[:, -2:]  # Targets: screen coordinates
monitor = get_monitors()[0]
# Extract the width and height
screen_width = monitor.width
screen_height = monitor.height
# Normalize screen points
y = y / [screen_width, screen_height]

# Define a simple neural network
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),

    Conv1D(128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),

    # LSTM(64, activation='relu', input_shape=(time_steps, X.shape[2]), return_sequences=True),
    # BatchNormalization(),
    # Dropout(0.3),
    #
    # LSTM(128, activation='relu', return_sequences=False),
    # BatchNormalization(),
    # Dropout(0.3),

    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(128, activation='relu'),
    BatchNormalization(),
    #Dropout(0.3),

    Dense(2)  # Output layer for screen coordinates (x, y)
])

# Compile and train the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model
loss, mae = model.evaluate(X, y)
print(f"Mean Absolute Error: {mae}")
# Save the model for later use
model.save('model/my_model.keras')
print("\nModel saved to model/my_model.keras")


def plot_training_history(history):
    """
    Plots the training and validation loss and MAE from a Keras history object.

    Args:
        history: A Keras History object returned from model.fit()
    """

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')

    # Plot training & validation MAE values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.show()

plot_training_history(history)