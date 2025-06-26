from screeninfo import get_monitors
from read_data import load_from_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define training configuration here
TIME_STEPS = 5
print("model training will start shortly.")
# Load the collected calibration data
X, y = load_from_csv("training_data.csv", time_steps=TIME_STEPS)

monitor = get_monitors()[0]
# Extract the width and height
screen_width = monitor.width
screen_height = monitor.height
# Normalize screen points
y = y / [screen_width, screen_height]

# --- MODEL ARCHITECTURE ---
# Define the neural network with 'same' padding to prevent the dimension reduction error
model = Sequential([
    # The input shape should be (time_steps, num_features)
    Conv1D(32, kernel_size=3, activation='relu', padding='same', input_shape=(X.shape[1], X.shape[2])),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.1),

    Conv1D(64, kernel_size=3, activation='relu', padding='same'), # Added padding='same'
    BatchNormalization(),
    MaxPooling1D(pool_size=2), # This is okay now because of the padding above
    Dropout(0.5),

    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    Dense(64, activation='relu'),
    BatchNormalization(),

    Dense(2)  # Output layer for screen coordinates (x, y)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# --- CALLBACKS ---
# Stop training early if there is no improvement
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Save only the best model based on validation loss
model_checkpoint = ModelCheckpoint(
    filepath='model/my_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Train the model
history = model.fit(
    X,
    y,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluate the model (it will have the best weights restored by EarlyStopping)
loss, mae = model.evaluate(X, y)
print(f"Mean Absolute Error of the best model: {mae}")

print("\nBest model saved to model/my_model.keras")


def plot_training_history(history):
    """
    Plots the training and validation loss and MAE from a Keras history object.
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
