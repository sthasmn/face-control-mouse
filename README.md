# Face Control Mouse

An assistive technology tool that allows you to control your computer's mouse pointer and perform clicks using only your head movements and eye blinks, powered by a standard webcam and a custom-trained neural network.

**Note:** The current version provides mouse control functions specifically for **macOS**.

---

## Features

-   **Gaze-Based Pointer Control:** A trained neural network estimates where you are looking on the screen to position the mouse cursor.
-   **Blink to Click:**
    -   Blinking your **left eye** performs a **left click**.
    -   Blinking your **right eye** performs a **right click**.
-   **Complete Training Pipeline:** Includes scripts to collect your own data, process it, and train the model from scratch.

---

## How It Works

The project is divided into a two-stage pipeline: **Training** and **Execution**.

1.  **Training Pipeline:**
    -   `training/training_data_collection.py`: Runs a calibration process where you follow a dot on the screen. It records your facial landmark data and the corresponding screen coordinates, saving them to a `.csv` file.
    -   `training/read_data.py`: Loads the collected CSV data and processes it into sequences suitable for training a time-series model.
    -   `training/mouse_trainer.py`: Builds and trains a Keras/TensorFlow model on the sequence data and saves the final trained model as `model/my_model.keras`.

2.  **Execution:**
    -   `main.py`: Loads the pre-trained `my_model.keras`.
    -   It uses `MediaPipe Face Mesh` to get real-time facial landmarks from your webcam.
    -   The landmarks are fed into the model to predict screen coordinates for the mouse.
    -   It calculates the Eye Aspect Ratio (EAR) to detect blinks for left and right clicks.
    -   It uses macOS-specific libraries to move the system's mouse and perform clicks.

---

## Setup and Installation

It is highly recommended to use a virtual environment.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/sthasmn/face-control-mouse.git
    cd face-control-mouse
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    # For Mac/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage Instructions

This is a two-step process. You must train the model first.

### Step 1: Train Your Custom Model

The model needs to be trained on your unique facial features and screen setup for the best accuracy.

1.  **Collect Training Data:**
    Run the data collection script from your terminal. It will open a full-screen window and show a moving dot. **Follow the dot with your eyes for the entire duration (default is 2 minutes).**
    ```bash
    python training/training_data_collection.py
    ```
    This will create a `test_training_data.csv` file in your root directory.

2.  **Train the Model:**
    Next, run the trainer script. This will load the `.csv` file, build the neural network, and start the training process.
    ```bash
    python training/mouse_trainer.py
    ```
    This process will take some time. Once it is finished, it will save your personalized model to `model/my_model.keras`.

### Step 2: Run the Face Control Mouse

Once your model is trained and saved, you can run the main application.

```bash
python main.py
```
Look at your screen, move your head to control the cursor, and blink your left or right eye to click! Press 'q' in the preview window to quit.

### Citation
If you use this project in your research or work, please cite it as follows.
```BibTex
@software{Shrestha_Face_Control_Mouse_2024,
  author = {Shrestha, Suman},
  title = {{Face Control Mouse}},
  url = {https://github.com/sthasmn/face-control-mouse.git},
  year = {2025}
}
```

### Technology Stack
- Python 3.8+
- TensorFlow/Keras: For building and training the neural network.
- OpenCV: For video capture.
- MediaPipe: For facial landmark detection.
- Pandas & NumPy: For data manipulation.
- PyObjC: For native mouse control on macOS.


