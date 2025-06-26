import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from screeninfo import get_monitors

from . import utils
from . import mouse_controller_macos as mouse


class FaceMouseController:
    """
    Controls the mouse using facial landmarks detected via a webcam.
    """

    def __init__(self, model_path='model/my_model.keras', time_steps=5):
        # --- Configuration ---
        self.time_steps = time_steps
        self.smoother = utils.SmoothPointerEMA(alpha=0.15)
        self.blink_frames_required = 3
        self.click_cooldown = 1.0  # seconds

        # --- Screen and Mouse ---
        monitor = get_monitors()[0]
        self.screen_width = monitor.width
        self.screen_height = monitor.height
        self.mouse = mouse

        # --- MediaPipe Initialization ---
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        # --- Model Loading ---
        print(f"Loading gaze estimation model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")

        # --- State Variables ---
        self.input_data_sequence = np.zeros((self.time_steps, 1434))
        self.left_eye_history = []
        self.right_eye_history = []
        self.last_click_time = 0

        # --- Landmark indices for eyes ---
        self.LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]

    @tf.function
    def _predict_gaze(self, data):
        """TensorFlow function for optimized model prediction."""
        return self.model(data)

    def _process_frame(self, image_rgb):
        """Processes a single RGB frame to find face and predict gaze."""
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None, None

        landmarks_raw = utils.data_extractor(results.multi_face_landmarks)
        if not landmarks_raw:
            return None, None

        landmarks_np = np.array(landmarks_raw)

        # --- Gaze Prediction ---
        flattened_landmarks = landmarks_np.flatten()

        self.input_data_sequence = np.roll(self.input_data_sequence, -1, axis=0)
        self.input_data_sequence[-1, :] = flattened_landmarks

        model_input = np.expand_dims(self.input_data_sequence, axis=0)
        predicted_norm = self._predict_gaze(model_input)[0]

        # --- COORDINATE TRANSFORMATION ---
        # Denormalize the raw model output
        raw_x = predicted_norm[0] * self.screen_width
        raw_y = predicted_norm[1] * self.screen_height

        # Invert the Y-axis to map from OpenCV coords (top-left origin)
        # to macOS screen coords (bottom-left origin).
        transformed_y = self.screen_height - raw_y

        # Smooth the final transformed coordinates
        screen_pos = self.smoother.smooth((raw_x, transformed_y))

        return screen_pos, landmarks_np

    def _detect_blinks(self, landmarks):
        """Detects left and right eye blinks."""
        left_eye_points = np.array([[landmarks[i][0], landmarks[i][1]] for i in self.LEFT_EYE_LANDMARKS])
        right_eye_points = np.array([[landmarks[i][0], landmarks[i][1]] for i in self.RIGHT_EYE_LANDMARKS])

        left_ear = utils.get_eye_aspect_ratio(left_eye_points)
        right_ear = utils.get_eye_aspect_ratio(right_eye_points)

        EAR_THRESHOLD = 0.2
        is_left_closed = left_ear < EAR_THRESHOLD
        is_right_closed = right_ear < EAR_THRESHOLD

        self.left_eye_history.append(is_left_closed)
        self.right_eye_history.append(is_right_closed)
        if len(self.left_eye_history) > self.blink_frames_required:
            self.left_eye_history.pop(0)
            self.right_eye_history.pop(0)

        left_blink = all(self.left_eye_history)
        right_blink = all(self.right_eye_history)

        return left_blink, right_blink

    def run(self):
        """Starts the main loop for face mouse control."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            # --- FIX: REMOVED THE cv2.flip() FROM HERE ---
            # Process the original, unflipped image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # --- Core Logic ---
            screen_pos, landmarks = self._process_frame(image_rgb)

            if screen_pos and landmarks is not None:
                self.mouse.move(screen_pos[0], screen_pos[1])

                left_blink, right_blink = self._detect_blinks(landmarks)

                can_click = (cv2.getTickCount() - self.last_click_time) / cv2.getTickFrequency() > self.click_cooldown

                if can_click:
                    if left_blink and not right_blink:
                        print("Left Click!")
                        self.mouse.left_click(screen_pos[0], screen_pos[1])
                        self.last_click_time = cv2.getTickCount()
                    elif right_blink and not left_blink:
                        print("Right Click!")
                        self.mouse.right_click(screen_pos[0], screen_pos[1])
                        self.last_click_time = cv2.getTickCount()

            # For a mirrored preview, flip the image *after* all processing is done
            preview_image = cv2.flip(image, 1)
            cv2.imshow('Face Control Mouse - Preview (Press Q to quit)', preview_image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        self.face_mesh.close()
        cv2.destroyAllWindows()
