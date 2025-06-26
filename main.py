from face_control_mouse import FaceMouseController
import os

if __name__ == '__main__':
    # Check if the model file exists before starting
    if not os.path.exists('model/my_model.keras'):
        print("Error: Model file not found at 'model/my_model.keras'")
        print("Please run the training pipeline first using the scripts in the 'training/' directory.")
    else:
        controller = FaceMouseController()
        controller.run()