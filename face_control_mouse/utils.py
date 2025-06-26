import numpy as np
from collections import deque


def data_extractor(coordinates):
    """Extracts landmark data from MediaPipe's result object."""
    for face_landmarks in coordinates:
        keypoints = []
        for landmark in face_landmarks.landmark:
            keypoints.append([landmark.x, landmark.y, landmark.z])
        return keypoints
    return []


def euclidean_distance(p1, p2):
    """Calculates the Euclidean distance between two points in 2D."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


class SmoothPointerEMA:
    """
    Applies Exponential Moving Average smoothing to a stream of points
    to reduce jitter.
    """

    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.last_point = None

    def smooth(self, point):
        if self.last_point is None:
            self.last_point = point
        else:
            smoothed_x = self.alpha * point[0] + (1 - self.alpha) * self.last_point[0]
            smoothed_y = self.alpha * point[1] + (1 - self.alpha) * self.last_point[1]
            self.last_point = (smoothed_x, smoothed_y)
        return int(self.last_point[0]), int(self.last_point[1])


def get_eye_aspect_ratio(eye_landmarks):
    """
    Calculates the Eye Aspect Ratio (EAR) for a single eye.
    The EAR is a measure of how open the eye is.
    """
    # Vertical distances
    ver_dist1 = euclidean_distance(eye_landmarks[1], eye_landmarks[5])
    ver_dist2 = euclidean_distance(eye_landmarks[2], eye_landmarks[4])
    # Horizontal distance
    hor_dist = euclidean_distance(eye_landmarks[0], eye_landmarks[3])

    # Avoid division by zero
    if hor_dist == 0:
        return 0.0

    ear = (ver_dist1 + ver_dist2) / (2.0 * hor_dist)
    return ear