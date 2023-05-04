from typing import Optional

import numpy as np

from .eye import Eye
from .face_parts import FaceParts, FacePartsName


class Face(FaceParts):
    def __init__(self, bbox: np.ndarray, landmarks: np.ndarray):
        super().__init__(FacePartsName.FACE)
        self.bbox = bbox
        self.landmarks = landmarks

        self.reye: Eye = Eye(FacePartsName.REYE)
        self.leye: Eye = Eye(FacePartsName.LEYE)

        self.head_position: Optional[np.ndarray] = None
        self.model3d: Optional[np.ndarray] = None

    @staticmethod
    def change_coordinate_system(euler_angles: np.ndarray) -> np.ndarray:
        return euler_angles * np.array([-1, 1, -1])

    def get_distance(self, camera):
        pd = np.linalg.norm(self.reye.center - self.leye.center)
        avarage_pd = 63.0  # average distance between pupils in mm

        left_pupil = np.array([self.landmarks[36][0], self.landmarks[36][1]])
        right_pupil = np.array([self.landmarks[45][0], self.landmarks[45][1]])

        # Calculate the pixel distance between the pupils
        pixel_distance = np.linalg.norm(right_pupil - left_pupil)
        # Use the average real-world distance between pupils (63mm)
        real_distance = 0.063

        # Calculate the focal length in pixels
        focal_length_pixel = (camera.camera_matrix[0, 0] + camera.camera_matrix[1, 1]) / 2

        # Apply the pinhole camera model to estimate the distance
        distance = (focal_length_pixel * pd) / pixel_distance

        return pd, distance
