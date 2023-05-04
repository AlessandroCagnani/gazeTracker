from typing import List

import dlib
import numpy as np
import yacs.config

from ..common import Face


class LandmarkEstimator:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            "data/dlib/shape_predictor_68_face_landmarks.dat")

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        return self._detect_faces_dlib(image)

    def _detect_faces_dlib(self, image: np.ndarray) -> List[Face]:
        bboxes = self.detector(image[:, :, ::-1], 0)
        detected = []
        for bbox in bboxes:
            predictions = self.predictor(image[:, :, ::-1], bbox)
            landmarks = np.array([(pt.x, pt.y) for pt in predictions.parts()],
                                 dtype=np.float)
            bbox = np.array([[bbox.left(), bbox.top()],
                             [bbox.right(), bbox.bottom()]],
                            dtype=np.float)
            detected.append(Face(bbox, landmarks))
        return detected
