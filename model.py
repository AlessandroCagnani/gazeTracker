from gaze_estimation.gaze_estimator.common import Camera, visualizer
from gaze_estimation.gaze_estimator import GazeEstimator

import cv2
import numpy as np
from typing import Tuple
import json

import argparse

def get_point_on_screen(monitor_mm: Tuple[float, float], monitor_pixels: Tuple[float, float], result: np.ndarray) -> Tuple[int, int]:
    """
    Calculate point in screen in pixels.

    :param monitor_mm: dimensions of the monitor in mm
    :param monitor_pixels: dimensions of the monitor in pixels
    :param result: predicted point on the screen in mm
    :return: point in screen in pixels
    """
    result_x = result[0]
    result_x = -result_x + monitor_mm[0] / 2
    result_x = result_x * (monitor_pixels[0] / monitor_mm[0])

    result_y = result[1]
    result_y = result_y - 20  # 20 mm offset
    result_y = min(result_y, monitor_mm[1])
    result_y = result_y * (monitor_pixels[1] / monitor_mm[1])

    return tuple(np.asarray([result_x, result_y]).round().astype(int))

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# mode 0: calibration, mode 1: mirror, mode 2: tracker
class model:
    def __init__(self, mode=0, user_calib = None):
        self.camera = Camera("data/camera/camera_params.yaml")
        self.gaze_estimator = GazeEstimator(self.camera, True)

        self.visualizer = visualizer.Visualizer(self.camera)

        self.ref_points = [(1792 // 2, 1120 // 2),
                          (1792 // 5, (1120 // 5) * 2),
                          (1792 // 5, (1120 // 5) * 3),
                          (1792 // 5 * 2, 1120 // 5),
                          (1792 // 5 * 3, 1120 // 5),
                          (1792 // 5 * 2, 1120 // 5 * 4),
                          (1792 // 5 * 3, 1120 // 5 * 4),
                          (1792 // 5 * 4, (1120 // 5) * 2),
                          (1792 // 5 * 4, (1120 // 5) * 3)]

        self.calib_data = None

    def point_of_gaze(self):

        ret, frame = self.camera.get_frame()
        if not ret:
            return None
        undistorted = cv2.undistort(frame, self.camera.camera_matrix, self.camera.dist_coefficients)
        faces = self.gaze_estimator.detect_faces(undistorted)

        for face in faces:
            self.gaze_estimator.estimate_gaze(undistorted, face)
            # print("face center: ", face.center)
            # print("gaze vecotr: ", face.gaze_vector)
            # print("distnce from camera: ", face.get_distance(self.camera))

            # face.center[2] = face.get_distance(self.camera)[1]
            estimated_point = self.gaze_to_screen(face)
            correction_vector = np.asarray([0, 0])
            if self.calib_data is not None:
                correction_vector = self.idw_interpolation(estimated_point)

            x = int(estimated_point[0] + correction_vector[0])
            y = int(estimated_point[1] + correction_vector[1])

            return (x, y)

    def display_self(self):
        # print("[ MODEL ] self display ")

        ret, frame = self.camera.get_frame()
        if not ret:
            return None

        self.visualizer.set_image(frame.copy())

        undistorted = cv2.undistort(frame, self.camera.camera_matrix, self.camera.dist_coefficients)
        faces = self.gaze_estimator.detect_faces(undistorted)

        for face in faces:
            self.gaze_estimator.estimate_gaze(undistorted, face)
            self._draw_face_bbox(face)
            self._draw_head_pose(face)
            self._draw_landmarks(face)
            self._draw_face_template_model(face)
            self._draw_gaze_vector(face)

        return self.visualizer.get_image()



    def gaze_to_screen(self, face):
        plane_norm = np.array([0, 0, 1])

        d = np.dot(plane_norm, face.gaze_vector)

        face.center[2] = face.center[2] - 0.0009

        if d != 0:
            p_point = np.array([0, 0, 0])
            t = np.dot(p_point - face.leye.center * 1000, plane_norm) / d
            gaze_point = face.leye.center * 1000 + t * face.gaze_vector
            print(f"Gaze point: {gaze_point}")
            # Check if gaze_point is within the screen boundaries
            x, y = get_point_on_screen((340, 220), (1792, 1120), gaze_point)
            print(f"Screen coordinates: {x}, {y}")

            return x, y
        else:
            print("Gaze vector is parallel to the screen")
            return None, None

    def euclidean_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def idw_interpolation(self, estimated_point, power=2):
        errors = [self.get_correection_vector(data["point"], data["mean"]) for data in self.calib_data]

        weights = []
        for data in self.calib_data:
            distance = self.euclidean_distance(estimated_point, data["point"])
            if distance <= 10:
                # If the input point is exactly at a known point, return the corresponding error
                return self.get_correection_vector(data["point"], data["mean"])
            weight = 1 / (distance ** power)
            weights.append(weight)

        weights = np.array(weights) / np.sum(weights)

        interpolated_error = np.dot(weights, errors)
        return interpolated_error

    def get_error_vector(self, true_point, estimated_point):
        # magnitude = np.linalg.norm(np.array(estimated_point) - np.array(true_point))
        # normalized = (np.array(estimated_point) - np.array(true_point)) / magnitude
        return np.array(estimated_point) - np.array(true_point)

    def get_correection_vector(self, true_point, estimated_point):
        # magnitude = np.linalg.norm(np.array(true_point) - np.array(estimated_point))
        # normalized = (np.array(true_point) - np.array(estimated_point)) / magnitude
        return np.array(true_point) - np.array(estimated_point)

    def get_calib_data(self, name):
        with open(f"data/calib/{name}.json", "r") as f:
            data = json.load(f)
        return data

    def _draw_face_bbox(self, face):

        self.visualizer.draw_bbox(face.bbox)

    def _draw_head_pose(self, face):
        length = 0.05
        self.visualizer.draw_model_axes(face, length, lw=2)

    def _draw_landmarks(self, face):
        self.visualizer.draw_points(face.landmarks,
                                    color=(0, 255, 255),
                                    size=1)

    def _draw_face_template_model(self, face):
        self.visualizer.draw_3d_points(face.model3d,
                                       color=(255, 0, 525),
                                       size=1)

    def _draw_gaze_vector(self, face):
        length = 0.3
        self.visualizer.draw_3d_line(
            face.center, face.center + length * np.array([1,1,-1]) * face.gaze_vector)

