from gaze_estimation.gaze_estimator.common import Camera, visualizer
from gaze_estimation.gaze_estimator import GazeEstimator

import cv2
import numpy as np
from typing import Tuple
import json
from filterpy.kalman import KalmanFilter


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

        # TODO: move calib structure to another class
        self.calib_file = None
        # TODO: calib data = [{}], check if calib data < len(ref_points)
        self.calib_data = None
        self.current_ref_point = 0  # index of current reference point

        self.calib_record = None
        self.states = []


    def point_of_gaze(self, mode):

        ret, frame = self.camera.get_frame()
        if not ret:
            return None
        undistorted = cv2.undistort(frame, self.camera.camera_matrix, self.camera.dist_coefficients)

        states = []
        window_size = 5
        while len(states) < window_size:
            faces = self.gaze_estimator.detect_faces(undistorted)

            # TODO: if no face detected, return None
            # TODO: return or series of points or return only from biggest bbox
            faces.sort(key=lambda x: (x.bbox[1][0] - x.bbox[0][0]) * (x.bbox[1][1] - x.bbox[0][1]), reverse=True)
            biggest_face = faces[0] if len(faces) > 0 else None

            if biggest_face is None:
                continue

            self.gaze_estimator.estimate_gaze(undistorted, biggest_face)
            # print("face center: ", face.center)
            # print("gaze vecotr: ", face.gaze_vector)
            # print("distnce from camera: ", face.get_distance(self.camera))

            # face.center[2] = face.get_distance(self.camera)[1]
            estimated_point = self.gaze_to_screen(biggest_face)
            correction_vector = np.asarray([0, 0])
            if self.calib_data is not None and mode == 1:
                correction_vector = self.idw_interpolation(estimated_point)

            x = int(estimated_point[0] + correction_vector[0])
            y = int(estimated_point[1] + correction_vector[1])

            states.append([int(x), int(y)])

        self.kalman_filter_2d(states)
        # self.moving_average_filter_2d(states)

    def coord_dispatch(self):
        if len(self.states) == 0:
            self.point_of_gaze(1)
        x, y = self.states.pop(0)
        return x, y






    def display_self(self, config):
        # print("[ MODEL ] self display ")

        ret, frame = self.camera.get_frame()
        if not ret:
            return None

        self.visualizer.set_image(frame.copy())

        undistorted = cv2.undistort(frame, self.camera.camera_matrix, self.camera.dist_coefficients)
        faces = self.gaze_estimator.detect_faces(undistorted)

        for face in faces:
            self.gaze_estimator.estimate_gaze(undistorted, face)
            if config["bbox"]:
                self._draw_face_bbox(face)
            if config["head_pose"]:
                self._draw_head_pose(face)
            if config["landmarks"]:
                self._draw_face_template_model(face)
            # self._draw_landmarks(face)

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
            # print(f"Gaze point: {gaze_point}")
            # Check if gaze_point is within the screen boundaries
            x, y = get_point_on_screen((340, 220), (1792, 1120), gaze_point)
            # print(f"Screen coordinates: {x}, {y}")

            return x, y
        else:
            # print("Gaze vector is parallel to the screen")
            return None, None

    def moving_average_filter_2d(self, positions, kernel_size=3):
        # Convert the list of positions to a numpy array
        positions = np.array(positions, dtype=np.float32)

        # Reshape the array to be 2D with shape (n, 1, 2) for cv2.blur
        positions = positions.reshape(-1, 1, 2)

        # Apply the blur function to perform moving average
        filtered_positions = cv2.blur(positions, (kernel_size, 1))

        # Reshape the array back to (n, 2)
        filtered_positions = filtered_positions.reshape(-1, 2)

        self.states = filtered_positions.tolist()
    def kalman_filter_2d(self, measurements, process_noise=(1e-5, 1e-3), measurement_noise=1e-1):
        # Initialize the Kalman filter
        measurements = np.array(measurements, np.float32)
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 0, 1, 0]], np.float32)

        kf.transitionMatrix = np.array([[1, 1, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 1],
                                        [0, 0, 0, 1]], np.float32)

        kf.processNoiseCov = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]], np.float32) * process_noise[0]

        kf.measurementNoiseCov = np.array([[1, 0],
                                           [0, 1]], np.float32) * measurement_noise

        kf.errorCovPost = np.eye(4).astype(np.float32)
        kf.statePost = np.array([measurements[0][0], 0, measurements[0][1], 0], dtype=np.float32)

        states = []
        for measurement in measurements:
            # predict the state
            predicted = kf.predict()
            # correct the state with the measurement
            corrected = kf.correct(np.array(measurement, np.float32))
            states.append(corrected)


        self.states = [np.array([int(x[0]), int(x[2])]) for x in states]


    def init_calib_record(self):
        self.calib_record = dict()
        for ref_point in self.ref_points:
            point_name = f"{ref_point[0]}_{ref_point[1]}"
            self.calib_record[point_name] = dict()
            self.calib_record[point_name]["data"] = []
            self.calib_record[point_name]["mean"] = None

    def register_point(self, point):
        ref_point = self.ref_points[self.current_ref_point]
        point_name = f"{ref_point[0]}_{ref_point[1]}"
        self.calib_record[point_name]["data"].append(point)

    def calib_next_point(self):
        self.current_ref_point += 1
        self.current_ref_point %= len(self.ref_points)

    def save_calib(self):
        for point_ref in self.calib_record.keys():
            self.calib_record[point_ref]["mean"] = np.mean(self.calib_record[point_ref]["data"],
                                                           axis=0).tolist()

        calib_data = list()
        for point_ref in self.calib_record.keys():
            point_coord = point_ref.split("_")
            point_coord = (int(point_coord[0]), int(point_coord[1]))
            calib_data.append(dict(point=point_coord,
                                   mean=self.calib_record[point_ref]["mean"],
                                   data=self.calib_record[point_ref]["data"]))

        with open(f"{self.calib_file}.json", "w") as f:
            json.dump(calib_data, f)

    def set_calib_file(self, calib_file):
        self.calib_file = calib_file
        self.get_calib_data()

    def get_calib_data(self):
        if self.calib_file is None:
            self.calib_data = None
            return

        with open(self.calib_file, "r") as f:
            data = json.load(f)
        self.calib_data = data

    def set_file(self, filename):
        self.calib_file = filename

    # def write_calib_point(self):



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

    def euclidean_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def get_error_vector(self, true_point, estimated_point):
        # magnitude = np.linalg.norm(np.array(estimated_point) - np.array(true_point))
        # normalized = (np.array(estimated_point) - np.array(true_point)) / magnitude
        return np.array(estimated_point) - np.array(true_point)

    def get_correection_vector(self, true_point, estimated_point):
        # magnitude = np.linalg.norm(np.array(true_point) - np.array(estimated_point))
        # normalized = (np.array(true_point) - np.array(estimated_point)) / magnitude
        return np.array(true_point) - np.array(estimated_point)


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

