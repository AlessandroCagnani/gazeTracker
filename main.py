import pygame

from gaze_estimation.gaze_estimator.common import Camera
from gaze_estimation.gaze_estimator import GazeEstimator
import cv2
import numpy as np
from typing import Tuple
from utils.visualizer import Visualizer
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
class Demo:
    def __init__(self, mode=0, user_calib = None):
        self.mode = mode
        self.camera = Camera("data/camera/camera_params.yaml")
        self.gaze_estimator = GazeEstimator(self.camera, True)

        self.ref_points = [(1792 // 2, 1120 // 2),
                          (1792 // 5, (1120 // 5) * 2),
                          (1792 // 5, (1120 // 5) * 3),
                          (1792 // 5 * 2, 1120 // 5),
                          (1792 // 5 * 3, 1120 // 5),
                          (1792 // 5 * 2, 1120 // 5 * 4),
                          (1792 // 5 * 3, 1120 // 5 * 4),
                          (1792 // 5 * 4, (1120 // 5) * 2),
                          (1792 // 5 * 4, (1120 // 5) * 3)]

        if not mode == 1:
            pygame.init()
            screen = pygame.display.set_mode((1792, 1120), pygame.FULLSCREEN)
            self.visualizer = Visualizer(screen)
            if mode == 2:
                if user_calib is None:
                    print("[ TRACKER ] calibration file not provided, might be inaccurate")
                    self.calib_data = None
                else:
                    self.calib_data = self.get_calib_data(user_calib)
        else:
            self.visualizer = Visualizer()

        self.show_ref = False
        self.show_error_vectors = False
        self.show_correct_gaze = False
        self.run()



    def run(self):
        if self.mode == 0:
            ret = self.calib_routine()
            print("[ CALIBRATION ] name of the user: ")
            name = input()

            for point_ref in ret.keys():
                ret[point_ref]["mean"] = np.mean(ret[point_ref]["data"], axis=0).tolist()

            calib_data = list()
            for point_ref in ret.keys():
                point_coord = point_ref.split("_")
                point_coord = (int(point_coord[0]), int(point_coord[1]))
                calib_data.append(dict(point=point_coord, mean=ret[point_ref]["mean"], data=ret[point_ref]["data"]))

            with open(f"data/calib/{name}.json", "w") as f:
                json.dump(calib_data, f, cls=NpEncoder)

            print("[ CALIBRATION ] Calibration data saved")
        elif self.mode == 1:
            pass
        else:
            self.run_tracker()

    def run_tracker(self):

        print("[ TRACKER ] starting eye tracker ")

        while True:
            self.visualizer.fill_background((255, 255, 255))

            ret, frame = self.camera.get_frame()
            if not ret:
                continue

            undistorted = cv2.undistort(frame, self.camera.camera_matrix, self.camera.dist_coefficients)
            faces = self.gaze_estimator.detect_faces(undistorted)

            for face in faces:
                self.gaze_estimator.estimate_gaze(undistorted, face)
                print("face center: ", face.center)
                print("gaze vecotr: ", face.gaze_vector)
                print("distnce from camera: ", face.get_distance(self.camera))

                # face.center[2] = face.get_distance(self.camera)[1]

                if self.show_ref:
                    self.visualizer.draw_ref_points(self.ref_points)

                if self.show_error_vectors:
                    for calib in self.calib_data:
                        error_v = self.get_error_vector(calib["point"], calib["mean"])
                        self.visualizer.draw_arrow(calib["point"], calib["point"]+error_v, (255, 0, 0), 2)

                estimated_point = self.gaze_to_screen(face)

                correction_vector = np.asarray([0, 0])
                if self.show_correct_gaze:
                    correction_vector = self.idw_interpolation(estimated_point)

                x = int(estimated_point[0] + correction_vector[0])
                y = int(estimated_point[1] + correction_vector[1])

                self.visualizer.draw_point((x, y), (0, 0, 255), 10)

                pygame.display.update()
            pygame.time.delay(10)

                # print(polar_to_screen_coordinates(face.prediction))

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit()
                        return
                    elif event.key == pygame.K_e:
                        self.show_error_vectors = not self.show_error_vectors
                    elif event.key == pygame.K_c:
                        self.show_correct_gaze = not self.show_correct_gaze
                    elif event.key == pygame.K_r:
                        self.show_ref = not self.show_ref

    def calib_routine(self):

        calib_count = 0  #
        results = dict()
        ref_points = [(1792 // 2, 1120 // 2),
                      (1792 // 5, (1120 // 5) * 2),
                      (1792 // 5, (1120 // 5) * 3),
                      (1792 // 5 * 2, 1120 // 5),
                      (1792 // 5 * 3, 1120 // 5),
                      (1792 // 5 * 2, 1120 // 5 * 4),
                      (1792 // 5 * 3, 1120 // 5 * 4),
                      (1792 // 5 * 4, (1120 // 5) * 2),
                      (1792 // 5 * 4, (1120 // 5) * 3)]
        for ref_point in ref_points:
            point_name = f"{ref_point[0]}_{ref_point[1]}"
            results[point_name] = dict()
            results[point_name]["data"] = []
            results[point_name]["mean"] = None
        point_idx = 0

        while True:
            self.visualizer.fill_background((255, 255, 255))

            ret, frame = self.camera.get_frame()
            if not ret:
                continue

            undistorted = cv2.undistort(frame, self.camera.camera_matrix, self.camera.dist_coefficients)
            faces = self.gaze_estimator.detect_faces(undistorted)

            for face in faces:
                self.gaze_estimator.estimate_gaze(undistorted, face)
                print("face center: ", face.center)
                print("gaze vecotr: ", face.gaze_vector)

                self.visualizer.draw_ref_points(ref_points)
                #draw the point to look at
                self.visualizer.draw_point(ref_points[point_idx], (255, 0, 0), 10)

                self.visualizer.draw_text("Look at the red marker and press space - "
                                          "Press 'n' to calibrate next marker - "
                                          "Press 's' to save calibration data",
                                          (255, 0, 0),
                                          (10, 10))

                estimated_point = self.gaze_to_screen(face)

                self.visualizer.draw_point(estimated_point, (0, 0, 255), 10)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            pygame.quit()
                            exit()
                        elif event.key == pygame.K_SPACE:
                            if estimated_point is None:
                                print("[ CALIBRATION ] No gaze point estimated")
                                continue

                            ref_point = ref_points[point_idx]
                            point_name = f"{ref_point[0]}_{ref_point[1]}"
                            results[point_name]["data"].append(estimated_point)

                            self.visualizer.draw_point(estimated_point, (255, 255, 0), 10)
                        elif event.key == pygame.K_s and self.mode == 0:
                            pygame.quit()
                            return results
                        elif event.key == pygame.K_n:
                            if point_idx < len(ref_points) - 1:
                                point_idx += 1
                            else:
                                point_idx = 0


                pygame.display.update()
            pygame.time.delay(10)


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



if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--mode", type=int, default=0,
                            help="0 - calibrate, 1 - mirror, 2 - tracker")
    argparse.add_argument("--user_calib", type=str)

    args = argparse.parse_args()
    mode = args.mode
    demo = Demo(mode, args.user_calib)
    print("[ MAIN ] program end")
