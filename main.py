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
        self.user_calib = user_calib
        self.camera = Camera("data/camera/camera_params.yaml")
        self.gaze_estimator = GazeEstimator(self.camera, True)

        if not mode == 1:
            pygame.init()
            screen = pygame.display.set_mode((1792, 1120), pygame.FULLSCREEN)
            self.visualizer = Visualizer(screen)
        else:
            self.visualizer = Visualizer()
        self.run()

        #TODO: add flag to see correction / error vectors

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
        off_x = 0
        off_y = 0
        if self.user_calib is not None:
            print("[ TRACKER ] getting calibration ")
            data = self.get_calib_data(self.user_calib)
            center = next(obj for obj in data if obj["point"][0] == 896 and obj["point"][1] == 560)
            off_x = 896 - center["mean"][0]
            off_y = 560 - center["mean"][1]

        ref_points = [(1792 // 2, 1120 // 2),
                      (1792 // 5, (1120 // 5) * 2),
                      (1792 // 5, (1120 // 5) * 3),
                      (1792 // 5 * 2, 1120 // 5),
                      (1792 // 5 * 3, 1120 // 5),
                      (1792 // 5 * 2, 1120 // 5 * 4),
                      (1792 // 5 * 3, 1120 // 5 * 4),
                      (1792 // 5 * 4, (1120 // 5) * 2),
                      (1792 // 5 * 4, (1120 // 5) * 3)]

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

                self.visualizer.draw_ref_points(ref_points)

                estimated_point = self.gaze_to_screen(face)

                corrected_point = (estimated_point[0] + off_x, estimated_point[1] + off_y)
                self.visualizer.draw_point(corrected_point, (0, 0, 255), 10)

                pygame.display.update()
            pygame.time.delay(10)

                # print(polar_to_screen_coordinates(face.prediction))

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit()
                        return

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
