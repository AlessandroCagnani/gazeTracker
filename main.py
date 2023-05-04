import pygame

from gaze_estimation.gaze_estimator.common import Camera
from gaze_estimation.gaze_estimator import GazeEstimator
import cv2
import numpy as np
from typing import Tuple
import os
import yaml

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


# mode 0: calibration, mode 1: mirror, mode 2: tracker
class Demo:
    def __init__(self, mode=0, user_calib = None):
        self.mode = mode
        self.user_calib = user_calib
        self.camera = Camera("data/camera/camera_params.yaml")
        self.gaze_estimator = GazeEstimator(self.camera, True)

    def run(self):
        if self.mode == 0:
            print("[ CALIBRATION ] name of the user: ")
            name = input()
            self.run_calibration(name)
            avg, data = self.get_calib_data(name)
            print("mean x and y: ", avg)
        elif self.mode == 1:
            pass
        else:
            self.run_tracker()

    def run_tracker(self):
        print("[ TRACKER ] starting eye tracker ")
        pygame.init()

        screen = pygame.display.set_mode((1792, 1120), pygame.FULLSCREEN)

        off_x = 0
        off_y = 0
        if self.user_calib is not None:
            print("[ TRACKER ] getting calibration ")
            avg, data = self.get_calib_data(self.user_calib)
            off_x = 896 - avg[0]
            off_y = 560 - avg[1]

        while True:
            screen.fill((255, 255, 255))

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

                pygame.draw.circle(screen, (0, 255, 0), (0, 0), 10)
                pygame.draw.circle(screen, (0, 255, 0), (1792 // 2, 1120 // 2), 10)
                pygame.draw.circle(screen, (0, 255, 0), (1792 // 4, 1120 // 4), 10)
                pygame.draw.circle(screen, (0, 255, 0), (1792 // 4 * 3, 1120 // 4), 10)
                pygame.draw.circle(screen, (0, 255, 0), (1792 // 4, 1120 // 4 * 3), 10)
                pygame.draw.circle(screen, (0, 255, 0), (1792 // 4 * 3, 1120 // 4 * 3), 10)
                pygame.draw.circle(screen, (0, 255, 0), (0, 1120), 10)
                pygame.draw.circle(screen, (0, 255, 0), (1792, 0), 10)
                pygame.draw.circle(screen, (0, 255, 0), (1792, 1120), 10)

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
                    pygame.draw.circle(screen, (255, 0, 0), (x + off_x, y + off_y), 10)
                else:
                    print("Gaze vector is parallel to the screen")

                pygame.display.update()
                pygame.time.delay(10)

                # print(polar_to_screen_coordinates(face.prediction))

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit()
                        return

    def run_calibration(self, name):
        print("[ CALIBRATION ] starting calibration ")

        pygame.init()

        screen = pygame.display.set_mode((1792, 1120), pygame.FULLSCREEN)

        calib_count = 0
        results = dict()

        while True:
            ret, frame = self.camera.get_frame()
            if not ret:
                continue

            undistorted = cv2.undistort(frame, self.camera.camera_matrix, self.camera.dist_coefficients)
            faces = self.gaze_estimator.detect_faces(undistorted)

            if len(faces) == 0:
                continue

            screen.fill((255, 255, 255))
            pygame.draw.circle(screen, (0, 255, 0), (0, 0), 10)
            pygame.draw.circle(screen, (0, 255, 0), (1792 // 2, 1120 // 2), 10)
            pygame.draw.circle(screen, (0, 255, 0), (0, 1120), 10)
            pygame.draw.circle(screen, (0, 255, 0), (1792, 0), 10)
            pygame.draw.circle(screen, (0, 255, 0), (1792, 1120), 10)

            self.gaze_estimator.estimate_gaze(undistorted, faces[0])
            print("faces[0] center: ", faces[0].center)
            print("gaze vecotr: ", faces[0].gaze_vector)
            print("distnce from camera: ", faces[0].get_distance(self.camera))
            # faces[0].center[2] = faces[0].get_distance(self.camera)[1]
            plane_norm = np.array([0, 0, 1])

            d = np.dot(plane_norm, faces[0].gaze_vector)

            estimated_point = None

            if d != 0:
                p_point = np.array([0, 0, 0])
                t = np.dot(p_point - faces[0].leye.center * 1000, plane_norm) / d
                gaze_point = faces[0].leye.center * 1000 + t * faces[0].gaze_vector
                print(f"Gaze point: {gaze_point}")
                # Check if gaze_point is within the screen boundaries
                estimated_point = get_point_on_screen((340, 220), (1792, 1120), gaze_point)
                print(f"Screen coordinates: {estimated_point}")
                pygame.draw.circle(screen, (255, 0, 0), estimated_point, 10)
            else:
                print("[ CALIBRATION ] Gaze vector is parallel to the screen")

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit()
                        return
                    elif event.key == pygame.K_SPACE:
                        if estimated_point is None:
                            print("[ CALIBRATION ] No gaze point estimated")
                            continue
                        calib_data = dict()

                        calib_data["estimated"] = dict()
                        calib_data["estimated"]["x"] = int(estimated_point[0])
                        calib_data["estimated"]["y"] = int(estimated_point[1])
                        calib_data["actual"] = dict()
                        calib_data["actual"]["x"] = 1792 // 2
                        calib_data["actual"]["y"] = 1120 // 2
                        results[calib_count] = calib_data
                        calib_count += 1
                        pygame.draw.circle(screen, (0, 255, 0), estimated_point, 10)
                    elif event.key == pygame.K_s:
                        with open(f"data/calib/{name}.yaml", "w") as f:
                            yaml.dump(results, f)
                        print("[ CALIBRATION ] Calibration data saved")
                        return

            pygame.display.update()
            pygame.time.delay(10)

    def get_calib_data(self, name):
        with open(f"data/calib/{name}.yaml", "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        avg_x = 0
        avg_y = 0
        for key, value in data.items():
            avg_x += value["estimated"]["x"]
            avg_y += value["estimated"]["y"]
        avg_x /= len(data)
        avg_y /= len(data)

        return (avg_x, avg_y), data



if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--mode", type=int, default=0,
                            help="0 - calibrate, 1 - mirror, 2 - tracker")
    argparse.add_argument("--user_calib", type=str)

    args = argparse.parse_args()
    mode = args.mode
    demo = Demo(mode, args.user_calib)
    print("[ Starting demo ]")
    demo.run()
    print("[ Demo stopped ]")
