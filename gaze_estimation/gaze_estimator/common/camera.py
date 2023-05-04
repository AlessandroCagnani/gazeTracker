import numpy as np
import cv2
import yaml

class Camera:

    def __init__(self, camera_params_path):
        self.cap = cv2.VideoCapture(0)

        with open(camera_params_path) as f:
            data = yaml.safe_load(f)
        self.width = data['image_width']
        self.height = data['image_height']
        self.camera_matrix = np.array(data['camera_matrix']['data']).reshape(
            3, 3)
        self.dist_coefficients = np.array(
            data['distortion_coefficients']['data']).reshape(-1, 1)

    def get_frame(self):
        return self.cap.read()

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def project_points(self,
                       points3d,
                       rvec = None,
                       tvec = None) -> np.ndarray:
        assert points3d.shape[1] == 3
        if rvec is None:
            rvec = np.zeros(3, dtype=np.float)
        if tvec is None:
            tvec = np.zeros(3, dtype=np.float)
        points2d, _ = cv2.projectPoints(points3d, rvec, tvec,
                                        self.camera_matrix,
                                        self.dist_coefficients)
        return points2d.reshape(-1, 2)