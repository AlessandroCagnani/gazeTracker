from skimage.transform import resize

from gaze_estimation.gaze_estimator.common import Camera, visualizer
from gaze_estimation.gaze_estimator import GazeEstimator

from deepgaze.saliency_map import FasaSaliencyMapping

import cv2
import numpy as np
from typing import Tuple
import json
import torchvision
import torchvision.transforms as T
import torch
import torchvision.models as models
from PIL import Image



slc_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.Lambda(lambda x: x[None]),
])

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
        self.sift = self.get_sift()
        self.use_saliency_map = False

    def get_saliency(self):

        def preprocess(image, size=224):
            transform = T.Compose([
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.Lambda(lambda x: x[None]),
            ])
            return transform(image)

        def deprocess(image):
            transform = T.Compose([
                T.Lambda(lambda x: x[0]),
                T.Normalize(mean=[0, 0, 0], std=[4.3668, 4.4643, 4.4444]),
                T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
                T.ToPILImage(),
            ])
            return transform(image)

        model_ = 50
        if model_ == 18:
            model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights)
        elif model_ == 34:
            model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights)
        elif model_ == 50:
            model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights)
        elif model_ == 101:
            model = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights)
        elif model_ == 152:
            model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights)

        img = Image.open('data/backgrounds/bear-abandoned1.png').convert("RGB")
        X = preprocess(img)
        model.eval()
        X.requires_grad_()
        scores = model(X)
        score_max_index = scores.argmax()
        score_max = scores[0, score_max_index]
        score_max.backward()
        slc, _ = torch.max(X.grad.data.abs(), dim=1)
        slc = (slc - slc.min()) / (slc.max() - slc.min())

        # Resize the saliency map to the original image size
        slc_resized = resize(slc[0].numpy(), img.size[::-1], mode='reflect', anti_aliasing=True)
        return slc_resized

    def get_sift(self):
        img = cv2.imread('data/backgrounds/bear-abandoned1.png')
        # Converting image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Applying SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(gray, None)

        # insert in a np matrix
        kp, des = sift.compute(gray, kp)
        size_threshold = 10.0

        # Keep only the keypoints with size greater than the threshold
        kp = [keypoint for keypoint in kp if keypoint.size > size_threshold]
        return kp

    # this one use sift keypoints
    def adjust_gaze_point(self, gaze_point):
        if not self.sift:
            return gaze_point

        # Compute the distance and size from the gaze point to each keypoint
        distances_and_sizes = [(np.linalg.norm(np.array(gaze_point) - np.array(kp.pt)), kp.size) for kp in
                               self.sift]

        # Compute the weights (inverse of the distance squared, multiplied by size)
        weights = [(size / (dist ** 2)) if dist != 0 else 0 for dist, size in distances_and_sizes]

        # Compute the weighted average of the keypoint positions
        adjusted_gaze_point = np.average([kp.pt for kp in self.sift], weights=weights, axis=0)

        return adjusted_gaze_point

    def point_of_gaze(self, mode):

        ret, frame = self.camera.get_frame()
        if not ret:
            return None
        undistorted = cv2.undistort(frame, self.camera.camera_matrix, self.camera.dist_coefficients)
        faces = self.gaze_estimator.detect_faces(undistorted)

        # TODO: if no face detected, return None
        # TODO: return or series of points or return only from biggest bbox
        faces.sort(key=lambda x: (x.bbox[1][0] - x.bbox[0][0]) * (x.bbox[1][1] - x.bbox[0][1]), reverse=True)
        biggest_face = faces[0] if len(faces) > 0 else None

        if biggest_face:
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

            return (int(x), int(y))




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

    def init_saliency_map(self, img):
        self.saliency_map = self.get_saliency(img)

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

    def get_saliency_correction(self, point, radius, power=2):
        weights = []
        salient_points = []

        neighbours = self.saliency_map[point[1] - radius:point[1] + radius, point[0] - radius:point[0] + radius]

        for y in len(neighbours):
            for x in len(neighbours[0]):
                if neighbours[y][x] == 0:
                    continue

                salient_x = point[0] - radius + x
                salient_y = point[1] - radius + y

                salient_points.append((salient_x, salient_y))
                weight = 1 / (np.euclidean_distance(point, (salient_x, salient_y)) ** power) * neighbours[y][x]
                weights.append(weight)

        weights = np.array(weights) / np.sum(weights)
        saliency_correction = np.dot(weights, salient_points)

        return saliency_correction



    def idw_interpolation(self, estimated_point, power=2):
        errors = [self.get_correection_vector(data["point"], data["mean"]) for data in self.calib_data]

        weights = []
        for data in self.calib_data:
            distance = self.euclidean_distance(estimated_point, data["point"])
            if distance <= 40:  # TODO: try different values
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

