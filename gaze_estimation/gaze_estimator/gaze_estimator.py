import logging
from typing import List

import numpy as np
import torch

from ..models.mpiifacegaze import resnet_simple
from ..transforms import create_transform, _create_mpiifacegaze_transform
from .common import MODEL3D, Camera, Face, FacePartsName
from .head_pose_estimation import HeadPoseNormalizer, LandmarkEstimator

logger = logging.getLogger(__name__)


class GazeEstimator:
    EYE_KEYS = [FacePartsName.REYE, FacePartsName.LEYE]

    def __init__(self, camera, face):

        self.isFace = face
        self.camera = camera
        self._normalized_camera = Camera(
            "data/camera/normalized_camera_params_face.yaml")

        self._landmark_estimator = LandmarkEstimator()
        self._head_pose_normalizer = HeadPoseNormalizer(
            self.camera, self._normalized_camera, 1)
        self._gaze_estimation_model = self._load_model()
        self._transform = _create_mpiifacegaze_transform()

    def _load_model(self) -> torch.nn.Module:
        model = resnet_simple.Model()
        checkpoint = torch.load("data/models/mpiifacegaze/mpiifacegaze_resnet_simple.pth",
                                map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.to(torch.device('cpu'))
        model.eval()
        return model

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        return self._landmark_estimator.detect_faces(image)

    def estimate_gaze(self, image: np.ndarray, face: Face) -> None:
        MODEL3D.estimate_head_pose(face, self.camera)
        MODEL3D.compute_3d_pose(face)
        MODEL3D.compute_face_eye_centers(face)

        if not self.isFace:
            for key in self.EYE_KEYS:
                eye = getattr(face, key.name.lower())
                self._head_pose_normalizer.normalize(image, eye)
            self._run_mpiigaze_model(face)
        elif self.isFace:
            self._head_pose_normalizer.normalize(image, face)
            self._run_mpiifacegaze_model(face)

    def _run_mpiigaze_model(self, face: Face) -> None:
        images = []
        head_poses = []
        for key in self.EYE_KEYS:
            eye = getattr(face, key.name.lower())
            image = eye.normalized_image
            normalized_head_pose = eye.normalized_head_rot2d
            if key == FacePartsName.REYE:
                image = image[:, ::-1]
                normalized_head_pose *= np.array([1, -1])
            image = self._transform(image)
            images.append(image)
            head_poses.append(normalized_head_pose)
        images = torch.stack(images)
        head_poses = np.array(head_poses).astype(np.float32)
        head_poses = torch.from_numpy(head_poses)

        device = torch.device(self._config.device)
        with torch.no_grad():
            images = images.to(device)
            head_poses = head_poses.to(device)
            predictions = self._gaze_estimation_model(images, head_poses)
            predictions = predictions.cpu().numpy()
            print(predictions)
            face.prediction = predictions[0]

        for i, key in enumerate(self.EYE_KEYS):
            eye = getattr(face, key.name.lower())
            eye.normalized_gaze_angles = predictions[i]
            if key == FacePartsName.REYE:
                eye.normalized_gaze_angles *= np.array([1, -1])
            eye.angle_to_vector()
            eye.denormalize_gaze_vector()


    def _run_mpiifacegaze_model(self, face: Face) -> None:
        image = self._transform(face.normalized_image).unsqueeze(0)

        device = torch.device('cpu')
        with torch.no_grad():
            image = image.to(device)
            prediction = self._gaze_estimation_model(image)
            prediction = prediction.cpu().numpy()

        face.prediction = prediction
        face.normalized_gaze_angles = prediction[0]
        face.angle_to_vector()
        face.denormalize_gaze_vector()
