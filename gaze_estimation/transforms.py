from typing import Any

import cv2
import numpy as np
import torch
import torchvision
import yacs.config


def create_transform(isFace) -> Any:
    if not isFace:
        return _create_mpiigaze_transform()
    elif isFace:
        return _create_mpiifacegaze_transform()
    else:
        raise ValueError

def convert_to_float(x):
    return x.astype(np.float32) / 255

def add_dim(x):
    return x[None, :, :]
def _create_mpiigaze_transform() -> Any:
    scale = torchvision.transforms.Lambda(convert_to_float)
    transform = torchvision.transforms.Compose([
        scale,
        torch.from_numpy,
        torchvision.transforms.Lambda(add_dim),
    ])
    return transform

def convert_color(x):
    return cv2.cvtColor(
        cv2.equalizeHist(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)), cv2.
        COLOR_GRAY2BGR)

def trans(x):
    return x.transpose(2, 0, 1)

def myresize(x, size = 224):
    return cv2.resize(x, (size, size))

def _identity(x):
    return x
def _create_mpiifacegaze_transform() -> Any:
    scale = convert_to_float
    identity = _identity
    size = 224
    if size != 224:
        resize = myresize
    else:
        resize = identity
    # if config.transform.mpiifacegaze_gray:
    #     to_gray = convert_color
    # else:
    #     to_gray = identity

    transform = torchvision.transforms.Compose([
        resize,
        # to_gray,
        trans,
        scale,
        torch.from_numpy,
        torchvision.transforms.Normalize(mean=[0.406, 0.456, 0.485],
                                         std=[0.225, 0.224, 0.229]),
    ])
    return transform
