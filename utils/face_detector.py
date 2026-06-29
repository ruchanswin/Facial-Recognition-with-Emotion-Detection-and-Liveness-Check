import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from torchvision import transforms

_mtcnn = None

def _get_mtcnn():
    global _mtcnn
    if _mtcnn is None:
        device = torch.device("cpu")  # MPS doesn't support adaptive_avg_pool2d with non-divisible sizes
        _mtcnn = MTCNN(
            image_size=112,
            margin=20,
            min_face_size=40,
            keep_all=False,   # return only the highest-confidence face
            device=device,
            post_process=False,  # return uint8 crop, we normalise ourselves
        )
    return _mtcnn


_normalise = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def detect_and_crop(image_bgr: np.ndarray, output_size: tuple = (112, 112)) -> torch.Tensor | None:
    """
    Detect the highest-confidence face in a BGR image and return a normalised tensor.

    Returns shape (3, H, W) float32 in range ~[-2.1, 2.6], or None if no face found.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    crop = _get_mtcnn()(rgb)   # returns (3, H, W) uint8 tensor or None
    if crop is None:
        return None
    # MTCNN returns float32 0-255 when post_process=False
    tensor = crop.float() / 255.0
    if output_size != (112, 112):
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0), size=output_size, mode="bilinear", align_corners=False
        ).squeeze(0)
    return _normalise(tensor)


def detect_and_crop_both(image_bgr: np.ndarray) -> tuple[torch.Tensor, np.ndarray] | tuple[None, None]:
    """
    Single MTCNN call returning both the normalised tensor (for models) and
    the uint8 RGB array (for display / liveness CNN).

    Returns (tensor, raw_arr) or (None, None) if no face found.
    tensor:   (3, 112, 112) float32 normalised
    raw_arr:  (112, 112, 3) uint8 RGB
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    crop = _get_mtcnn()(rgb)
    if crop is None:
        return None, None
    raw_arr = crop.permute(1, 2, 0).numpy().astype(np.uint8)
    tensor = _normalise(crop.float() / 255.0)
    return tensor, raw_arr


def detect_and_crop_raw(image_bgr: np.ndarray, output_size: tuple = (112, 112)) -> np.ndarray | None:
    """
    Same as detect_and_crop but returns a uint8 RGB numpy array.
    Useful for display or saving crops.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    crop = _get_mtcnn()(rgb)
    if crop is None:
        return None
    arr = crop.permute(1, 2, 0).numpy().astype(np.uint8)
    if arr.shape[:2] != output_size[::-1]:
        arr = cv2.resize(arr, output_size)
    return arr
