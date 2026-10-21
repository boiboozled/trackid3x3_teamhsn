"""Player crops shared by the training-crop builder and pipeline inference."""

from __future__ import annotations

import cv2
import numpy as np

CROP_W, CROP_H = 96, 192


def padded_crop(frame: np.ndarray, xyxy: np.ndarray, pad: float = 0.1) -> np.ndarray:
    """Crop with `pad` context, widened/heightened to a 1:2 aspect, zero-padded at the borders."""
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    h = (y2 - y1) * (1 + pad)
    w = max((x2 - x1) * (1 + pad), h / 2)
    h = max(h, 2 * w)
    box = np.round([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]).astype(int)
    H, W = frame.shape[:2]
    out = np.zeros((box[3] - box[1], box[2] - box[0], 3), np.uint8)
    sx1, sy1, sx2, sy2 = max(box[0], 0), max(box[1], 0), min(box[2], W), min(box[3], H)
    if sx2 > sx1 and sy2 > sy1:
        out[sy1 - box[1] : sy2 - box[1], sx1 - box[0] : sx2 - box[0]] = frame[sy1:sy2, sx1:sx2]
    return cv2.resize(out, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)
