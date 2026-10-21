"""Person detection over a whole video, with per-detection torso colour features.

Decoding is the expensive part (and the only pass that needs pixels so far), so the
colour features are computed in the same pass. Output is cached as an .npz per
(video, detector config).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from handball.identity.team_features import HsvTorsoFeatureExtractor

DET_DTYPE = np.dtype([("frame", "i4"), ("x1", "f4"), ("y1", "f4"), ("x2", "f4"), ("y2", "f4"), ("conf", "f4")])


def iter_frames(video: Path, batch: int):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    frame_id, buf = 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        buf.append((frame_id, frame))
        if len(buf) == batch:
            yield buf
            buf = []
    if buf:
        yield buf
    cap.release()


class UltralyticsPersonDetector:
    """COCO person class from an Ultralytics checkpoint, batched."""

    def __init__(self, weights: str, imgsz: int = 1280, conf: float = 0.1, half: bool = True, batch: int = 16):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.imgsz, self.conf, self.half, self.batch = imgsz, conf, half, batch

    def __call__(self, frames_bgr: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray | None]]:
        """Per frame: (boxes (n, 5) xyxy+conf, COCO keypoints (n, 17, 3) x/y/conf or None for box models)."""
        results = self.model.predict(
            frames_bgr, imgsz=self.imgsz, conf=self.conf, classes=[0], half=self.half, verbose=False
        )
        out = []
        for r in results:
            boxes = np.c_[r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()]
            kps = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None
            out.append((boxes, kps))
        return out


def detect_video(video: Path, detector, features: HsvTorsoFeatureExtractor, max_frames: int | None = None):
    """Returns (detections structured array, colour features (N, F), keypoints (N, 17, 3) or None)."""
    rows, feats, kps_all = [], [], []
    for batch in iter_frames(video, detector.batch):
        if max_frames is not None and batch[0][0] > max_frames:
            break
        outputs = detector([f for _, f in batch])
        for (frame_id, frame), (boxes, kps) in zip(batch, outputs):
            for k, (x1, y1, x2, y2, c) in enumerate(boxes):
                rows.append((frame_id, x1, y1, x2, y2, c))
                feats.append(features.extract(frame, np.array([x1, y1, x2, y2])))
                if kps is not None:
                    kps_all.append(kps[k])
    dets = np.array(rows, dtype=DET_DTYPE)
    keypoints = np.asarray(kps_all, dtype=np.float32) if kps_all else None
    return dets, np.asarray(feats, dtype=np.float32).reshape(len(dets), features.feature_size), keypoints


def save_detections(path: Path, dets: np.ndarray, feats: np.ndarray, keypoints: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = {"keypoints": keypoints} if keypoints is not None else {}
    np.savez_compressed(path, dets=dets, feats=feats, **extra)


def load_detections(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    data = np.load(path)
    return data["dets"], data["feats"], data["keypoints"] if "keypoints" in data.files else None


def attach_keypoints(dets: np.ndarray, pose_dets: np.ndarray, pose_kps: np.ndarray, min_iou: float = 0.5) -> np.ndarray:
    """(N, 17, 3) keypoints for `dets` from another detector's pose output, matched by
    best IoU per frame; zero confidence where nothing matches."""
    out = np.zeros((len(dets), 17, 3), np.float32)
    order_a, order_b = np.argsort(dets["frame"], kind="stable"), np.argsort(pose_dets["frame"], kind="stable")
    frames = np.unique(dets["frame"])
    sa = np.searchsorted(dets["frame"][order_a], frames)
    ea = np.searchsorted(dets["frame"][order_a], frames, side="right")
    sb = np.searchsorted(pose_dets["frame"][order_b], frames)
    eb = np.searchsorted(pose_dets["frame"][order_b], frames, side="right")
    box = lambda d: np.c_[d["x1"], d["y1"], d["x2"], d["y2"]]  # noqa: E731
    for f_i in range(len(frames)):
        ia, ib = order_a[sa[f_i] : ea[f_i]], order_b[sb[f_i] : eb[f_i]]
        if not len(ib):
            continue
        a, b = box(dets[ia]), box(pose_dets[ib])
        x1 = np.maximum(a[:, None, 0], b[None, :, 0])
        y1 = np.maximum(a[:, None, 1], b[None, :, 1])
        x2 = np.minimum(a[:, None, 2], b[None, :, 2])
        y2 = np.minimum(a[:, None, 3], b[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
        iou = inter / (area(a)[:, None] + area(b)[None] - inter)
        j = iou.argmax(1)
        ok = iou[np.arange(len(ia)), j] >= min_iou
        out[ia[ok]] = pose_kps[ib[j[ok]]]
    return out
