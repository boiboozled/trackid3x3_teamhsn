"""Runs a handball `Tracker` (ByteTrack, BoT-SORT, ...) over cached detections.

Returns one track id per detection (-1 = not kept by the tracker). Image-dependent
trackers (BoT-SORT ReID/CMC) get frames via `frames_for`; the purely motion-based
ones run without decoding the video.
"""

from __future__ import annotations

import numpy as np
import supervision as sv


def run_tracker(tracker, dets: np.ndarray, n_frames: int, frame_source=None) -> np.ndarray:
    track_ids = np.full(len(dets), -1, dtype=np.int64)
    order = np.argsort(dets["frame"], kind="stable")
    starts = np.searchsorted(dets["frame"][order], np.arange(1, n_frames + 2))
    tracker.reset()
    frames = iter(frame_source) if frame_source is not None else None
    for frame_id in range(1, n_frames + 1):
        idx = order[starts[frame_id - 1] : starts[frame_id]]
        frame_bgr = next(frames) if frames is not None else None
        d = dets[idx]
        detections = sv.Detections(
            xyxy=np.c_[d["x1"], d["y1"], d["x2"], d["y2"]].astype(np.float32).reshape(-1, 4),
            confidence=d["conf"].astype(np.float32),
            class_id=np.zeros(len(d), dtype=int),
            data={"det_index": idx},
        )
        out = tracker.update(detections, frame_idx=frame_id - 1, frame_bgr=frame_bgr)
        if out.tracker_id is not None and len(out):
            track_ids[out.data["det_index"]] = out.tracker_id
    return track_ids
