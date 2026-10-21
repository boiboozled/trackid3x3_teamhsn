"""Median background plate per video (players removed) and a camera-stability check.

    python scripts/data_prep/background_plates.py --videos /data/trackid3x3/challenge/IMG_0114.MP4 ... \
        --out-dir /data/trackid3x3/plates

Writes <out-dir>/<video>.png. Stability: every sampled frame is ECC-aligned
(translation) to the plate; the max shift is printed, and a drawing of the provided
court registration points on each plate goes to <out-dir>/<video>_registration.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from bb3x3.data import load_registration


def sample_frames(path: Path, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    return frames


def ecc_shift(reference: np.ndarray, frame: np.ndarray) -> np.ndarray:
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    cur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    _, warp = cv2.findTransformECC(ref, cur, warp, cv2.MOTION_TRANSLATION, criteria, None, 5)
    return warp[:, 2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-frames", type=int, default=150)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for video in args.videos:
        frames = sample_frames(video, args.n_frames)
        plate = np.median(np.stack(frames), axis=0).astype(np.uint8)
        cv2.imwrite(str(args.out_dir / f"{video.stem}.png"), plate)
        shifts = np.array([ecc_shift(plate, f) for f in frames])
        mag = np.linalg.norm(shifts, axis=1)
        print(f"{video.stem}: {plate.shape[1]}x{plate.shape[0]}, ECC shift vs plate: median {np.median(mag):.2f}px, "
              f"max {mag.max():.2f}px (frame sample {int(mag.argmax())})")

        image_pts, court_pts = load_registration(video.parent)
        scale = plate.shape[1] / 1280.0 if video.suffix.lower() == ".mp4" else 1.0
        vis = plate.copy()
        for (x, y), (cx, cy) in zip(image_pts * scale, court_pts):
            cv2.circle(vis, (int(round(x)), int(round(y))), 6, (0, 0, 255), 2)
            cv2.putText(vis, f"({cx:.0f},{cy:.0f})", (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(str(args.out_dir / f"{video.stem}_registration.png"), vis)


if __name__ == "__main__":
    main()
