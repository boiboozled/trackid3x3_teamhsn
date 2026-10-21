"""YOLO detection dataset (class 0 = on-court player) from the 720p training transcodes.

    python scripts/data_prep/build_yolo_dataset.py [--every 10] [--val-videos IMG_0108 IMG_0109]

Only the six GT players are labelled; spectators, waiting squads and staff are
background, so a fine-tuned detector learns "player of this game", which is what the
pipeline wants. Writes <out>/{images,labels}/{train,val}/ and <out>/data.yaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from bb3x3.data import DATA_ROOT, load_boxes, load_games, training_paths

SCALE_4K_TO_720 = 1280.0 / 3840.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--val-videos", nargs="*", default=["IMG_0108", "IMG_0109"])
    parser.add_argument("--out", type=Path, default=DATA_ROOT / "yolo_players")
    args = parser.parse_args()

    for video in sorted(load_games(DATA_ROOT / "training")):
        split = "val" if video in args.val_videos else "train"
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        boxes = load_boxes(training_paths(video)["boxes"])
        cap = cv2.VideoCapture(str(DATA_ROOT / "training" / "videos_720p" / f"{video}.mp4"))
        frame_id, n = 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            if frame_id % args.every:
                continue
            b = boxes[boxes[:, 0] == frame_id][:, 2:6] * SCALE_4K_TO_720
            h, w = frame.shape[:2]
            x1, y1 = np.clip(b[:, 0], 0, w), np.clip(b[:, 1], 0, h)
            x2, y2 = np.clip(b[:, 0] + b[:, 2], 0, w), np.clip(b[:, 1] + b[:, 3], 0, h)
            keep = (x2 - x1 > 2) & (y2 - y1 > 2)
            lines = [
                f"0 {(a + c) / 2 / w:.6f} {(bb + d) / 2 / h:.6f} {(c - a) / w:.6f} {(d - bb) / h:.6f}"
                for a, bb, c, d in zip(x1[keep], y1[keep], x2[keep], y2[keep])
            ]
            stem = f"{video}_{frame_id:06d}"
            cv2.imwrite(str(args.out / "images" / split / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (args.out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            n += 1
        print(video, split, n, "frames", flush=True)
    (args.out / "data.yaml").write_text(
        f"path: {args.out}\ntrain: images/train\nval: images/val\nnames:\n  0: player\n"
    )


if __name__ == "__main__":
    main()
