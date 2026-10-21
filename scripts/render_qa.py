"""Visual QA: draw predicted identities (court positions projected back into the image)
and a top-down court map on video frames.

    python scripts/render_qa.py --positions run/challenge/IMG_0114.txt --video /data/trackid3x3/challenge/IMG_0114.MP4 \
        --homography /data/trackid3x3/registration/IMG_0114.npy --out qa.jpg [--frames 500 3000 6000] [--mp4 qa.mp4 --start 3000 --seconds 20]

Used only to find bugs — never to edit predictions (the rules forbid manual identity correction).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from bb3x3.court import LINES, W, D
from bb3x3.data import load_positions, project
from bb3x3.registration import draw_template

TEAM_COLOURS = {"A": (0, 0, 255), "B": (255, 128, 0)}


def court_map(rows: np.ndarray, scale: float = 0.2) -> np.ndarray:
    m = 20
    img = np.full((int(D * scale) + 2 * m, int(W * scale) + 2 * m, 3), 40, np.uint8)
    for pl in LINES.values():
        cv2.polylines(img, [np.round(pl * scale + m).astype(np.int32)], False, (200, 200, 200), 1, cv2.LINE_AA)
    for r in rows:
        p = (int(r["x"] * scale + m), int(r["y"] * scale + m))
        cv2.circle(img, p, 5, TEAM_COLOURS.get(r["team"], (0, 255, 0)), -1)
        cv2.putText(img, r["jersey"], (p[0] + 6, p[1] + 4), 0, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw(frame: np.ndarray, rows: np.ndarray, H: np.ndarray, frame_id: int) -> np.ndarray:
    out = draw_template(frame, H, (0, 255, 255))
    if len(rows):
        img_pts = project(np.linalg.inv(H), np.c_[rows["x"], rows["y"]])
        for r, p in zip(rows, img_pts):
            c = TEAM_COLOURS.get(r["team"], (0, 255, 0))
            q = (int(p[0]), int(p[1]))
            cv2.circle(out, q, 4, c, -1)
            cv2.putText(out, f"{r['team']}{r['jersey']}", (q[0] - 12, q[1] + 16), 0, 0.5, c, 2, cv2.LINE_AA)
    cmap = court_map(rows)
    out[: cmap.shape[0], -cmap.shape[1] :] = cmap
    cv2.putText(out, f"frame {frame_id}  rows {len(rows)}", (10, 25), 0, 0.7, (255, 255, 255), 2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--homography", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[300, 2000, 4000, 6000, 8000, 10000])
    parser.add_argument("--mp4", type=Path, default=None)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=20)
    args = parser.parse_args()

    pos = load_positions(args.positions)
    H = np.load(args.homography) if args.homography.suffix == ".npy" else None
    cap = cv2.VideoCapture(str(args.video))
    tiles = []
    for f in args.frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
        ok, frame = cap.read()
        if ok:
            tiles.append(cv2.resize(draw(frame, pos[pos["frame"] == f], H, f), (960, 540)))
    if tiles:
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(args.out), np.vstack([np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]))
    if args.mp4:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start - 1)
        fps = cap.get(cv2.CAP_PROP_FPS)
        writer = None
        for f in range(args.start, args.start + int(args.seconds * fps)):
            ok, frame = cap.read()
            if not ok:
                break
            img = draw(frame, pos[pos["frame"] == f], H, f)
            if writer is None:
                writer = cv2.VideoWriter(str(args.mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, img.shape[1::-1])
            writer.write(img)
        if writer is not None:
            writer.release()


if __name__ == "__main__":
    main()
