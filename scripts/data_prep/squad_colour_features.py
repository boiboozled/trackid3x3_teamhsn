"""Torso colour features of ground-truth players in the 720p training videos, per squad.

    python scripts/data_prep/squad_colour_features.py [--every 10]

Writes /data/trackid3x3/features/squad_colour.npz with, per training video, the
features and squad of every sampled GT box (used for leave-one-video-out squad
prototypes). GT boxes are in 4K pixels and are scaled to 720p.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from handball.identity.team_features import HsvTorsoFeatureExtractor

from bb3x3.data import DATA_ROOT, image_to_court_homography, load_boxes, load_games, load_positions, match_boxes_to_identities, training_paths
from bb3x3.identity import squad_of_numbers

SCALE_4K_TO_720 = 1280.0 / 3840.0




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--out", type=Path, default=DATA_ROOT / "features" / "squad_colour.npz")
    args = parser.parse_args()

    games = load_games(DATA_ROOT / "training")
    H_4k = image_to_court_homography(DATA_ROOT / "training")
    extractor = HsvTorsoFeatureExtractor()
    result = {}
    for video, game in sorted(games.items()):
        paths = training_paths(video)
        boxes = load_boxes(paths["boxes"])
        positions = load_positions(paths["positions"])
        boxes = boxes[(boxes[:, 0] % args.every) == 0]
        ident = match_boxes_to_identities(boxes, positions, H_4k)
        video_720p = DATA_ROOT / "training" / "videos_720p" / f"{video}.mp4"
        if not video_720p.exists():
            print(video, "skipped: no 720p transcode yet")
            continue
        cap = cv2.VideoCapture(str(video_720p))
        feats, squads, frame_ids = [], [], []
        frame_id = 0
        wanted = set(boxes[:, 0].astype(int))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            if frame_id not in wanted:
                continue
            for i in np.flatnonzero(boxes[:, 0] == frame_id):
                if ident[i] < 0:
                    continue
                x, y, w, h = boxes[i, 2:6] * SCALE_4K_TO_720
                feats.append(extractor.extract(frame, np.array([x, y, x + w, y + h])))
                team = positions["team"][ident[i]]
                squads.append(squad_of_numbers(game.roster[team]))
                frame_ids.append(frame_id)
        result[f"{video}/feats"] = np.asarray(feats, np.float32)
        result[f"{video}/squads"] = np.asarray(squads)
        result[f"{video}/frames"] = np.asarray(frame_ids)
        print(video, len(feats), "features", dict(zip(*np.unique(squads, return_counts=True))))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **result)


if __name__ == "__main__":
    main()
