"""Labelled player crops from the 720p training transcodes, for the player classifier.

    python scripts/data_prep/build_player_crops.py [--every 3] [--out /data/trackid3x3/crops]

One JPEG per ground-truth box (box padded to 1:2 and resized to 96x192), labelled
with the player = squad + jersey number (12 classes over the four squads), plus
<out>/index.csv: path,video,frame,player,squad,team,jersey,box_h.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from bb3x3.data import DATA_ROOT, image_to_court_homography, load_boxes, load_games, load_positions, match_boxes_to_identities, training_paths
from bb3x3.crops import padded_crop
from bb3x3.identity import squad_of_numbers

SCALE_4K_TO_720 = 1280.0 / 3840.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--every", type=int, default=3)
    parser.add_argument("--out", type=Path, default=DATA_ROOT / "crops")
    args = parser.parse_args()

    games = load_games(DATA_ROOT / "training")
    H_4k = image_to_court_homography(DATA_ROOT / "training")
    rows = []
    for video, game in sorted(games.items()):
        paths = training_paths(video)
        boxes = load_boxes(paths["boxes"])
        boxes = boxes[(boxes[:, 0] % args.every) == 0]
        positions = load_positions(paths["positions"])
        ident = match_boxes_to_identities(boxes, positions, H_4k)
        by_frame: dict[int, list[int]] = {}
        for i in np.flatnonzero(ident >= 0):
            by_frame.setdefault(int(boxes[i, 0]), []).append(i)
        (args.out / video).mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(DATA_ROOT / "training" / "videos_720p" / f"{video}.mp4"))
        frame_id = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            for i in by_frame.get(frame_id, []):
                x, y, w, h = boxes[i, 2:6] * SCALE_4K_TO_720
                team, jersey = positions["team"][ident[i]], positions["jersey"][ident[i]]
                squad = squad_of_numbers(game.roster[team])
                rel = f"{video}/{frame_id:06d}_{squad}_{jersey}.jpg"
                cv2.imwrite(str(args.out / rel), padded_crop(frame, np.array([x, y, x + w, y + h])), [cv2.IMWRITE_JPEG_QUALITY, 92])
                rows.append((rel, video, frame_id, f"{squad}_{jersey}", squad, team, jersey, round(float(h), 1)))
        print(video, sum(r[1] == video for r in rows), "crops", flush=True)
    with open(args.out / "index.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "video", "frame", "player", "squad", "team", "jersey", "box_h"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
