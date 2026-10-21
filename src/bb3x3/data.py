"""Loaders for the TrackID3x3 challenge data and the submission writer.

Layout (unpacked from the official Google Drive zips)::

    <root>/training/videos/IMG_01xx.MOV
    <root>/training/court_registration.csv
    <root>/training/games_roster.csv
    <root>/training/full player annotations/court position/IMG_01xx.txt
    <root>/training/full player annotations/Image-space bounding boxes/IMG_01xx.txt
    <root>/challenge/IMG_011x.MP4
    <root>/challenge/court_registration.csv
    <root>/challenge/games_roster.csv

Court coordinates are centimetres on a 1505 x 1105 cm court. Positions are rows of
``frame_id,court_x,court_y,team_id,jersey_number`` (frames start at 1), which is
both the training ground-truth format and the challenge submission format.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DATA_ROOT = Path("/data/trackid3x3")
COURT_SIZE_CM = (1505.0, 1105.0)

Identity = tuple[str, str]  # (team_id, jersey_number), e.g. ("A", "10")


@dataclass(frozen=True)
class Game:
    video: str
    roster: dict[str, tuple[str, ...]]  # team_id -> jersey numbers
    game_start_frame: int | None = None  # training only
    offense_team: str | None = None  # training only: team with the ball at game start

    @property
    def identities(self) -> list[Identity]:
        return [(team, number) for team, numbers in sorted(self.roster.items()) for number in numbers]


def _parse_team_jersey(field: str) -> dict[str, tuple[str, ...]]:
    roster = {}
    for part in field.split(";"):
        team, numbers = part.split(":")
        roster[team.strip()] = tuple(n.strip() for n in numbers.split(","))
    return roster


def load_games(split_dir: Path) -> dict[str, Game]:
    """Read ``games_roster.csv``; training rows also carry the start frame and offense team."""
    games = {}
    with open(split_dir / "games_roster.csv", newline="") as f:
        for row in csv.DictReader(f):
            roster = _parse_team_jersey(row["team: jersey"])
            start = int(row["game_start_frame"]) if row.get("game_start_frame") else None
            offense = None
            if row.get("offense_track_ids"):
                track_jersey = dict(
                    (k.strip(), v.strip()) for k, v in (p.split(":") for p in row["track_id: jersey"].split(","))
                )
                offense_numbers = {track_jersey[t.strip()] for t in row["offense_track_ids"].split(",")}
                offense = next(t for t, nums in roster.items() if set(nums) == offense_numbers)
            games[row["video"]] = Game(row["video"], roster, start, offense)
    return games


def load_registration(split_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (image_px, court_cm) correspondences from ``court_registration.csv``, each (N, 2)."""
    with open(split_dir / "court_registration.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    image = np.array([[float(r["image_x"]), float(r["image_y"])] for r in rows])
    court = np.array([[float(r["court_x"]), float(r["court_y"])] for r in rows])
    return image, court


def image_to_court_homography(split_dir: Path) -> np.ndarray:
    """Homography mapping image pixels to court centimetres from the official registration."""
    image, court = load_registration(split_dir)
    H, _ = cv2.findHomography(image, court)
    return H


def project(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, H).reshape(-1, 2)


def load_positions(path: Path) -> np.ndarray:
    """Read a positions file (ground truth or submission) into a structured array."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            rows.append((int(parts[0]), float(parts[1]), float(parts[2]), parts[3].strip(), parts[4].strip()))
    return np.array(rows, dtype=POSITION_DTYPE)


POSITION_DTYPE = np.dtype([("frame", "i4"), ("x", "f8"), ("y", "f8"), ("team", "U4"), ("jersey", "U4")])


def write_positions(path: Path, positions: np.ndarray) -> None:
    """Write the headerless submission file, sorted by frame then identity."""
    order = np.lexsort((positions["jersey"], positions["team"], positions["frame"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in positions[order]:
            f.write(f"{int(r['frame'])},{r['x']:.3f},{r['y']:.3f},{r['team']},{r['jersey']}\n")


def load_boxes(path: Path) -> np.ndarray:
    """Read a MOT-style box file: columns frame, id, x, y, w, h, conf, -1, -1."""
    return np.loadtxt(path, delimiter=",", ndmin=2)


def training_paths(video: str, root: Path = DATA_ROOT) -> dict[str, Path]:
    ann = root / "training" / "full player annotations"
    return {
        "video": root / "training" / "videos" / f"{video}.MOV",
        "positions": ann / "court position" / f"{video}.txt",
        "boxes": ann / "Image-space bounding boxes" / f"{video}.txt",
    }


def challenge_video_path(video: str, root: Path = DATA_ROOT) -> Path:
    return root / "challenge" / f"{video}.MP4"


def match_boxes_to_identities(boxes: np.ndarray, positions: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Row index into `positions` for each training GT box (-1 if none): GT court
    position is exactly H(bbox bottom-centre), so the match is unambiguous."""
    out = np.full(len(boxes), -1)
    feet = project(H, np.c_[boxes[:, 2] + boxes[:, 4] / 2, boxes[:, 3] + boxes[:, 5]])
    for frame in np.unique(boxes[:, 0]).astype(int):
        bi = np.flatnonzero(boxes[:, 0] == frame)
        pi = np.flatnonzero(positions["frame"] == frame)
        if not len(pi):
            continue
        d = np.linalg.norm(feet[bi][:, None] - np.c_[positions["x"][pi], positions["y"][pi]][None], axis=2)
        j = d.argmin(axis=1)
        ok = d[np.arange(len(bi)), j] < 1.0
        out[bi[ok]] = pi[j[ok]]
    return out
