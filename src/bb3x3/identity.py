"""Identity: squad from bib colour, then jersey number within the squad.

Four squads play in every TrackID3x3 outdoor game, each in its own bib colour (seen
in all 8 training games); a game's roster names two of them as teams A/B. Squad
colour prototypes come from training ground-truth boxes, so the challenge videos
are only ever *scored* against them (no fitting on challenge data).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from handball.identity.team_features import histogram_distance

from bb3x3.data import Game

SQUADS: dict[str, tuple[str, ...]] = {
    "purple": ("4", "10", "12"),
    "yellow": ("6", "9", "10"),
    "green": ("6", "9", "11"),
    "pink": ("1", "3", "5"),
}


def squad_of_numbers(numbers: tuple[str, ...]) -> str:
    return next(s for s, nums in SQUADS.items() if set(nums) == set(numbers))


def game_squads(game: Game) -> dict[str, str]:
    """squad -> team id (A/B) for the two squads of this game."""
    return {squad_of_numbers(nums): team for team, nums in game.roster.items()}


@dataclass
class SquadPrototypes:
    centroids: dict[str, np.ndarray]  # squad -> mean torso histogram
    max_distance: float  # beyond this from every centroid: not a bibbed player

    def distances(self, feats: np.ndarray) -> dict[str, np.ndarray]:
        root = np.sqrt(np.clip(feats, 0, None))
        return {
            s: np.linalg.norm(root - np.sqrt(c)[None], axis=1) / np.sqrt(2.0) for s, c in self.centroids.items()
        }


def fit_squad_prototypes(feats_by_squad: dict[str, np.ndarray], percentile: float = 95.0, slack: float = 1.25):
    centroids = {}
    own = []
    for squad, feats in feats_by_squad.items():
        mean = feats.mean(axis=0)
        centroids[squad] = mean / mean.sum()
        own += [histogram_distance(f, centroids[squad]) for f in feats]
    return SquadPrototypes(centroids, slack * float(np.percentile(own, percentile)))


def assign_track_squads(
    track_ids: np.ndarray, feats: np.ndarray, protos: SquadPrototypes, min_share: float = 0.5
) -> dict[int, str | None]:
    """Each track takes the squad most of its detections are nearest to; None when
    most detections are far from every squad (spectators, referees, staff)."""
    dists = protos.distances(feats)
    names = list(dists)
    stacked = np.stack([dists[s] for s in names], axis=1)
    nearest = stacked.argmin(axis=1)
    near_enough = stacked.min(axis=1) <= protos.max_distance
    out = {}
    for tid in np.unique(track_ids[track_ids >= 0]):
        m = track_ids == tid
        votes = np.bincount(nearest[m & near_enough], minlength=len(names))
        out[int(tid)] = names[int(votes.argmax())] if votes.max() >= min_share * m.sum() else None
    return out


def placeholder_numbers(
    track_ids: np.ndarray, frames: np.ndarray, track_squad: dict[int, str | None], numbers_of: dict[str, tuple]
) -> dict[int, str]:
    """v0 jersey: no number recognition yet. Tracks of a squad get its numbers so
    that tracks overlapping in time never share one (greedy interval colouring,
    longest track first) — the labels themselves are arbitrary."""
    spans = {}
    for tid in np.unique(track_ids[track_ids >= 0]):
        f = frames[track_ids == tid]
        spans[int(tid)] = (int(f.min()), int(f.max()), len(f))
    out: dict[int, str] = {}
    taken: dict[str, list[tuple[int, int, str]]] = {}
    for tid in sorted(spans, key=lambda t: -spans[t][2]):
        squad = track_squad.get(tid)
        if squad is None or squad not in numbers_of:
            continue
        lo, hi, _ = spans[tid]
        busy = {n for a, b, n in taken.get(squad, []) if a <= hi and lo <= b}
        free = [n for n in numbers_of[squad] if n not in busy]
        number = free[0] if free else numbers_of[squad][0]
        out[tid] = number
        taken.setdefault(squad, []).append((lo, hi, number))
    return out
