"""Fixed-cardinality identity assignment: every squad has exactly three players.

Per frame and squad, detections are matched to the squad's three identity slots
(one per jersey number) with the Hungarian algorithm. The cost combines court-space
motion (distance to the slot's constant-velocity prediction) and, when a player
classifier is available, the detection's evidence for the slot's jersey number.
A slot unseen for `lost_after` frames is re-acquired anywhere on court, by identity
evidence only. Short gaps per identity are then filled by linear interpolation.

Without a classifier the slot labels are arbitrary but never collide, which is what
the identity-free and team-only scores measure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from bb3x3.data import POSITION_DTYPE

INFEASIBLE = 1e6


@dataclass
class SlotConfig:
    dist_scale_cm: float = 50.0  # 50 cm of prediction error costs as much as 1 nat of identity evidence
    identity_weight: float = 1.0
    max_identity_nll: float = 3.0  # cap per-frame jersey evidence: one bad crop must not veto a motion match
    gate_base_cm: float = 150.0
    gate_per_frame_cm: float = 30.0  # ~9 m/s
    lost_after: int = 45  # frames without a match before a slot is re-acquired anywhere
    velocity_alpha: float = 0.3  # EMA on per-frame velocity
    max_unmatched_cost: float = 8.0  # do not force a match that costs more than this
    max_gap_fill: int = 30  # frames; linear interpolation between matches of one identity


class _Slot:
    def __init__(self, number: str):
        self.number = number
        self.pos: np.ndarray | None = None
        self.vel = np.zeros(2)
        self.last = -(10**9)

    def predict(self, frame: int) -> np.ndarray | None:
        if self.pos is None:
            return None
        return self.pos + self.vel * min(frame - self.last, 15)

    def update(self, frame: int, pos: np.ndarray, alpha: float) -> None:
        if self.pos is not None and frame - self.last <= 5:
            v = (pos - self.pos) / (frame - self.last)
            self.vel = (1 - alpha) * self.vel + alpha * v
        else:
            self.vel = np.zeros(2)
        self.pos, self.last = pos, frame


def assign_slots(
    frames: np.ndarray,
    court_xy: np.ndarray,
    det_squad: np.ndarray,
    number_nll: np.ndarray | None,
    squad_numbers: dict[str, tuple[str, ...]],
    cfg: SlotConfig,
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """Returns {(squad, number): (frames, court_xy)} of matched detections.

    `det_squad`: squad name per detection ("" = not a player of this game).
    `number_nll`: (N, 3) negative log-probabilities of the detection being each of
    its squad's numbers in `squad_numbers[squad]` order, or None.
    """
    out: dict[tuple[str, str], tuple[list, list]] = {}
    for squad, numbers in squad_numbers.items():
        slots = [_Slot(n) for n in numbers]
        idx_all = np.flatnonzero(det_squad == squad)
        order = idx_all[np.argsort(frames[idx_all], kind="stable")]
        bounds = np.flatnonzero(np.diff(frames[order])) + 1
        for group in np.split(order, bounds):
            if not len(group):
                continue
            frame = int(frames[group[0]])
            cost = np.full((len(slots), len(group)), INFEASIBLE)
            for k, slot in enumerate(slots):
                pred = slot.predict(frame)
                gap = frame - slot.last
                ident = (
                    cfg.identity_weight * np.minimum(number_nll[group, k], cfg.max_identity_nll)
                    if number_nll is not None
                    else np.zeros(len(group))
                )
                if pred is None or gap > cfg.lost_after:
                    cost[k] = 2.0 + ident  # re-acquire anywhere; flat motion cost
                    continue
                d = np.linalg.norm(court_xy[group] - pred, axis=1)
                ok = d <= cfg.gate_base_cm + cfg.gate_per_frame_cm * gap
                cost[k, ok] = d[ok] / cfg.dist_scale_cm + ident[ok]
            rows, cols = linear_sum_assignment(cost)
            for k, j in zip(rows, cols):
                if cost[k, j] > cfg.max_unmatched_cost:
                    continue
                i = group[j]
                slots[k].update(frame, court_xy[i], cfg.velocity_alpha)
                f_list, xy_list = out.setdefault((squad, slots[k].number), ([], []))
                f_list.append(frame)
                xy_list.append(court_xy[i])
    return {key: (np.asarray(f), np.asarray(xy)) for key, (f, xy) in out.items()}


def fill_gaps(frames: np.ndarray, xy: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation over gaps of at most `max_gap` missing frames."""
    if len(frames) < 2 or max_gap <= 0:
        return frames, xy
    new_f, new_xy = [frames], [xy]
    for a in np.flatnonzero((np.diff(frames) > 1) & (np.diff(frames) <= max_gap + 1)):
        f0, f1 = frames[a], frames[a + 1]
        fill = np.arange(f0 + 1, f1)
        t = ((fill - f0) / (f1 - f0))[:, None]
        new_f.append(fill)
        new_xy.append((1 - t) * xy[a] + t * xy[a + 1])
    f = np.concatenate(new_f)
    order = np.argsort(f, kind="stable")
    return f[order], np.concatenate(new_xy)[order]


def slots_to_positions(
    tracks: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]], squad_team: dict[str, str], max_gap: int
) -> np.ndarray:
    rows = []
    for (squad, number), (f, xy) in tracks.items():
        f, xy = fill_gaps(f, xy, max_gap)
        rows += [(int(fr), float(p[0]), float(p[1]), squad_team[squad], number) for fr, p in zip(f, xy)]
    return np.array(rows, dtype=POSITION_DTYPE)
