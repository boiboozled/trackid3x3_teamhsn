"""TI-HOTA, matching the official implementation (open-starlab/TrackID3x3,
``script/compute_TI-HOTA.ipynb``, Apache-2.0) but grouped per (frame, identity).

IdSim is 1 only for identical (team, jersey), so a frame's assignment problem splits
into independent blocks, one per identity; the common 1 GT x 1 prediction block is
just a threshold test and is vectorised. ``tests/test_ti_hota.py`` checks equality with
the official per-frame Hungarian version.

Units follow the files: positions in cm, so the official ``tau = 100`` is 1 m.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

ALPHAS = np.arange(0.05, 1.0, 0.05)
_LN_005 = np.log(0.05)


def loc_sim(sq_dist: np.ndarray, tau: float) -> np.ndarray:
    return np.exp(_LN_005 * sq_dist / tau**2)


def collapse_identity(positions: np.ndarray, mode: str) -> np.ndarray:
    """Diagnostic variants: "full" scores as-is, "team" ignores jersey numbers,
    "none" ignores identity entirely (pure detection + position quality)."""
    if mode == "full":
        return positions
    out = positions.copy()
    out["jersey"] = "-"
    if mode == "none":
        out["team"] = "-"
    elif mode != "team":
        raise ValueError(mode)
    return out


def _identity_keys(positions: np.ndarray) -> np.ndarray:
    return np.char.add(np.char.add(positions["team"], ":"), positions["jersey"])


def ti_hota(gt: np.ndarray, pred: np.ndarray, tau: float = 100.0, alphas: np.ndarray = ALPHAS) -> dict:
    """Score predictions against ground truth; both are ``data.POSITION_DTYPE`` arrays."""
    gt_ids, pred_ids = _identity_keys(gt), _identity_keys(pred)
    identities = sorted(set(gt_ids))

    groups: dict[tuple[int, str], list[list[int]]] = defaultdict(lambda: [[], []])
    for i, (f, k) in enumerate(zip(gt["frame"], gt_ids)):
        groups[(int(f), k)][0].append(i)
    for j, (f, k) in enumerate(zip(pred["frame"], pred_ids)):
        groups[(int(f), k)][1].append(j)

    gt_xy = np.c_[gt["x"], gt["y"]]
    pred_xy = np.c_[pred["x"], pred["y"]]

    # 1x1 blocks: a single similarity per block.
    single_sim, single_id = [], []
    # Everything else: a small similarity matrix per block.
    multi = []
    for (_, key), (gi, pj) in groups.items():
        if not gi or not pj:
            continue
        sims = loc_sim(((gt_xy[gi][:, None] - pred_xy[pj][None]) ** 2).sum(-1), tau)
        if sims.shape == (1, 1):
            single_sim.append(sims[0, 0])
            single_id.append(key)
        else:
            multi.append((key, sims))
    single_sim = np.asarray(single_sim)
    single_id = np.asarray(single_id)

    n_gt = {k: int((gt_ids == k).sum()) for k in identities}
    n_pred = {k: int((pred_ids == k).sum()) for k in identities}

    per_alpha, per_identity_alpha = [], []
    for alpha in alphas:
        tpa = {k: 0 for k in identities}
        hits = single_sim >= alpha
        for k, c in zip(*np.unique(single_id[hits], return_counts=True)):
            tpa[str(k)] += int(c)
        for key, sims in multi:
            cost = 1.0 - sims
            cost[sims < alpha] = 1e6
            rows, cols = linear_sum_assignment(cost)
            tpa[key] += int((sims[rows, cols] >= alpha).sum())
        tp = sum(tpa.values())
        fp = len(pred) - tp
        fn = len(gt) - tp
        det_a = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        ass = [tpa[k] / (n_gt[k] + n_pred[k] - tpa[k]) for k in identities]
        ass_a = float(np.mean(ass)) if ass else 0.0
        per_alpha.append((np.sqrt(det_a * ass_a), det_a, ass_a, tp, fp, fn))
        per_identity_alpha.append(ass)

    arr = np.array(per_alpha)
    # Identity-level AssA averaged over alphas, for diagnosing which player costs points.
    per_identity = dict(zip(identities, np.mean(per_identity_alpha, axis=0).tolist()))
    return {
        "TI-HOTA": float(arr[:, 0].mean()),
        "TI-DetA": float(arr[:, 1].mean()),
        "TI-AssA": float(arr[:, 2].mean()),
        "TI-TP": float(arr[:, 3].mean()),
        "TI-FP": float(arr[:, 4].mean()),
        "TI-FN": float(arr[:, 5].mean()),
        "TI-HOTA_alphas": arr[:, 0].tolist(),
        "TI-AssA_per_identity": per_identity,
    }
