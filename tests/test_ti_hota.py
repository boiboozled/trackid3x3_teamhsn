import numpy as np
import pytest

from bb3x3.data import POSITION_DTYPE
from bb3x3.ti_hota import ti_hota

from _official_ti_hota import compute_ti_hota

IDS = [("A", "4"), ("A", "10"), ("A", "12"), ("B", "1"), ("B", "3"), ("B", "5")]


def _official(gt, pred, tau):
    frames = []
    for f in sorted(set(gt["frame"]) | set(pred["frame"])):
        to_objs = lambda arr: [  # noqa: E731
            {"position": [r["x"], r["y"]], "attributes": f"{r['team']}:{r['jersey']}"} for r in arr[arr["frame"] == f]
        ]
        frames.append({"ground_truths": to_objs(gt), "predictions": to_objs(pred)})
    metrics, _ = compute_ti_hota(frames, tau)
    return metrics


def _random_tracks(rng, n_frames, ids):
    rows = [(f, *rng.uniform(0, 1100, 2), t, j) for f in range(1, n_frames + 1) for t, j in ids]
    return np.array(rows, dtype=POSITION_DTYPE)


def _perturb(rng, gt):
    pred = gt.copy()
    pred["x"] += rng.normal(0, 40, len(pred))
    pred["y"] += rng.normal(0, 40, len(pred))
    keep = rng.random(len(pred)) > 0.1  # missed detections
    swap = rng.random(len(pred)) < 0.05  # identity errors
    pred["jersey"][swap] = "9"
    dup = pred[rng.random(len(pred)) < 0.05].copy()  # duplicate identity in a frame
    dup["x"] += 30
    extra = gt[rng.random(len(gt)) < 0.03].copy()  # predictions in frames past the GT
    extra["frame"] += 1000
    return np.concatenate([pred[keep], dup, extra])


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("tau", [50.0, 100.0])
def test_matches_official(seed, tau):
    rng = np.random.default_rng(seed)
    gt = _random_tracks(rng, 40, IDS)
    gt = np.concatenate([gt, gt[rng.random(len(gt)) < 0.03]])  # GT with occasional duplicate identity rows
    pred = _perturb(rng, gt)
    ours, ref = ti_hota(gt, pred, tau=tau), _official(gt, pred, tau)
    for key in ["TI-HOTA", "TI-DetA", "TI-AssA", "TI-TP", "TI-FP", "TI-FN"]:
        assert ours[key] == pytest.approx(ref[key], abs=1e-9), key


def test_sanity():
    rng = np.random.default_rng(0)
    gt = _random_tracks(rng, 30, IDS)
    assert ti_hota(gt, gt)["TI-HOTA"] == pytest.approx(1.0)

    swapped = gt.copy()
    swapped["team"] = np.where(gt["team"] == "A", "B", "A")
    assert ti_hota(gt, swapped)["TI-HOTA"] == 0.0

    shifted = gt.copy()
    shifted["x"] += 50.0  # LocSim = 0.05 ** 0.25 ~= 0.47 at tau = 1 m
    assert ti_hota(gt, shifted)["TI-HOTA"] == pytest.approx(9 / 19)  # matched for alpha <= 0.45 only
