"""Fit the GT position convention from pose ankles (training only).

NEGATIVE RESULT (20 Oct): the fitted offset is ~81 cm, mostly the training camera's
own projection of an above-ground point (ankles ~10 cm up, camera ~13 deg elevation),
so it does not transfer to the challenge camera. The pipeline uses bbox bottom-centre
plus a GT-camera bias instead (configs/pipeline.yaml foot.gt_camera_bias_cm). Kept
for the record.

GT court position = the training (top) camera's bbox bottom-centre, which is biased
toward that camera (lowest foot, arm spread). On the challenge view our own box
bottom has a different bias, so instead we estimate a view-independent point — the
ankle midpoint on the ground — and add the top camera's convention offset, fitted
here as a quadratic function of court position:

    GT ≈ H(ankle_mid) + offset(x, y)

    python scripts/fit_foot_convention.py --det-cache /data/trackid3x3/cache/det/yolo11m_pose_<key> \
        [--val-videos IMG_0108 IMG_0109] [--out /data/trackid3x3/models/foot_convention.npz]

Reports, on held-out videos, the error to GT of: detector bbox bottom-centre (the
upper bound for transfer: same camera as GT), raw ankle midpoint, ankle midpoint +
offset. Logged to MLflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np

from bb3x3.data import DATA_ROOT, image_to_court_homography, load_positions, project, training_paths
from bb3x3.detect import load_detections
from bb3x3.foot import ankle_midpoints, quad_features

SCALE_720_TO_4K = np.diag([3.0, 3.0, 1.0])




def matched_samples(video: str, cache_dir: Path, H720: np.ndarray, min_conf: float, max_dist_cm: float = 100.0):
    dets, _, kps = load_detections(cache_dir / f"{video}.npz")
    gt = load_positions(training_paths(video)["positions"])
    bottom = project(H720, np.c_[(dets["x1"] + dets["x2"]) / 2, dets["y2"]])
    ankle_img, ok = ankle_midpoints(kps, min_conf)
    ankle = project(H720, ankle_img)
    keep = ok & (dets["conf"] >= 0.3)
    out = {"gt": [], "bottom": [], "ankle": []}
    gt_by_frame = {f: np.flatnonzero(gt["frame"] == f) for f in np.unique(gt["frame"])}
    for i in np.flatnonzero(keep):
        g = gt_by_frame.get(int(dets["frame"][i]))
        if g is None:
            continue
        gxy = np.c_[gt["x"][g], gt["y"][g]]
        d = np.linalg.norm(gxy - bottom[i], axis=1)
        j = d.argmin()
        if d[j] <= max_dist_cm:
            out["gt"].append(gxy[j])
            out["bottom"].append(bottom[i])
            out["ankle"].append(ankle[i])
    return {k: np.asarray(v) for k, v in out.items()}


def summary(err: np.ndarray) -> dict[str, float]:
    return {"median_cm": float(np.median(err)), "p75_cm": float(np.percentile(err, 75)), "p90_cm": float(np.percentile(err, 90))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--det-cache", type=Path, required=True)
    parser.add_argument("--videos", nargs="+", default=[f"IMG_01{n:02d}" for n in range(4, 12)])
    parser.add_argument("--val-videos", nargs="+", default=["IMG_0108", "IMG_0109"])
    parser.add_argument("--min-ankle-conf", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=DATA_ROOT / "models" / "foot_convention.npz")
    args = parser.parse_args()

    H720 = image_to_court_homography(DATA_ROOT / "training") @ SCALE_720_TO_4K
    data = {v: matched_samples(v, args.det_cache, H720, args.min_ankle_conf) for v in args.videos}
    train = [v for v in args.videos if v not in args.val_videos]
    cat = lambda vs, k: np.concatenate([data[v][k] for v in vs])  # noqa: E731

    def fit(vs):
        X = quad_features(cat(vs, "ankle"))
        Y = cat(vs, "gt") - cat(vs, "ankle")
        coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
        return coef

    coef = fit(train)
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("trackid3x3")
    with mlflow.start_run(run_name="foot_convention"):
        mlflow.set_tag("task", "foot_convention")
        mlflow.log_params({"det_cache": str(args.det_cache), "min_ankle_conf": args.min_ankle_conf, "val_videos": ",".join(args.val_videos)})
        gt, bottom, ankle = cat(args.val_videos, "gt"), cat(args.val_videos, "bottom"), cat(args.val_videos, "ankle")
        corrected = ankle + quad_features(ankle) @ coef
        for name, est in [("bbox_bottom", bottom), ("ankle_raw", ankle), ("ankle_offset", corrected)]:
            s = summary(np.linalg.norm(est - gt, axis=1))
            mlflow.log_metrics({f"val_{name}_{k}": v for k, v in s.items()})
            print(f"{name:13s} " + ", ".join(f"{k} {v:.1f}" for k, v in s.items()))
        mean_offset = (gt - ankle).mean(0)
        print(f"mean offset ankle->GT: ({mean_offset[0]:.1f}, {mean_offset[1]:.1f}) cm, n_val={len(gt)}")
        final = fit(args.videos)  # all training videos for the challenge
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, coef=final, min_ankle_conf=args.min_ankle_conf)
        mlflow.log_artifact(str(args.out))


if __name__ == "__main__":
    main()
