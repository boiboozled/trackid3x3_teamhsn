"""Register each challenge video's background plate to the court template (classic CV).

    python scripts/register_court.py [--videos IMG_0114 IMG_0115] [--landmarks /data/trackid3x3/plates/court_landmarks.xml]

Writes <out-dir>/<video>.npy (image px -> court cm), an overlay PNG (red = official
registration, yellow = refined) and report.json; logs everything to MLflow.
With --landmarks (manually clicked court landmarks, CVAT for images 1.1) it also
reports the reprojection error of both homographies at the clicked points.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mlflow
import numpy as np

from bb3x3.court import LANDMARKS
from bb3x3.data import DATA_ROOT, image_to_court_homography, project
from bb3x3.registration import draw_template, register_plate


def load_landmarks(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """{image stem: {landmark name: (x, y)}} from a CVAT for images 1.1 export."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for image in ET.parse(path).getroot().iter("image"):
        stem = Path(image.get("name")).stem
        for pts in image.iter("points"):
            xy = np.array([float(v) for v in pts.get("points").split(";")[0].split(",")])
            out.setdefault(stem, {})[pts.get("label")] = xy
    return out


def court_error_grid(H_a: np.ndarray, H_b: np.ndarray) -> np.ndarray:
    """|H_b(H_a^-1(p)) - p| in cm over a grid of court points: how far positions move between two homographies."""
    gx, gy = np.meshgrid(np.linspace(0, 1505, 31), np.linspace(0, 1105, 23))
    pts = np.c_[gx.ravel(), gy.ravel()]
    img = project(np.linalg.inv(H_a), pts)
    return np.linalg.norm(project(H_b, img) - pts, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", nargs="+", default=["IMG_0114", "IMG_0115"])
    parser.add_argument("--plates", type=Path, default=DATA_ROOT / "plates")
    parser.add_argument("--out-dir", type=Path, default=DATA_ROOT / "registration")
    parser.add_argument("--landmarks", type=Path, default=None)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    H_official = image_to_court_homography(DATA_ROOT / "challenge")
    clicks = load_landmarks(args.landmarks) if args.landmarks and args.landmarks.exists() else {}
    report = {}
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("trackid3x3")
    with mlflow.start_run(run_name="court_registration"):
        mlflow.set_tag("task", "court_registration")
        for video in args.videos:
            plate = cv2.imread(str(args.plates / f"{video}.png"))
            res = register_plate(plate, H_official)
            np.save(args.out_dir / f"{video}.npy", res.H_img_to_court)
            vis = draw_template(draw_template(plate, H_official, (0, 0, 255)), res.H_img_to_court, (0, 255, 255))
            cv2.imwrite(str(args.out_dir / f"{video}_overlay.png"), vis)
            shift = court_error_grid(res.H_img_to_court, H_official)
            entry = {
                "rms_px": res.rms_px,
                "n_points": res.n_points,
                "per_line": res.per_line,
                "official_vs_refined_cm": {"median": float(np.median(shift)), "p90": float(np.percentile(shift, 90)), "max": float(shift.max())},
            }
            if video in clicks:
                names = [n for n in clicks[video] if n in LANDMARKS]
                img = np.array([clicks[video][n] for n in names])
                court = np.array([LANDMARKS[n] for n in names])
                for tag, H in [("official", H_official), ("refined", res.H_img_to_court)]:
                    px = np.linalg.norm(project(np.linalg.inv(H), court) - img, axis=1)
                    cm = np.linalg.norm(project(H, img) - court, axis=1)
                    entry[f"landmarks_{tag}"] = {"n": len(names), "rms_px": float(np.sqrt(np.mean(px**2))),
                                                 "median_cm": float(np.median(cm)), "max_cm": float(cm.max())}
                    mlflow.log_metric(f"{video}/landmarks_{tag}_rms_px", entry[f"landmarks_{tag}"]["rms_px"])
            report[video] = entry
            mlflow.log_metrics({f"{video}/rms_px": res.rms_px, f"{video}/official_shift_median_cm": entry["official_vs_refined_cm"]["median"],
                                f"{video}/official_shift_max_cm": entry["official_vs_refined_cm"]["max"]})
            mlflow.log_artifact(str(args.out_dir / f"{video}_overlay.png"))
            print(video, json.dumps({k: v for k, v in entry.items() if k != "per_line"}, indent=1))
        (args.out_dir / "report.json").write_text(json.dumps(report, indent=1))
        mlflow.log_artifact(str(args.out_dir / "report.json"))


if __name__ == "__main__":
    main()
