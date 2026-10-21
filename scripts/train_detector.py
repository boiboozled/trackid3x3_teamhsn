"""Fine-tune an Ultralytics detector on the training players; logged to MLflow.

    python scripts/train_detector.py --run-name det_yolo11m_ft [--weights .../yolo11m.pt] [--epochs 30]

Ultralytics' own MLflow callback logs per-epoch metrics into the active run opened here.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
from ultralytics import YOLO, settings

from bb3x3.data import DATA_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--weights", default="/home/mbalu/handball_field_registration/yolo11m.pt")
    parser.add_argument("--data", type=Path, default=DATA_ROOT / "yolo_players" / "data.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", type=Path, default=DATA_ROOT / "models" / "detector")
    args = parser.parse_args()

    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "trackid3x3"
    settings.update({"mlflow": True})
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("trackid3x3")
    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tag("task", "detector_finetune")
        mlflow.log_params({k: str(v) for k, v in vars(args).items()})
        model = YOLO(args.weights)
        model.train(
            data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, project=str(args.project),
            name=args.run_name, exist_ok=True, seed=0, patience=10, cos_lr=True, close_mosaic=5,
            hsv_h=0.01, hsv_s=0.5, hsv_v=0.4, fliplr=0.5, degrees=0.0, perspective=0.0005, workers=6,
        )
        mlflow.set_tag("weights", str(args.project / args.run_name / "weights" / "best.pt"))


if __name__ == "__main__":
    main()
