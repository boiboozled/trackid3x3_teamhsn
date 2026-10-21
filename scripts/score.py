"""Score prediction files against training ground truth with TI-HOTA.

    python scripts/score.py --pred-dir outputs/run1 [--videos IMG_0104 IMG_0105] [--tau 100 50]

Prediction files are named <video>.txt in the submission format. Writes a markdown
table to stdout and <pred-dir>/scores.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bb3x3.data import DATA_ROOT, load_games, load_positions, training_paths
from bb3x3.ti_hota import ti_hota


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--videos", nargs="*", help="default: every <video>.txt in --pred-dir with training GT")
    parser.add_argument("--tau", type=float, nargs="+", default=[100.0, 50.0], help="cm; official notebook uses 100")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args()

    games = load_games(args.data_root / "training")
    videos = args.videos or sorted(p.stem for p in args.pred_dir.glob("*.txt") if p.stem in games)
    lines = ["| video | tau | TI-HOTA | DetA | AssA | worst identity (AssA) |", "|---|---|---|---|---|---|"]
    summary: dict[float, list[float]] = {tau: [] for tau in args.tau}
    for video in videos:
        gt = load_positions(training_paths(video, args.data_root)["positions"])
        pred = load_positions(args.pred_dir / f"{video}.txt")
        for tau in args.tau:
            m = ti_hota(gt, pred, tau=tau)
            worst = min(m["TI-AssA_per_identity"].items(), key=lambda kv: kv[1])
            lines.append(
                f"| {video} | {tau:.0f} | {100 * m['TI-HOTA']:.2f} | {100 * m['TI-DetA']:.2f} | "
                f"{100 * m['TI-AssA']:.2f} | {worst[0]} ({100 * worst[1]:.1f}) |"
            )
            summary[tau].append(m["TI-HOTA"])
    for tau, vals in summary.items():
        if vals:
            lines.append(f"| **mean** | {tau:.0f} | **{100 * np.mean(vals):.2f}** ± {100 * np.std(vals):.2f} | | | |")
    table = "\n".join(lines)
    print(table)
    (args.pred_dir / "scores.md").write_text(table + "\n")


if __name__ == "__main__":
    main()
