"""How much TI-HOTA does position error alone cost? Perturb training GT and score it.

    python scripts/noise_budget.py [--videos IMG_0109] [--sigmas 0 5 10 20 30 50]

Isotropic Gaussian noise (per row, independent) and a constant bias, with identities
kept perfect, at tau = 100 cm (official notebook) and 50 cm (paper text).
"""

from __future__ import annotations

import argparse

import numpy as np

from bb3x3.data import DATA_ROOT, load_positions, training_paths
from bb3x3.ti_hota import ti_hota


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", nargs="+", default=["IMG_0109"])
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0, 5, 10, 15, 20, 30, 50])
    parser.add_argument("--biases", type=float, nargs="+", default=[10, 20, 30, 50])
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    print("| perturbation | tau=100 | tau=50 |\n|---|---|---|")
    for kind, values in [("noise sigma (cm/axis)", args.sigmas), ("bias (cm, along x)", args.biases)]:
        for v in values:
            scores = {100.0: [], 50.0: []}
            for video in args.videos:
                gt = load_positions(training_paths(video, DATA_ROOT)["positions"])
                pred = gt.copy()
                if kind.startswith("noise"):
                    pred["x"] += rng.normal(0, v, len(pred))
                    pred["y"] += rng.normal(0, v, len(pred))
                else:
                    pred["x"] += v
                for tau in scores:
                    scores[tau].append(ti_hota(gt, pred, tau=tau)["TI-HOTA"])
            print(f"| {kind} = {v:g} | {100 * np.mean(scores[100.0]):.1f} | {100 * np.mean(scores[50.0]):.1f} |")


if __name__ == "__main__":
    main()
