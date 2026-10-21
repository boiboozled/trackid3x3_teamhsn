"""Ground-contact point model shared by the convention fit and the pipeline.

GT court position = the GT (top) camera's bbox bottom-centre. From another view we
estimate the ankle midpoint on the ground and add that camera's convention offset,
a quadratic function of court position fitted on training (scripts/fit_foot_convention.py).
"""

from __future__ import annotations

import numpy as np

L_ANKLE, R_ANKLE = 15, 16


def ankle_midpoints(kps: np.ndarray, min_conf: float) -> tuple[np.ndarray, np.ndarray]:
    """Image ankle midpoints (N, 2) and whether both ankles have confidence >= min_conf."""
    ok = (kps[:, L_ANKLE, 2] >= min_conf) & (kps[:, R_ANKLE, 2] >= min_conf)
    return (kps[:, L_ANKLE, :2] + kps[:, R_ANKLE, :2]) / 2, ok


def quad_features(xy: np.ndarray) -> np.ndarray:
    x, y = xy[:, 0] / 1505.0, xy[:, 1] / 1105.0
    return np.c_[np.ones_like(x), x, y, x * y, x**2, y**2]
