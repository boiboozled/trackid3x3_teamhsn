"""3x3 half-court template in the challenge's court coordinates (cm, line centres).

Axes follow ``court_registration.csv``: x runs along the baseline (0..1505), y from
the baseline (y = 0, basket side) to the top / check-ball line (y = 1105). Corners are
the centres of the 5 cm boundary lines, so the inner court is 1500 x 1100 (FIBA 3x3).

Interior markings use FIBA dimensions (key 4.90 m wide x 5.80 m deep to outer
edges, 2-pt arc 6.75 m to the outer edge with straight parts 0.90 m inside the
sidelines, basket centre 1.575 m from the inner baseline edge, no-charge
semicircle 1.25 m to the inner edge) converted to line centres. Registration fits
report residuals per line, which shows if the RKU court deviates from these.
"""

from __future__ import annotations

import numpy as np

W, D = 1505.0, 1105.0
HALF_LINE = 2.5
BASKET = np.array([W / 2, HALF_LINE + 157.5])  # (752.5, 160)
KEY_X = (W / 2 - (245.0 - HALF_LINE), W / 2 + (245.0 - HALF_LINE))  # (510, 995)
FT_Y = HALF_LINE + 580.0 - HALF_LINE  # 580
ARC_R = 675.0 - HALF_LINE  # 672.5
ARC_X = (HALF_LINE + 90.0 + HALF_LINE, W - (HALF_LINE + 90.0 + HALF_LINE))  # (95, 1410)
ARC_JOIN_Y = BASKET[1] + np.sqrt(ARC_R**2 - (BASKET[0] - ARC_X[0]) ** 2)  # ~301
NO_CHARGE_R = 125.0 + HALF_LINE


def _arc(radius: float, start_deg: float, end_deg: float, n: int = 64) -> np.ndarray:
    t = np.deg2rad(np.linspace(start_deg, end_deg, n))
    return BASKET + radius * np.c_[np.cos(t), np.sin(t)]


def _two_point_arc() -> np.ndarray:
    a0 = np.degrees(np.arctan2(ARC_JOIN_Y - BASKET[1], ARC_X[1] - BASKET[0]))
    return _arc(ARC_R, a0, 180.0 - a0)


# Ground-plane line classes as polylines (cm). Names are shared with the labelling
# spec (docs/court_labeling.md) and the registration residual report.
LINES: dict[str, np.ndarray] = {
    "baseline": np.array([[0.0, 0.0], [W, 0.0]]),
    "sideline_x0": np.array([[0.0, 0.0], [0.0, D]]),
    "sideline_x1505": np.array([[W, 0.0], [W, D]]),
    "top_line": np.array([[0.0, D], [W, D]]),
    "key_x510": np.array([[KEY_X[0], 0.0], [KEY_X[0], FT_Y]]),
    "key_x995": np.array([[KEY_X[1], 0.0], [KEY_X[1], FT_Y]]),
    "free_throw_line": np.array([[KEY_X[0], FT_Y], [KEY_X[1], FT_Y]]),
    "arc_straight_x95": np.array([[ARC_X[0], 0.0], [ARC_X[0], ARC_JOIN_Y]]),
    "arc_straight_x1410": np.array([[ARC_X[1], 0.0], [ARC_X[1], ARC_JOIN_Y]]),
    "arc": _two_point_arc(),
    "no_charge": _arc(NO_CHARGE_R, 0.0, 180.0),
}

# Point landmarks: line intersections that can be clicked on a plate.
LANDMARKS: dict[str, tuple[float, float]] = {
    "c_0_0": (0.0, 0.0),
    "c_0_1105": (0.0, D),
    "c_1505_1105": (W, D),
    "c_1505_0": (W, 0.0),
    "base_arc_x95": (ARC_X[0], 0.0),
    "base_arc_x1410": (ARC_X[1], 0.0),
    "base_key_x510": (KEY_X[0], 0.0),
    "base_key_x995": (KEY_X[1], 0.0),
    "ft_key_x510": (KEY_X[0], FT_Y),
    "ft_key_x995": (KEY_X[1], FT_Y),
}

# The official registration ids for the four corners.
OFFICIAL_CORNER_IDS = {"RKU0": "c_0_0", "RKU1": "c_0_1105", "RKU10": "c_1505_0", "RKU11": "c_1505_1105"}


def court_polygon(margin: float = 0.0) -> np.ndarray:
    return np.array([[-margin, -margin], [W + margin, -margin], [W + margin, D + margin], [-margin, D + margin]])
