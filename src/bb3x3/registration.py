"""Static-camera court registration from a background plate, by classic CV.

1. White-line mask: morphological white top-hat on brightness (thin bright lines on
   the blue court and on the light-grey key), low saturation, restricted to near the
   court under the initial homography, thinned to a skeleton.
2. Iterative closest line: each skeleton pixel is assigned to the nearest projected
   template line (bb3x3.court.LINES) within a gate, then the 8-dof homography is
   refined by robust least squares (soft-L1) on point-to-line distances in pixels.
   The gate shrinks over iterations so the initial (possibly bad) homography only has
   to be roughly right.

Same idea as the handball line registrar, but with observations from classic CV
instead of a learned line segmenter (no basketball line labels exist, and one plate
per static video makes per-frame robustness unnecessary).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from bb3x3.court import LINES, court_polygon


@dataclass
class RegistrationResult:
    H_img_to_court: np.ndarray
    rms_px: float
    n_points: int
    per_line: dict[str, tuple[int, float]] = field(default_factory=dict)  # name -> (n points, rms px)
    history: list[tuple[float, int, float]] = field(default_factory=list)  # (gate, n, rms) per stage


def line_mask(plate_bgr: np.ndarray, H_court_to_img: np.ndarray, near_court_cm: float = 150.0,
              tophat_px: int = 9, min_response: int = 18, max_saturation: int = 90,
              hull_dilate_px: int = 5) -> np.ndarray:
    hsv = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[..., 2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tophat_px, tophat_px))
    tophat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
    mask = (tophat >= min_response) & (hsv[..., 1] <= max_saturation)
    region = np.zeros(mask.shape, np.uint8)
    poly = cv2.perspectiveTransform(court_polygon(near_court_cm)[None].astype(np.float64), H_court_to_img)[0]
    cv2.fillPoly(region, [np.round(poly).astype(np.int32)], 1)
    # The playing surface is blue (the key is grey but enclosed): keep the dilated
    # convex hull of the largest blue component, which drops pavement texture.
    blue = ((hsv[..., 0] >= 95) & (hsv[..., 0] <= 130) & (hsv[..., 1] >= 80)).astype(np.uint8) & region
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blue, connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        hull = cv2.convexHull(np.argwhere(labels == largest)[:, ::-1].astype(np.int32))
        court_region = np.zeros(mask.shape, np.uint8)
        cv2.fillPoly(court_region, [hull], 1)
        region = cv2.dilate(court_region, np.ones((2 * hull_dilate_px + 1, 2 * hull_dilate_px + 1), np.uint8))
    mask = (mask & (region > 0)).astype(np.uint8)
    # Drop big blobs (logos, text) — lines are thin and long, logos are compact and wide.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        w, h, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        fill = area / max(w * h, 1)
        keep[i] = area >= 8 and not (fill > 0.25 and min(w, h) > 12)
    mask = keep[labels].astype(np.uint8)
    return cv2.ximgproc.thinning(mask * 255) > 0 if hasattr(cv2, "ximgproc") else mask > 0


def _dense(polyline: np.ndarray, step_cm: float = 2.0) -> np.ndarray:
    pts = [polyline[0]]
    for a, b in zip(polyline[:-1], polyline[1:]):
        n = max(int(np.ceil(np.linalg.norm(b - a) / step_cm)), 1)
        pts.append(a + (b - a) * np.linspace(0, 1, n + 1)[1:, None])
    return np.vstack([np.atleast_2d(p) for p in pts])


def _h_from_params(p: np.ndarray) -> np.ndarray:
    return np.append(p, 1.0).reshape(3, 3)


def _project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(pts[None].astype(np.float64), H)[0]


def register_plate(plate_bgr: np.ndarray, H_img_to_court_init: np.ndarray,
                   gates_px: tuple[float, ...] = (30, 20, 12, 8, 5, 4, 3),
                   lines: dict[str, np.ndarray] = LINES, mask: np.ndarray | None = None) -> RegistrationResult:
    H_c2i = np.linalg.inv(H_img_to_court_init)
    H_c2i /= H_c2i[2, 2]
    if mask is None:
        mask = line_mask(plate_bgr, H_c2i)
    ys, xs = np.nonzero(mask)
    obs = np.c_[xs, ys].astype(np.float64)
    dense = {name: _dense(pl) for name, pl in lines.items()}
    names = list(dense)
    history = []
    params = H_c2i.ravel()[:8].copy()
    assign = np.full(len(obs), -1)
    for gate in gates_px:
        H = _h_from_params(params)
        proj = {n: _project(H, dense[n]) for n in names}
        d = np.stack([cKDTree(proj[n]).query(obs)[0] for n in names], 1)
        assign = np.where(d.min(1) <= gate, d.argmin(1), -1)
        sel = np.flatnonzero(assign >= 0)
        if len(sel) < 20:
            break

        def residuals(p, sel=sel):
            Hp = _h_from_params(p)
            out = np.empty(len(sel))
            for k, n in enumerate(names):
                m = assign[sel] == k
                if m.any():
                    out[m] = cKDTree(_project(Hp, dense[n])).query(obs[sel[m]])[0]
            return out

        fit = least_squares(residuals, params, loss="soft_l1", f_scale=1.5, x_scale="jac", max_nfev=200)
        params = fit.x
        r = residuals(params)
        history.append((float(gate), int(len(sel)), float(np.sqrt(np.mean(np.minimum(r, gate) ** 2)))))

    H_c2i = _h_from_params(params)
    r_all = np.stack([cKDTree(_project(H_c2i, dense[n])).query(obs)[0] for n in names], 1)
    sel = np.flatnonzero(assign >= 0)
    r_sel = r_all[sel, assign[sel]]
    inl = r_sel <= gates_px[-1]
    per_line = {}
    for k, n in enumerate(names):
        m = (assign[sel] == k) & inl
        per_line[n] = (int(m.sum()), float(np.sqrt(np.mean(r_sel[m] ** 2))) if m.any() else float("nan"))
    H_i2c = np.linalg.inv(H_c2i)
    return RegistrationResult(H_i2c / H_i2c[2, 2], float(np.sqrt(np.mean(r_sel[inl] ** 2))), int(inl.sum()), per_line, history)


def draw_template(img: np.ndarray, H_img_to_court: np.ndarray, color=(0, 0, 255), lines=LINES) -> np.ndarray:
    out = img.copy()
    H_c2i = np.linalg.inv(H_img_to_court)
    for pl in lines.values():
        p = _project(H_c2i, _dense(pl, 5.0))
        cv2.polylines(out, [np.round(p).astype(np.int32)], False, color, 1, cv2.LINE_AA)
    return out
