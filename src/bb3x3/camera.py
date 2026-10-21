"""Single-view camera calibration from the court homography.

A static camera sees one plane, so the multi-view (Zhang) calibration in the handball
project cannot run. With zero skew, square pixels and the principal point at the
image centre, the plane homography alone fixes the focal length (the two rotation
constraints r1.r2 = 0 and |r1| = |r2|), then rotation and camera centre follow.
That gives the camera height and the vertical direction, so image points at a known
height above the court (e.g. ankles) can be dropped to the ground.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Camera:
    K: np.ndarray
    R: np.ndarray  # world (court cm, z = x cross y) -> camera
    t: np.ndarray
    up: float  # +1 if the camera is on the +z side of the court plane, else -1

    @property
    def centre(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def height_cm(self) -> float:
        return float(self.up * self.centre[2])

    def ground_point(self, pixels: np.ndarray, height_cm: float | np.ndarray = 0.0) -> np.ndarray:
        """Court (x, y) of the points seen at `pixels` that lie `height_cm` above the court."""
        pixels = np.atleast_2d(pixels).astype(np.float64)
        rays = (self.R.T @ np.linalg.inv(self.K) @ np.c_[pixels, np.ones(len(pixels))].T).T
        c = self.centre
        lam = (self.up * np.asarray(height_cm) - c[2]) / rays[:, 2]
        return (c[None] + lam[:, None] * rays)[:, :2]

    def project(self, points_xyz: np.ndarray) -> np.ndarray:
        p = (self.K @ (self.R @ np.atleast_2d(points_xyz).T + self.t[:, None])).T
        return p[:, :2] / p[:, 2:]


def focal_from_homography(H_court_to_img: np.ndarray, principal_point: tuple[float, float]) -> float:
    cx, cy = principal_point
    h = H_court_to_img / H_court_to_img[2, 2]
    a = h[0, :2] - cx * h[2, :2]
    b = h[1, :2] - cy * h[2, :2]
    c = h[2, :2]
    estimates = []
    # r1 . r2 = 0
    den = c[0] * c[1]
    if abs(den) > 1e-12:
        estimates.append(-(a[0] * a[1] + b[0] * b[1]) / den)
    # |r1| = |r2|
    den = c[1] ** 2 - c[0] ** 2
    if abs(den) > 1e-12:
        estimates.append((a[0] ** 2 + b[0] ** 2 - a[1] ** 2 - b[1] ** 2) / den)
    f2 = [e for e in estimates if e > 0]
    if not f2:
        raise ValueError("homography gives no positive focal length")
    return float(np.sqrt(np.mean(f2)))


def _pnp(f: float, w: int, h: int, court: np.ndarray, image: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    obj = np.c_[court, np.zeros(len(court))]
    ok, rvec, tvec = cv2.solvePnP(obj, image, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - image) ** 2, axis=1))))
    return rms, K, cv2.Rodrigues(rvec)[0], tvec.ravel()


def calibrate(H_img_to_court: np.ndarray, image_size: tuple[int, int]) -> tuple[Camera, float]:
    """Pinhole camera reproducing the homography on a court grid (+ 3 m margin).

    `image_size` = (width, height); principal point at the centre. The focal length
    is a bounded 1-D search on PnP reprojection error, seeded by the closed form.
    Returns the camera and its reprojection RMS in pixels (how well a pinhole camera
    with these assumptions explains the homography).
    """
    from scipy.optimize import minimize_scalar

    w, h = image_size
    gx, gy = np.meshgrid(np.linspace(-300, 1805, 15), np.linspace(-300, 1405, 12))
    court = np.c_[gx.ravel(), gy.ravel()]
    H_c2i = np.linalg.inv(H_img_to_court)
    image = cv2.perspectiveTransform(court[None].astype(np.float64), H_c2i)[0]
    try:
        f0 = focal_from_homography(H_c2i, (w / 2, h / 2))
    except ValueError:
        f0 = float(w)
    res = minimize_scalar(lambda f: _pnp(f, w, h, court, image)[0], bounds=(0.3 * f0, 3.0 * f0), method="bounded")
    rms, K, R, t = _pnp(res.x, w, h, court, image)
    cam = Camera(K, R, t, up=1.0)
    cam.up = 1.0 if cam.centre[2] > 0 else -1.0
    return cam, rms
