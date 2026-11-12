"""Pixels to millimetres (SWITCH px_to_mm, off by default): report how big a suspicious spot is.

Changelog
  0.1.0  2026-11-12  first version. Adapted from alvik-fleet-capstone tools/recognition/floorcal.py
                     ("rectangle" mode: something rectangular of known size, its 4 corners give the
                     plane's scale). The full floor calibration is kept in tools/recognition/floorcal.py.

The camera looks straight down at one spot, so one scale (mm per pixel) is enough there. Two ways:

  card     lay a flat card of known size (e.g. a bank card, 85.6 x 54.0 mm) on the EMPTY spot and run
           `python -m qc calibrate-mm --card 85.6x54`. The card is found against the empty reference
           (no clicking), its rectangle fitted, mm per pixel = known size / measured size. The two
           sides give two estimates; their disagreement is reported (it shows camera tilt).
  corners  four clicked corners of a rectangle (x1,y1,...,x4,y4) plus its size: a homography, as in
           floorcal.py; mm per pixel is taken at the spot's centre.
"""
import math

import cv2
import numpy as np


def from_card(empty_ref, frame, width_mm, height_mm):
    loc = empty_ref.part_mask(frame)
    if loc is None:
        raise ValueError("No card found at the spot (is the empty reference up to date?)")
    (cx, cy), (w, h), a = cv2.minAreaRect(loc.contour)
    long_px, short_px = max(w, h), min(w, h)
    long_mm, short_mm = max(width_mm, height_mm), min(width_mm, height_mm)
    s1, s2 = long_mm / long_px, short_mm / short_px
    mm = (s1 + s2) / 2
    return dict(mm_per_px=mm, mm_per_px_long=s1, mm_per_px_short=s2,
                disagreement_pct=100 * abs(s1 - s2) / mm, centre=(cx, cy), size_px=(long_px, short_px),
                text="%.4f mm per pixel (long side %.4f, short side %.4f: %.1f %% apart%s)" % (
                    mm, s1, s2, 100 * abs(s1 - s2) / mm,
                    "; more than 2 % suggests the camera is tilted" if abs(s1 - s2) / mm > 0.02 else ""))


def from_corners(corners_px, width_mm, height_mm, at=None):
    """Homography from 4 clicked corners (in order around the rectangle) to mm; mm per pixel at `at`."""
    src = np.float32(corners_px).reshape(4, 2)
    dst = np.float32([[0, 0], [width_mm, 0], [width_mm, height_mm], [0, height_mm]])
    H = cv2.getPerspectiveTransform(src, dst)
    if at is None:
        at = src.mean(0)
    p = np.float32([[at, [at[0] + 1, at[1]], [at[0], at[1] + 1]]])
    q = cv2.perspectiveTransform(p, H)[0]
    mm = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[0])) / 2
    return dict(mm_per_px=float(mm), homography=H.tolist())


def hot_spot_size_mm(heat, threshold, canon_px_to_frame_px, mm_per_px):
    """Longest extent of the area above the threshold in the heat map (canonical px), in mm."""
    if not threshold:
        return None
    m = (heat > threshold).astype(np.uint8)
    if not m.any():
        return 0.0
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    return round(float(max(w, h, 1.0) * canon_px_to_frame_px * mm_per_px), 2)


def bank_card():
    return 85.6, 54.0


def diag(w, h):
    return math.hypot(w, h)
