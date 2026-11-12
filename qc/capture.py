"""Getting pictures: the webcam (settings locked), a simulated camera, and picking the sharpest frame.

Changelog
  0.1.0  2026-11-12  first version.

When the arm says "paused", a short burst of frames is grabbed and the sharpest one is kept (variance of
the Laplacian, measured on the part only, so the black backdrop never decides). The kept frame must also
be still: it is compared with its neighbour in the burst, and if the part moved more than the camera's
own noise allows, the result says so (the arm should wait a moment and ask again).

    cam = open_camera(cfg)                  # webcam with locked exposure / white balance / focus
    cam = SimCamera()                       # synthetic pictures (qc/sim.py); cam.next_part(...) sets the part
    frames = cam.burst(8)
    i, sharp = pick_sharpest(frames, mask)  # index and the sharpness of every frame
"""
import time

import cv2
import numpy as np

from . import sim


def sharpness(frame, mask=None):
    """Variance of the Laplacian (higher = sharper), on the part's bounding box when a mask is given."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if mask is not None and mask.any():
        ys, xs = np.nonzero(mask)
        y0, y1, x0, x1 = max(0, ys.min() - 4), ys.max() + 5, max(0, xs.min() - 4), xs.max() + 5
        g = g[y0:y1, x0:x1]
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def pick_sharpest(frames, mask=None):
    s = [sharpness(f, mask) for f in frames]
    return int(np.argmax(s)), s


def motion(frames, i, mask=None):
    """Mean absolute change of the part area between frame i and the more similar of its two neighbours in
    the burst (a still frame next to the last moving one is still). Overall brightness flicker of the light
    is divided out first: it is not movement."""
    if len(frames) < 2:
        return 0.0
    a = cv2.GaussianBlur(frames[i].astype(np.float32), (0, 0), 1.0)
    out = []
    for j in (i - 1, i + 1):
        if 0 <= j < len(frames):
            b = cv2.GaussianBlur(frames[j].astype(np.float32), (0, 0), 1.0)
            sel = mask > 0 if mask is not None and mask.any() else slice(None)
            b = b * (a[sel].mean() / max(1e-6, b[sel].mean()))
            d = np.abs(a - b).max(axis=2)
            if mask is not None and mask.any():
                d = d[mask > 0]
            out.append(float(d.mean()))
    return min(out)


STILL_FACTOR = 3.0      # [chosen] x the change between two empty pictures; DOE-able (see build report)


def is_still(frames, i, still_noise, mask=None):
    """Still when the frame-to-frame change is within STILL_FACTOR x the change between two pictures of
    the empty, still scene (background.EmptyReference.still_noise)."""
    m = motion(frames, i, mask)
    limit = STILL_FACTOR * still_noise
    return m <= limit, m, limit


class OpenCVCamera:
    """A USB webcam through OpenCV with automatic exposure, white balance and focus turned off.
    Which numbers a driver accepts differs between webcams; SETUP-GUIDE step 2 shows how to find them."""

    def __init__(self, cfg):
        self.cap = cv2.VideoCapture(int(cfg.get("camera_index", 0)))
        if not self.cap.isOpened():
            raise RuntimeError("Camera %s did not open. Is it plugged in, and closed in other programs?"
                               % cfg.get("camera_index", 0))
        c = self.cap
        c.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg.get("frame_width", 640)))
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg.get("frame_height", 480)))
        self.locked = {}
        if cfg.get("lock_camera_settings", True):
            # 0.25 / 1 mean "manual" on V4L2 (Linux) / DirectShow (Windows) respectively; both are tried
            for v in (0.25, 1):
                c.set(cv2.CAP_PROP_AUTO_EXPOSURE, v)
            c.set(cv2.CAP_PROP_EXPOSURE, float(cfg.get("exposure", -6)))
            c.set(cv2.CAP_PROP_AUTO_WB, 0)
            c.set(cv2.CAP_PROP_WB_TEMPERATURE, float(cfg.get("white_balance_k", 4500)))
            c.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            c.set(cv2.CAP_PROP_FOCUS, float(cfg.get("focus", 30)))
            self.locked = {k: c.get(p) for k, p in (("exposure", cv2.CAP_PROP_EXPOSURE),
                                                    ("auto_wb", cv2.CAP_PROP_AUTO_WB),
                                                    ("wb", cv2.CAP_PROP_WB_TEMPERATURE),
                                                    ("autofocus", cv2.CAP_PROP_AUTOFOCUS),
                                                    ("focus", cv2.CAP_PROP_FOCUS))}
        for _ in range(5):              # let the first frames (often dark) pass
            c.read()

    def frame(self):
        ok, f = self.cap.read()
        if not ok:
            raise RuntimeError("The camera stopped giving pictures.")
        return f

    def burst(self, n=8, settle_s=0.0):
        if settle_s:
            time.sleep(settle_s)
        return [self.frame() for _ in range(n)]

    def close(self):
        self.cap.release()


class SimCamera:
    """Synthetic camera for --simulate and tests: shows the empty spot until next_part() puts a part
    there, and goes back to empty after clear()."""

    def __init__(self, seed=0, drift=None):
        self.seed = seed
        self.count = 0
        self.part = None
        self.drift = drift

    def next_part(self, shape, face, defect=None, seed=None, angle=None):
        self.part = dict(shape=shape, face=face, defect=defect, seed=self.seed * 100003 + self.count if seed is None else seed,
                         angle=angle)
        self.count += 1

    def clear(self):
        self.part = None

    def frame(self):
        self.count += 1
        if self.part is None:
            return sim.empty_frame(seed=self.seed * 7 + self.count, drift=self.drift)
        p = self.part
        return sim.render(p["shape"], p["face"], p["defect"], seed=p["seed"], angle=p["angle"], drift=self.drift)["frame"]

    def burst(self, n=8, settle_s=0.0):
        if self.part is None:
            return [self.frame() for _ in range(n)]
        p = self.part
        frames, _ = sim.burst(p["shape"], p["face"], p["defect"], seed=p["seed"], n=n, angle=p["angle"])
        return frames

    def close(self):
        pass


def open_camera(cfg, simulate=False, seed=0):
    return SimCamera(seed) if simulate else OpenCVCamera(cfg)
