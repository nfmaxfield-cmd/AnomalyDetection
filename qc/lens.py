"""Printed checkerboard card: one-time lens calibration (undistortion) and millimetres per pixel at the spot.

Changelog
  0.1.0  2026-11-12  first version (Nick, 12 Nov): printable card, calibration from several tilted views,
                     a data-derived accept limit for the reprojection error, undistortion and mm scale in
                     the pipeline (setting lens_correction; nothing changes while no calibration exists).

Why: a webcam lens bends straight lines, most near the picture's edges, and the part's size in pixels
depends on the camera height. The card fixes both once:
  1. make_card()         a checkerboard to print (PNG and PDF) with its square size printed on it and a
                         100 mm check line (measure it: if it is not 100 mm, the printer scaled the page).
  2. calibrate()         8 or more pictures of the card at the spot, each tilted a little differently. OpenCV
                         finds the inner corners, fits the lens (cv2.calibrateCamera), and reports the
                         reprojection error: how far, in pixels, the fitted lens puts each corner from where
                         it was seen.
  3. Accept limit from data: every view is taken as a short burst; the same corner found in two frames
     of a still card moves only by the camera's noise (the "corner jitter"). A fit cannot be expected to
     be better than that, and a fit far worse than it is wrong (a bent card, a moved card, a bad corner).
     Limit = JITTER_FACTOR x the measured jitter; a view above it is dropped and the fit repeated.
  4. mm per pixel from one more picture of the card lying FLAT at the spot, after undistortion.
     (The part's top face sits higher than the card by the part's thickness: at 40 cm camera height a
     5 mm thick part looks about 1.3 % bigger. The report's sweep lists this.)

Saved as models/camera_calibration.json. The station undistorts every picture (camera, empty reference,
training pictures) with it when the setting lens_correction is on (default) and the file exists; models
remember which calibration they were trained with, and a different one makes the answer "unsure".

    paths = make_card("calibration_card", squares=(9, 6), square_mm=20)
    cal = calibrate(views, (8, 5), 20.0, flat=flat_frames)     # views: list of bursts (lists of frames)
    lc = LensCorrection.load("models/camera_calibration.json"); frame = lc.undistort(frame)
"""
import datetime as dt
import hashlib
import json
import math
import os

import cv2
import numpy as np

JITTER_FACTOR = 3.0         # [chosen] accept limit = 3x the measured corner jitter; DOE not useful (a quality gate)
JITTER_FLOOR = 0.02         # [chosen] px; corner positions are not trusted to be better than this; not DOE
MIN_VIEWS = 8               # [chosen] views needed for a fit (OpenCV's own guidance is 10-20); not DOE
MIN_TILT_SPREAD_DEG = 10.0  # [chosen] the views' tilts must differ by this much, or the focal length is unknown
MIN_COVER = 0.30            # [chosen] share of the picture (4 x 4 grid) the card's corners must touch over all views
FILE = os.path.join("models", "camera_calibration.json")


# ------------------------------------------------------------------------------------------- the card
def card_image(squares=(9, 6), square_mm=20.0, dpi=300, margin_mm=8.0):
    """The checkerboard as a white page (uint8), squares[0] x squares[1] squares of square_mm."""
    px = dpi / 25.4
    cols, rows = squares
    w = int(round((cols * square_mm + 2 * margin_mm) * px))
    h = int(round((rows * square_mm + 2 * margin_mm + 32) * px))
    img = np.full((h, w), 255, np.uint8)
    m = int(round(margin_mm * px))
    s = square_mm * px
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                cv2.rectangle(img, (int(round(m + c * s)), int(round(m + r * s))),
                              (int(round(m + (c + 1) * s)) - 1, int(round(m + (r + 1) * s)) - 1), 0, -1)
    y = int(round(m + rows * s + 8 * px))
    x0 = m
    cv2.line(img, (x0, y), (x0 + int(round(100 * px)), y), 0, max(1, int(px * 0.4)))
    for xx in (x0, x0 + int(round(100 * px))):
        cv2.line(img, (xx, y - int(2 * px)), (xx, y + int(2 * px)), 0, max(1, int(px * 0.4)))
    lines = ["%g mm squares  (%d x %d squares, %d x %d inner corners)" % (square_mm, cols, rows, cols - 1, rows - 1),
             "Print at 100 % / actual size: the line above must measure 100 mm.",
             "Glue flat on stiff card.  python -m qc calibrate-camera --square-mm %g" % square_mm]
    for k, t in enumerate(lines):
        cv2.putText(img, t, (x0, y + int((8 + 6 * k) * px)), cv2.FONT_HERSHEY_SIMPLEX, (0.14 if k == 0 else 0.11) * px,
                    0, max(1, int(round(px * (0.3 if k == 0 else 0.22)))), cv2.LINE_AA)
    return img


PAPER_MM = {"a4": (210.0, 297.0), "letter": (215.9, 279.4)}


def make_card(path_stem, squares=(9, 6), square_mm=20.0, dpi=300, paper="a4"):
    """Writes <stem>.png (300 dpi) and <stem>.pdf: the card at its TRUE size on an A4 or Letter page (print it
    at 100 % / "actual size"). Returns both paths."""
    img = card_image(squares, square_mm, dpi)
    png = path_stem + ".png"
    cv2.imwrite(png, img)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pw, ph = PAPER_MM[paper]
    h, w = img.shape
    w_mm, h_mm = w / dpi * 25.4, h / dpi * 25.4
    if w_mm > pw - 6:
        raise ValueError("the card (%.0f mm wide) does not fit on %s paper: use smaller --square-mm" % (w_mm, paper))
    fig = plt.figure(figsize=(pw / 25.4, ph / 25.4), dpi=dpi)
    ax = fig.add_axes([(pw - w_mm) / 2 / pw, 1 - (12 + h_mm) / ph, w_mm / pw, h_mm / ph])
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest", aspect="auto")
    ax.axis("off")
    pdf = path_stem + ".pdf"
    fig.savefig(pdf, dpi=dpi)
    plt.close(fig)
    return png, pdf


# ------------------------------------------------------------------------------------------- finding corners
def find_corners(frame, pattern):
    """Inner corners (N, 2) float32 of a checkerboard with `pattern` = (inner corners across, down), or None."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    ok, c = cv2.findChessboardCornersSB(g, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if not ok:
        ok, c = cv2.findChessboardCorners(g, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            return None
        c = cv2.cornerSubPix(g, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01))
    return c.reshape(-1, 2).astype(np.float32)


def object_points(pattern, square_mm):
    cols, rows = pattern
    g = np.zeros((rows * cols, 3), np.float32)
    g[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    return g


def corner_jitter(burst_corners):
    """Per-coordinate RMS difference of the same corners between frames of one still view, / sqrt(2): the
    noise of a single corner position (px)."""
    d = []
    for a, b in zip(burst_corners[:-1], burst_corners[1:]):
        d.append(((a - b) ** 2).mean())
    return float(math.sqrt(np.mean(d) / 2.0)) if d else None


def coverage(corner_sets, size, grid=4):
    """Share of a grid x grid split of the picture that the corners touch, over all views."""
    w, h = size
    hit = np.zeros((grid, grid), bool)
    for c in corner_sets:
        gx = np.clip((c[:, 0] / w * grid).astype(int), 0, grid - 1)
        gy = np.clip((c[:, 1] / h * grid).astype(int), 0, grid - 1)
        hit[gy, gx] = True
    return float(hit.mean())


def calibrate(views, pattern, square_mm, flat=None, size=None):
    """views: list of bursts (each a list of 2+ frames of the card held still). flat: frames of the card lying
    flat at the spot (for mm per pixel). Returns a dict (ok, problems, camera_matrix, dist, rms, limit,
    per_view, jitter, coverage, tilt_spread_deg, mm_per_px, ...)."""
    problems = []
    obj = object_points(pattern, square_mm)
    found, jit = [], []
    for k, burst in enumerate(views):
        cs = [find_corners(f, pattern) for f in burst]
        cs = [c for c in cs if c is not None]
        if not cs:
            continue
        found.append((k, cs[0]))
        j = corner_jitter(cs)
        if j is not None:
            jit.append(j)
    if size is None:
        f0 = views[0][0]
        size = (f0.shape[1], f0.shape[0])
    res = dict(time=dt.datetime.now().isoformat(timespec="seconds"), pattern=list(pattern), square_mm=float(square_mm),
               image_size=list(size), views_taken=len(views), views_found=len(found), problems=problems)
    if len(found) < MIN_VIEWS:
        problems.append("the card was found in only %d of %d pictures (need %d): keep the whole card in view, lit "
                        "evenly, not too steep" % (len(found), len(views), MIN_VIEWS))
        res["ok"] = False
        return res
    jitter = float(np.median(jit)) if jit else 0.05
    limit = JITTER_FACTOR * max(jitter, JITTER_FLOOR)
    keep = list(found)
    while True:                                          # fit; drop the worst view while it is above the limit
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera([obj] * len(keep), [c for _, c in keep], size, None, None)
        per = []
        for (k, c), rv, tv in zip(keep, rvecs, tvecs):
            p, _ = cv2.projectPoints(obj, rv, tv, K, dist)
            per.append(float(np.sqrt(((p.reshape(-1, 2) - c) ** 2).sum(1).mean())))
        worst = int(np.argmax(per))
        if per[worst] <= limit or len(keep) <= MIN_VIEWS:
            break
        keep.pop(worst)
    tilts = []
    for rv in rvecs:
        R, _ = cv2.Rodrigues(rv)
        tilts.append(math.degrees(math.acos(float(np.clip(abs(R[2, 2]), -1, 1)))))
    cov = coverage([c for _, c in keep], size)
    res.update(camera_matrix=K.tolist(), dist=dist.ravel().tolist(), rms=float(rms), limit=limit, jitter=jitter,
               per_view=[dict(view=k, error=e) for (k, _), e in zip(keep, per)], views_used=len(keep),
               dropped=[k for k, _ in found if k not in [kk for kk, _ in keep]],
               coverage=cov, tilt_spread_deg=float(max(tilts) - min(tilts)), tilts_deg=tilts)
    if max(per) > limit:
        problems.append("the fit is worse than the camera's own corner noise allows (worst view %.2f px, limit %.2f "
                        "px from a corner jitter of %.3f px): is the card flat and was it held still?" % (
                            max(per), limit, jitter))
    if res["tilt_spread_deg"] < MIN_TILT_SPREAD_DEG:
        problems.append("the views are tilted too alike (spread %.0f degrees, need %.0f): tilt the card more, in "
                        "different directions" % (res["tilt_spread_deg"], MIN_TILT_SPREAD_DEG))
    if cov < MIN_COVER:
        problems.append("the card covered only %.0f %% of the picture over all views (need %.0f %%): move it around "
                        "the spot a little between pictures" % (100 * cov, 100 * MIN_COVER))
    if flat:
        lc = LensCorrection(res)
        sp = []
        for f in flat:
            c = find_corners(lc.undistort(f), pattern)
            if c is None:
                continue
            grid = c.reshape(pattern[1], pattern[0], 2)
            d = np.concatenate([np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel(),
                                np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel()])
            sp.append(float(np.median(d)))
        if sp:
            res["mm_per_px"] = float(square_mm / np.median(sp))
            res["spot_px_per_square"] = float(np.median(sp))
        else:
            problems.append("the card lying flat at the spot was not found: mm per pixel not set")
    res["ok"] = not problems
    return res


def save(res, root="."):
    p = os.path.join(root, FILE)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    body = {k: v for k, v in res.items()}
    body["id"] = hashlib.sha256(json.dumps([res.get("camera_matrix"), res.get("dist")]).encode()).hexdigest()[:16]
    with open(p, "w") as f:
        json.dump(body, f, indent=2, default=float)
    return p


class LensCorrection:
    """Undistortion with a saved calibration. The corrected picture keeps the same size and camera matrix
    (straight lines become straight, sizes at the spot stay comparable)."""

    def __init__(self, res):
        self.res = res
        self.K = np.array(res["camera_matrix"], np.float64)
        self.dist = np.array(res["dist"], np.float64)
        self.size = tuple(res["image_size"])
        self.id = res.get("id") or hashlib.sha256(json.dumps([res.get("camera_matrix"), res.get("dist")]).encode()).hexdigest()[:16]
        self.mm_per_px = res.get("mm_per_px")
        self._maps = None

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def undistort(self, frame):
        h, w = frame.shape[:2]
        if (w, h) != self.size:
            return frame
        if self._maps is None:
            self._maps = cv2.initUndistortRectifyMap(self.K, self.dist, None, self.K, (w, h), cv2.CV_16SC2)
        return cv2.remap(frame, self._maps[0], self._maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def load_for(cfg, root="."):
    """The LensCorrection to use, or None (setting lens_correction off, or no calibration saved)."""
    if not cfg.get("lens_correction", True):
        return None
    p = os.path.join(root, FILE)
    return LensCorrection.load(p) if os.path.exists(p) else None


# ------------------------------------------------------------------------------------------- simulation
SIM_K = np.array([[560.0, 0, 322.0], [0, 560.0, 238.0], [0, 0, 1]])
SIM_DIST = np.array([-0.28, 0.11, 0.0006, -0.0004, 0.0])        # a cheap webcam's barrel distortion


def distort(img, K=SIM_K, dist=SIM_DIST):
    """What a lens with (K, dist) makes of an ideal (undistorted) picture."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    pts = np.stack([xx.ravel(), yy.ravel()], 1).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist, P=K).reshape(h, w, 2)
    return cv2.remap(img, und[..., 0], und[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def sim_view(seed, pattern=(8, 5), square_mm=20.0, tilt_deg=None, flat=False, n=3, size=(640, 480), K=SIM_K,
             dist=SIM_DIST, height_mm=400.0):
    """A burst of n pictures of the card at the spot, seen through a distorting lens: tilted (random
    direction) unless flat. Returns (frames, true rvec, tvec)."""
    rng = np.random.default_rng(seed)
    cols, rows = pattern[0] + 1, pattern[1] + 1
    sq = 24                                                     # px per square on the texture
    tex = np.full(((rows + 2) * sq, (cols + 2) * sq), 235, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                tex[(r + 1) * sq:(r + 2) * sq, (c + 1) * sq:(c + 2) * sq] = 25
    tilt = 0.0 if flat else (rng.uniform(8, 30) if tilt_deg is None else tilt_deg)
    axis_a = rng.uniform(0, 2 * math.pi)
    axis = np.array([math.cos(axis_a), math.sin(axis_a), 0.0]) * math.radians(tilt)
    rz = rng.uniform(-0.3, 0.3)
    R1, _ = cv2.Rodrigues(np.array([0.0, 0.0, rz]))
    R2, _ = cv2.Rodrigues(axis)
    R = R2 @ R1
    rvec, _ = cv2.Rodrigues(R)
    w_mm, h_mm = (cols + 2) * square_mm, (rows + 2) * square_mm
    centre = np.array([w_mm / 2, h_mm / 2, 0.0])
    off = np.array([rng.uniform(-90, 90), rng.uniform(-60, 60), height_mm]) if not flat else np.array([0, 0, height_mm])
    tvec = (off - R @ centre).reshape(3, 1)
    # texture px -> card mm -> picture (ideal pinhole): a homography
    S = np.array([[square_mm / sq, 0, 0], [0, square_mm / sq, 0], [0, 0, 1]])
    Hm = K @ np.column_stack([R[:, 0], R[:, 1], tvec.ravel()]) @ S
    # the card's object points start at the first INNER corner: shift by one square
    tvec_inner = tvec + (R @ np.array([square_mm, square_mm, 0.0])).reshape(3, 1)
    frames = []
    for i in range(n):
        ideal = cv2.warpPerspective(tex, Hm, size, flags=cv2.INTER_AREA, borderValue=18)
        img = cv2.cvtColor(distort(ideal, K, dist), cv2.COLOR_GRAY2BGR).astype(np.float32)
        img = cv2.GaussianBlur(img, (0, 0), 0.7) + np.random.default_rng(seed * 10 + i).normal(0, 2.0, img.shape)
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    return frames, rvec, tvec_inner
