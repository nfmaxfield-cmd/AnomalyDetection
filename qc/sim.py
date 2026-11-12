"""Computer-generated pictures of the QC station: the stand-in for a webcam in tests and --simulate.

Changelog
  0.1.0  2026-11-12  first version: disc / square / hexagon, top and bottom faces, five defect types,
                     empty frames, four kinds of background drift, bursts with motion blur.

The scene copies the agreed setup: a fixed webcam looking straight down at a fixed spot on a matte black
backdrop, lit by a ring light around the lens. The part lands near the spot (a few pixels of placement
error) at any angle. Small frame-to-frame lighting noise and sensor noise are added, and every good part
differs slightly from the next (size, brightness, surface grain), so "good" is a spread, not one picture.

Two fixture marks (small light dots) and the table edge on the right of the frame are part of the
scene: they let the software notice a bumped camera on an otherwise featureless black backdrop.

    from qc import sim
    f = sim.render("square", "top", seed=3)              # dict(frame, angle, centre, defect_mask, ...)
    f = sim.render("hexagon", "bottom", defect="chip", seed=4)
    e = sim.empty_frame(seed=5, drift="stray_object")
    frames, settled = sim.burst("disc", "top", seed=6)

All numbers in this file describe the made-up scene; none of them is used by the inspection itself.
"""
import math

import cv2
import numpy as np

W, H = 640, 480
SPOT = (320.0, 240.0)
SHAPES = ("disc", "square", "hexagon")
FACES = ("top", "bottom")
DEFECTS = ("scratch", "chip", "dent", "missing_corner", "stain")
DRIFTS = ("light_shift", "stray_object", "camera_bump", "glare")
CANVAS = 320                     # part drawn on a CANVAS x CANVAS square, centred
SS = 4                           # supersampling for smooth outlines
SIZE = {"disc": 108.0, "square": 92.0, "hexagon": 112.0}     # radius / half side / circumradius in px
FIDUCIALS = ((60, 60), (580, 420))
TABLE_X = 612                    # the table shows from this column to the right edge


def _rng(seed):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------------------------------
# backdrop
# ---------------------------------------------------------------------------------------------------
def _backdrop_base():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r2 = (xx - SPOT[0]) ** 2 + (yy - SPOT[1]) ** 2
    v = 13.0 + 9.0 * np.exp(-r2 / (2 * 190.0 ** 2))          # the ring light brightens the middle
    g = _rng(12345)                                            # the cloth's own weave: same every frame
    weave = cv2.GaussianBlur(g.normal(0, 1.6, (H, W)).astype(np.float32), (0, 0), 1.2)
    img = np.repeat((v + weave)[..., None], 3, axis=2)
    img[:, TABLE_X:] = (78, 84, 90)                            # the table beyond the backdrop
    for (x, y) in FIDUCIALS:
        cv2.circle(img, (x, y), 5, (150, 150, 150), -1, cv2.LINE_AA)
    return img


_BACK = None


def backdrop(rng, drift=None):
    global _BACK
    if _BACK is None:
        _BACK = _backdrop_base()
    img = _BACK.copy()
    if drift == "light_shift":                     # the ring light was knocked: brighter one side, dimmer overall
        xx = np.linspace(-1, 1, W, dtype=np.float32)[None, :, None]
        img = img * (0.8 + 0.45 * (xx + 1) / 2)
    elif drift == "stray_object":                  # a screw left on the backdrop
        x, y = int(rng.uniform(90, 200)), int(rng.uniform(380, 440))
        a = rng.uniform(0, 180)
        box = cv2.boxPoints(((x, y), (36, 7), a)).astype(np.int32)
        cv2.fillConvexPoly(img, box, (120, 122, 125), cv2.LINE_AA)
        cv2.circle(img, (int(x + 18 * math.cos(math.radians(a))), int(y + 18 * math.sin(math.radians(a)))),
                   7, (140, 142, 145), -1, cv2.LINE_AA)
    elif drift == "camera_bump":                   # the camera mount was bumped: the whole picture moved
        dx, dy = rng.choice([-1, 1]) * rng.uniform(6, 10), rng.choice([-1, 1]) * rng.uniform(4, 8)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    elif drift == "glare":                         # a reflection from a window or a shiny tool nearby
        x, y = rng.uniform(80, 220), rng.uniform(60, 160)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        img = img + 55.0 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 30.0 ** 2))[..., None]
    elif drift is not None:
        raise ValueError("unknown drift %r" % drift)
    return img


# ---------------------------------------------------------------------------------------------------
# parts
# ---------------------------------------------------------------------------------------------------
def outline_points(shape, scale=1.0):
    """Outline corners (canvas px, centred on 0, 0); a disc returns None."""
    R = SIZE[shape] * scale
    if shape == "square":
        return np.array([[-R, -R], [R, -R], [R, R], [-R, R]], np.float32)
    if shape == "hexagon":
        return np.array([[R * math.cos(math.radians(60 * k)), R * math.sin(math.radians(60 * k))]
                         for k in range(6)], np.float32)
    return None


def _draw_outline(mask, shape, scale, c):
    R = SIZE[shape] * scale
    if shape == "disc":
        cv2.circle(mask, (int(c * SS), int(c * SS)), int(R * SS), 1.0, -1, cv2.LINE_8)
    else:
        pts = ((outline_points(shape, scale) + c) * SS).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1.0, cv2.LINE_8)


def _down(a):
    return cv2.resize(a, (CANVAS, CANVAS), interpolation=cv2.INTER_AREA)


def _circle_mask(cx, cy, r):
    m = np.zeros((CANVAS * SS, CANVAS * SS), np.float32)
    cv2.circle(m, (int(cx * SS), int(cy * SS)), int(r * SS), 1.0, -1, cv2.LINE_8)
    return _down(m)


def _ring_mask(cx, cy, r, width):
    m = np.zeros((CANVAS * SS, CANVAS * SS), np.float32)
    cv2.circle(m, (int(cx * SS), int(cy * SS)), int(r * SS), 1.0, int(width * SS), cv2.LINE_8)
    return _down(m)


def make_part(shape, face, rng):
    """One good part in canonical orientation: (image BGR float32, alpha float32 0..1), CANVAS x CANVAS."""
    c = CANVAS / 2.0
    scale = 1.0 + rng.normal(0, 0.006)                     # +/- ~0.6 % size from part to part
    big = np.zeros((CANVAS * SS, CANVAS * SS), np.float32)
    _draw_outline(big, shape, scale, c)
    alpha = _down(big)
    R = SIZE[shape] * scale
    yy, xx = np.mgrid[0:CANVAS, 0:CANVAS].astype(np.float32)
    rr = np.hypot(xx - c, yy - c)
    base = np.array([168, 171, 176], np.float32) * (1 + rng.normal(0, 0.02))
    shade = 1.0 - 0.07 * (rr / R) ** 2                     # the ring light falls off a little to the edge
    grain = cv2.GaussianBlur(rng.normal(0, 3.5, (CANVAS, CANVAS)).astype(np.float32), (0, 0), 0.8)
    img = base[None, None, :] * shade[..., None] + grain[..., None]
    # rim: the edge chamfer catches the ring light
    dist = cv2.distanceTransform((alpha > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    img += (22.0 * np.clip(1 - dist / 4.0, 0, 1))[..., None]
    j = lambda: rng.normal(0, 0.4)                          # features sit within ~0.4 px of nominal
    holes = np.zeros_like(alpha)

    def darken(m, amount):
        img[:] = img - amount * m[..., None]

    if shape == "disc" and face == "top":
        holes = _circle_mask(c + j(), c + j(), 0.17 * R)
        for f in (0.45, 0.72):
            darken(_ring_mask(c + j(), c + j(), f * R, 3), 38)
    elif shape == "disc" and face == "bottom":
        darken(_ring_mask(c, c, 0.9 * R, 2), 25)
        stamp = np.zeros((CANVAS, CANVAS), np.uint8)
        cv2.putText(stamp, "A7", (int(c + 0.25 * R + j()), int(c + 12 + j())), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, 255, 3, cv2.LINE_AA)
        darken(stamp.astype(np.float32) / 255.0, 45)
    elif shape == "square" and face == "top":
        img += (18.0 * _circle_mask(c + j(), c + j(), 0.38 * R))[..., None]
        darken(_ring_mask(c, c, 0.38 * R, 2.5), 30)
        for sx in (-1, 1):
            for sy in (-1, 1):
                holes = np.maximum(holes, _circle_mask(c + sx * 0.66 * R + j(), c + sy * 0.66 * R + j(), 0.1 * R))
    elif shape == "square" and face == "bottom":
        darken(_ring_mask(c, c, 0.55 * R, 2), 20)
        stamp = np.zeros((CANVAS, CANVAS), np.uint8)
        cv2.putText(stamp, "A7", (int(c - 22 + j()), int(c - 0.62 * R + j())), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, 255, 3, cv2.LINE_AA)
        darken(stamp.astype(np.float32) / 255.0, 45)
    elif shape == "hexagon" and face == "top":
        holes = _circle_mask(c + j(), c + j(), 0.3 * R)
        outside = 1.0 - _circle_mask(c, c, 0.84 * R)
        darken(outside * alpha, 22)                        # chamfer around the nut's corners
        darken(_ring_mask(c, c, 0.42 * R, 3), 30)
    elif shape == "hexagon" and face == "bottom":
        holes = _circle_mask(c + j(), c + j(), 0.3 * R)
        darken(_ring_mask(c, c, 0.62 * R, 2.5), 28)
    alpha = alpha * (1.0 - holes)
    # inner rim of holes also catches the light
    if holes.max() > 0:
        dh = cv2.distanceTransform((holes < 0.5).astype(np.uint8), cv2.DIST_L2, 5)
        img += (16.0 * np.clip(1 - dh / 3.0, 0, 1) * (holes < 0.5))[..., None]
    return img.astype(np.float32), alpha.astype(np.float32)


def _point_on_part(alpha, rng, margin=12):
    inner = cv2.erode((alpha > 0.99).astype(np.uint8), np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    ys, xs = np.nonzero(inner)
    k = rng.integers(len(xs))
    return float(xs[k]), float(ys[k])


def apply_defect(img, alpha, shape, kind, rng, severity=1.0):
    """Damage a canonical part (copies): returns (img, alpha, defect_mask). severity < 1 makes the damage
    smaller AND fainter (scratch / dent / stain contrast scale with it; chips and cuts get smaller)."""
    img = img.copy()
    alpha = alpha.copy()
    c = CANVAS / 2.0
    dm = np.zeros(alpha.shape, np.float32)
    s = severity
    if kind == "scratch":
        x, y = _point_on_part(alpha, rng, 16)
        L = rng.uniform(28, 60) * s
        a = rng.uniform(0, 2 * math.pi)
        bend = rng.normal(0, 0.25)
        pts = []
        for t in np.linspace(-0.5, 0.5, 9):
            aa = a + bend * t
            pts.append([x + t * L * math.cos(aa), y + t * L * math.sin(aa)])
        layer = np.zeros(alpha.shape, np.float32)
        cv2.polylines(layer, [np.int32(np.array(pts) * 4)], False, 1.0, int(rng.integers(1, 3)), cv2.LINE_AA, 2)
        layer *= alpha
        sign = rng.choice([-1, 1])
        img += (sign * rng.uniform(40, 70) * min(1.0, s) * layer)[..., None]
        dm = layer
    elif kind == "chip":
        pts = np.argwhere(cv2.Canny((alpha > 0.5).astype(np.uint8) * 255, 50, 150) > 0)
        rr = np.hypot(pts[:, 0] - c, pts[:, 1] - c)
        pts = pts[rr > rr.max() * 0.8]                       # the outer outline, not a hole's edge
        py, px = pts[rng.integers(len(pts))]
        r = rng.uniform(8, 13) * s
        v = np.array([px - c, py - c]); v /= np.linalg.norm(v) + 1e-6
        bite = _circle_mask(px + v[0] * r * 0.35, py + v[1] * r * 0.35, r)
        dm = bite * alpha
        alpha = alpha * (1 - bite)
        dist = cv2.distanceTransform((bite < 0.5).astype(np.uint8), cv2.DIST_L2, 5)
        img += (25.0 * np.clip(1 - dist / 3.0, 0, 1))[..., None]   # fresh broken edge is bright
    elif kind == "dent":
        x, y = _point_on_part(alpha, rng, 20)
        r = rng.uniform(9, 15) * s
        yy, xx = np.mgrid[0:CANVAS, 0:CANVAS].astype(np.float32)
        a = rng.uniform(0, 2 * math.pi)
        d = ((xx - x) * math.cos(a) + (yy - y) * math.sin(a)) / r
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * r ** 2))
        img += (rng.uniform(30, 48) * min(1.0, s) * d * g * alpha)[..., None]
        dm = (g > 0.3).astype(np.float32) * alpha
    elif kind == "missing_corner":
        pts = outline_points(shape)
        if pts is None:                                    # disc: a flat cut off one side
            a = rng.uniform(0, 2 * math.pi)
            depth = rng.uniform(9, 15) * s
            yy, xx = np.mgrid[0:CANVAS, 0:CANVAS].astype(np.float32)
            proj = (xx - c) * math.cos(a) + (yy - c) * math.sin(a)
            cut = (proj > SIZE[shape] - depth).astype(np.float32)
        else:
            k = rng.integers(len(pts))
            p = pts[k] + c
            prev, nxt = pts[k - 1] + c, pts[(k + 1) % len(pts)] + c
            leg = rng.uniform(16, 26) * s
            a = p + (prev - p) / np.linalg.norm(prev - p) * leg
            b = p + (nxt - p) / np.linalg.norm(nxt - p) * leg
            tri = (np.array([p + (p - c) * 0.2, a, b]) * SS).astype(np.int32)
            big = np.zeros((CANVAS * SS, CANVAS * SS), np.float32)
            cv2.fillConvexPoly(big, tri, 1.0)
            cut = _down(big)
        dm = cut * alpha
        alpha = alpha * (1 - cut)
    elif kind == "stain":
        x, y = _point_on_part(alpha, rng, 18)
        r = rng.uniform(11, 20) * s
        yy, xx = np.mgrid[0:CANVAS, 0:CANVAS].astype(np.float32)
        blob = np.zeros(alpha.shape, np.float32)
        for _ in range(4):
            ox, oy = rng.normal(0, r * 0.4, 2)
            blob += np.exp(-((xx - x - ox) ** 2 + (yy - y - oy) ** 2) / (2 * (r * 0.6) ** 2))
        blob = np.clip(blob / blob.max() * 1.6 - 0.3, 0, 1) * alpha
        strength = rng.uniform(0.7, 1.0) * min(1.0, s)
        tint = np.array([0.72, 0.84, 0.95], np.float32)        # brownish: blue drops most
        img = img * (1 - blob[..., None] * strength * (1 - tint)[None, None, :])
        dm = (blob > 0.2).astype(np.float32)
    else:
        raise ValueError("unknown defect %r" % kind)
    return img, alpha, dm


# ---------------------------------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------------------------------
def _place(back, img, alpha, angle, centre, extra=None):
    c = CANVAS / 2.0
    M = cv2.getRotationMatrix2D((c, c), angle, 1.0)
    M[0, 2] += centre[0] - c
    M[1, 2] += centre[1] - c
    pi = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR)
    pa = cv2.warpAffine(alpha, M, (W, H), flags=cv2.INTER_LINEAR)[..., None]
    out = back * (1 - pa) + pi * pa
    ex = cv2.warpAffine(extra, M, (W, H), flags=cv2.INTER_LINEAR) if extra is not None else None
    return out, pa[..., 0], ex


def _finish(img, rng, gain_sd=0.02, noise_sd=2.0, blur=0.6):
    img = img * (1 + rng.normal(0, gain_sd))                   # small lighting noise frame to frame
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    img = img + rng.normal(0, noise_sd, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def render(shape, face, defect=None, seed=0, angle=None, drift=None, severity=1.0, centre=None,
           motion_px=0.0, motion_angle=0.0, defocus=0.0, noise_seed=None):
    """One picture of one part at the spot. Returns dict(frame uint8 BGR, angle, centre, part_mask,
    defect_mask (frame px, 0..1), shape, face, defect)."""
    if shape not in SHAPES or face not in FACES:
        raise ValueError("unknown shape/face %s/%s" % (shape, face))
    rng = _rng(seed)
    img, alpha = make_part(shape, face, rng)
    dm = None
    if defect:
        img, alpha, dm = apply_defect(img, alpha, shape, defect, rng, severity)
    if angle is None:
        angle = float(rng.uniform(0, 360))
    if centre is None:
        off = np.clip(rng.normal(0, 3.0, 2), -8, 8)            # the arm puts it down within a few px
        centre = (SPOT[0] + off[0], SPOT[1] + off[1])
    back = backdrop(rng, drift)
    if drift == "light_shift":
        img = img * 0.88
    frame, pmask, dmf = _place(back, img, alpha, angle, centre, dm)
    if motion_px > 0:
        k = int(max(3, round(motion_px)) | 1)
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1.0 / k
        R = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), motion_angle, 1.0)
        ker = cv2.warpAffine(ker, R, (k, k))
        ker /= ker.sum()
        frame = cv2.filter2D(frame, -1, ker)
    out = _finish(frame, rng if noise_seed is None else _rng(noise_seed), blur=0.6 + defocus)
    return dict(frame=out, angle=angle, centre=centre, part_mask=pmask, shape=shape, face=face,
                defect=defect, defect_mask=(dmf if dmf is not None else np.zeros((H, W), np.float32)))


def empty_frame(seed=0, drift=None):
    rng = _rng(seed)
    return _finish(backdrop(rng, drift), rng)


def burst(shape, face, defect=None, seed=0, n=8, angle=None):
    """A burst taken right after "paused": the first frames still slide and blur, later ones have settled
    (identical apart from sensor noise). Returns (frames, list of the settled frames' indices)."""
    rng = _rng(seed + 7919)
    base_angle = float(rng.uniform(0, 360)) if angle is None else angle
    k = max(2, n // 2)
    motion = (list(np.linspace(9, 0, k)) + [0.0] * (n - k))[:n]
    settled = [i for i, m in enumerate(motion) if m == 0]
    frames = []
    for i, m in enumerate(motion):
        wob = rng.normal(0, 0.4 * m, 2)          # still sliding a little just after release
        r = render(shape, face, defect, seed=seed, angle=base_angle + 0.05 * m,
                   centre=(SPOT[0] + 2 + wob[0], SPOT[1] - 1 + wob[1]), motion_px=m,
                   motion_angle=float(rng.uniform(0, 180)),
                   noise_seed=seed * 31 + i + 1)        # every frame has its own sensor noise
        frames.append(r["frame"])
    return frames, settled
