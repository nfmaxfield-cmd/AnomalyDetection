"""Straighten a part: turn it to a standard angle and cut out a standard-size square around it.

Changelog
  0.1.0  2026-11-12  first version.
  0.1.1  2026-11-12  chosen numbers marked [chosen] (audit L7); no change in behaviour.

Steps, for a part found against the empty reference (background.py):
  1. Outline -> radius at every angle around the part's centre, r(theta).
  2. Symmetry of the outline from the Fourier harmonics of r(theta): a square's radius repeats 4 times per
     turn, a hexagon's 6 times, a disc's not at all. The smallest strong harmonic is the symmetry order n
     (0 = round). It can also be set per shape in the settings instead of detected.
  3. Coarse angle: the phase of harmonic n puts a corner at angle 0. Any of the n turns by 360/n gives
     the same outline, so each is tried in step 5 (that is where a marking on the face, e.g. a stamp on
     one side of a square, decides which corner is "first"). A round part has no outline angle: the
     whole turn is searched against the face's picture.
  4. Crop: rotate about the centre and cut a square of the model's crop size, resized to
     canonical_px (224 for the backbone).
  5. Fine alignment: each candidate is turned within +/-fine_align_deg in fine steps (and shifted, by
     phase correlation) to best match the reference picture of a good part. The best match wins.

    sym = outline_symmetry(contour, centre)              # dict(order, angle_deg, strength)
    canon, cmask, M = crop(frame, mask, centre, angle_deg, crop_px, out_px)
    res = straighten(frame, loc, ref_img, crop_px, out_px, order)   # dict(image, mask, angle, M, score)
"""
import math

import cv2
import numpy as np

HARMONICS = range(2, 13)       # symmetry orders looked for
SIG_FACTOR = 8.0               # [chosen] a harmonic counts when this many times the noise floor
MIN_STRENGTH = 0.004           # [chosen] ... and at least 0.4 % of the mean radius
ROUND_STEP_DEG = 4.0           # [chosen] coarse step when a round part's whole turn is searched


def radius_profile(contour, centre, bins=360):
    """Largest radius of the outline in each of `bins` angle bins (image angle, y down)."""
    p = contour.reshape(-1, 2).astype(np.float64)
    dx, dy = p[:, 0] - centre[0], p[:, 1] - centre[1]
    th = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    r = np.hypot(dx, dy)
    idx = np.minimum((th / (2 * np.pi) * bins).astype(int), bins - 1)
    prof = np.full(bins, np.nan)
    np.fmax.at(prof, idx, r)
    ok = np.isfinite(prof)
    if ok.sum() < bins // 4:                    # [chosen] a quarter of the angle bins must be seen; not DOE
        raise ValueError("outline too small to measure")
    x = np.arange(bins)
    prof = np.interp(x, x[ok], prof[ok], period=bins)
    return prof


def outline_symmetry(contour, centre, order=None):
    """Symmetry order n of the outline (0 = round) and the angle (deg, image convention) that puts the
    first corner at angle 0. order given: use it, only measure the angle."""
    r = radius_profile(contour, centre)
    F = np.fft.rfft(r - r.mean())
    mags = np.abs(F) / len(r) * 2 / r.mean()
    noise = float(np.median(mags[13:60])) + 1e-9    # [chosen] harmonics 13-59 = the outline's noise floor; not DOE
    if order is None:
        order = 0
        for k in HARMONICS:
            if mags[k] > SIG_FACTOR * noise and mags[k] > MIN_STRENGTH:
                order = k
                break
    angle = 0.0
    if order and order > 0:
        phi = -np.angle(F[order])
        angle = math.degrees(phi / order)
    return dict(order=int(order), angle_deg=float(angle),
                strength=float(mags[order]) if order else 0.0, noise=noise,
                mean_radius=float(r.mean()), max_radius=float(r.max()))


def crop_matrix(centre, angle_deg, crop_px, out_px):
    """Affine matrix: frame -> canonical square (part centre in the middle, outline angle at 0)."""
    s = out_px / float(crop_px)
    M = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), angle_deg, s)
    M[0, 2] += out_px / 2.0 - centre[0]
    M[1, 2] += out_px / 2.0 - centre[1]
    return M


def crop(frame, mask, centre, angle_deg, crop_px, out_px):
    M = crop_matrix(centre, angle_deg, crop_px, out_px)
    img = cv2.warpAffine(frame, M, (out_px, out_px), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    m = None
    if mask is not None:
        m = cv2.warpAffine(mask.astype(np.uint8), M, (out_px, out_px), flags=cv2.INTER_NEAREST)
    return img, m, M


HIGHPASS_SIGMA = 6.0           # [chosen] canonical px; keeps the face's details, drops the overall shape
INTERIOR_ERODE = 9             # [chosen] canonical px; the outline itself is the same for every candidate


def interior(mask):
    """The inside of the reference part (outline edge removed): where markings decide the angle."""
    if mask is None:
        return None
    return cv2.erode(mask.astype(np.uint8), np.ones((INTERIOR_ERODE, INTERIOR_ERODE), np.uint8)).astype(np.float32)


GRAIN_SIGMA = 1.0              # [chosen] canonical px; surface grain differs part to part, so it is blurred away
_RADIAL = {}


def _radial_residual(g, mask):
    """Remove what is the same at every angle (rings, round holes): what is left is what tells angles apart."""
    s = g.shape[0]
    if s not in _RADIAL:
        yy, xx = np.mgrid[0:s, 0:s]
        _RADIAL[s] = np.hypot(xx - (s - 1) / 2.0, yy - (s - 1) / 2.0).astype(int)
    r = _RADIAL[s]
    w = mask if mask is not None else np.ones_like(g)
    num = np.bincount(r.ravel(), (g * w).ravel())
    den = np.bincount(r.ravel(), w.ravel()) + 1e-6
    return ((g - (num / den)[r]) * w).astype(np.float32)


def _prep(img, size=112, mask=None, asym=False):
    """Detail picture used for matching: grey, grain blurred away, high-pass, only inside the part when a
    mask is given, optionally only its non-round content (asym), shrunk to size x size, zero mean, unit
    spread."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), GRAIN_SIGMA)
    g = g - cv2.GaussianBlur(g, (0, 0), HIGHPASS_SIGMA)
    if mask is not None:
        g = g * mask
    if asym:
        g = _radial_residual(g, mask)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    g -= g.mean()
    return g / (g.std() + 1e-6)


def _rot(g, a):
    s = g.shape[0]
    M = cv2.getRotationMatrix2D((s / 2.0, s / 2.0), a, 1.0)
    return cv2.warpAffine(g, M, (s, s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


_HANN = {}


def _match(g, ref):
    """Best shift (phase correlation) and the normalised correlation after that shift."""
    s = g.shape[0]
    if s not in _HANN:
        _HANN[s] = cv2.createHanningWindow((s, s), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(ref, g, _HANN[s])
    if abs(dx) > s / 8 or abs(dy) > s / 8:      # [chosen] shifts beyond 1/8 of the picture are false matches; not DOE
        dx = dy = 0.0
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    gs = cv2.warpAffine(g, M, (s, s), borderMode=cv2.BORDER_REPLICATE)
    return float((gs * ref).mean()), dx, dy


def fine_align(img, ref, centre_angle=0.0, span=5.0, step=0.5, size=112, ref_mask=None, asym=False):
    """Search angles centre_angle +/- span against ref (both canonical images). ref_mask: interior()
    of the reference. Returns (best_angle_offset_deg, dx_px, dy_px (canonical px), correlation)."""
    r = _prep(ref, size, ref_mask, asym)
    best = None
    scores = []
    angles = np.arange(-span, span + 1e-9, step) + centre_angle
    s0 = img.shape[0]
    for a in angles:
        M = cv2.getRotationMatrix2D((s0 / 2.0, s0 / 2.0), a, 1.0)
        ia = cv2.warpAffine(img, M, (s0, s0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        c, dx, dy = _match(_prep(ia, size, ref_mask, asym), r)
        scores.append(c)
        if best is None or c > best[3]:
            best = (float(a), dx, dy, c)
    # parabola through the best and its neighbours: sub-step angle
    i = int(np.argmax(scores))
    a = best[0]
    if 0 < i < len(scores) - 1:
        y0, y1, y2 = scores[i - 1], scores[i], scores[i + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            a = best[0] + 0.5 * step * (y0 - y2) / den
    k = img.shape[0] / float(size)
    return a, best[1] * k, best[2] * k, best[3]


def straighten(frame, loc, ref_img, crop_px, out_px, order, fine_span=5.0, fine_step=0.5, sym=None, ref_mask=None):
    """Full straightening of one located part. ref_img None (the first training picture): only the
    outline angle is used. ref_mask: the reference part's mask (canonical). Returns dict(image, mask,
    angle_deg, M, corr, order, candidates, centre)."""
    inner = interior(ref_mask)
    if sym is None:
        sym = outline_symmetry(loc.contour, loc.centre, order)
    order = sym["order"]
    base = sym["angle_deg"]
    if ref_img is None:
        img, m, M = crop(frame, loc.filled, loc.centre, base, crop_px, out_px)
        return dict(image=img, mask=m, angle_deg=base, M=M, corr=None, order=order, candidates=1)
    # candidates: each equivalent outline angle, or the whole turn for a round part. The choice between
    # them looks only INSIDE the part (markings, holes); the outline is the same for all of them.
    def inner_score(a, span=0.0, step=0.5, size=112):
        im = crop(frame, None, loc.centre, a, crop_px, out_px)[0]
        if span <= 0:
            return _match(_prep(im, size, inner, True), r_in)[0], a
        da, _, _, c = fine_align(im, ref_img, 0.0, span, step, size, inner, asym=True)
        return c, a + da

    r_in = _prep(ref_img, 112, inner, True)
    if order > 0:
        cands = [base + k * 360.0 / order for k in range(order)]
        scored = [inner_score(a) for a in cands]
    else:
        im0 = crop(frame, None, loc.centre, 0.0, crop_px, out_px)[0]
        sc = []
        for a in np.arange(0, 360, ROUND_STEP_DEG):
            M = cv2.getRotationMatrix2D((out_px / 2.0, out_px / 2.0), a, 1.0)
            ia = cv2.warpAffine(im0, M, (out_px, out_px), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            sc.append((_match(_prep(ia, 112, inner, True), r_in)[0], float(a)))
        sc.sort(reverse=True)
        cands = [a for _, a in sc[:3]]          # [chosen] best 3 coarse angles refined; not DOE
        scored = [inner_score(a, ROUND_STEP_DEG, 0.5) for a in cands]
    inner_corr, a_best = max(scored)
    # final fine alignment on the whole detail picture: the outline fixes the angle where the face can't
    img0, _, _ = crop(frame, None, loc.centre, a_best, crop_px, out_px)
    da, dx, dy, c = fine_align(img0, ref_img, 0.0, fine_span if order > 0 else 1.0, fine_step)
    best = (a_best, da, dx, dy, c)
    a0, da, dx, dy, c = best
    angle = a0 + da
    # shift: (dx, dy) measured in the canonical picture after turning by angle; move the centre there
    s = crop_px / float(out_px)
    t = math.radians(angle)
    # canonical +x/+y expressed in frame px (cv2 turns counter-clockwise on screen for positive angles)
    ux = (math.cos(t), -math.sin(t))
    uy = (math.sin(t), math.cos(t))
    cx = loc.centre[0] + s * (dx * ux[0] + dy * uy[0])
    cy = loc.centre[1] + s * (dx * ux[1] + dy * uy[1])
    img, m, M = crop(frame, loc.filled, (cx, cy), angle, crop_px, out_px)
    return dict(image=img, mask=m, angle_deg=float(angle), M=M, corr=float(c), order=order,
                candidates=len(cands), centre=(cx, cy), inner_corr=float(inner_corr))


def content_symmetry(ref_img, order, ref_mask=None):
    """How well the reference face matches itself after each 1/order turn (1.0 = identical). A square
    whose face has a stamp on one side shows clearly less than 1: the stamp decides the angle."""
    if order <= 1:
        return []
    g = _prep(ref_img, 112, interior(ref_mask))
    return [float((_rot(g, k * 360.0 / order) * g).mean()) for k in range(1, order)]
