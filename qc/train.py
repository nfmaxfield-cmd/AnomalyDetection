"""Training, threshold calibration and the learning curve, from the picture folders.

Changelog
  0.1.0  2026-11-12  first version.

    res = train(cfg, root, "square", "top")          # model + held-out scores + threshold (from data)
    th = calibrate(cfg, root, "square", "top")       # threshold again (e.g. after adding damaged parts)
    rows = learning_curve(cfg, root, "square", "top", sizes=[5, 10, 20, 40], repeats=3)
"""
import glob
import json
import math
import os

import cv2
import numpy as np

from . import background as bgmod
from . import features
from . import threshold as thmod
from .model import NormalModel


def list_images(folder):
    out = []
    for ext in ("png", "jpg", "jpeg", "bmp"):
        out += glob.glob(os.path.join(folder, "*." + ext))
    return sorted(out)


def read(paths):
    return [cv2.imread(p) for p in paths]


def split(paths, holdout_fraction, seed=0):
    """Fixed shuffle so the same pictures are held out every time; at least 5 held out (when there are
    at least 10 pictures), never more than half."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(paths))
    k = int(round(len(paths) * holdout_fraction))
    if len(paths) >= 10:
        k = max(k, 5)
    k = min(k, len(paths) // 2)
    hold = sorted(paths[i] for i in idx[:k])
    tr = sorted(paths[i] for i in idx[k:])
    return tr, hold


def empty_ref(root):
    p = os.path.join(root, "models", "empty.npz")
    if not os.path.exists(p):
        raise FileNotFoundError("No empty reference (models/empty.npz). Run  python -m qc collect-empty  first.")
    return bgmod.EmptyReference.load(p)


def score_frames(model, ref, frames, backbone):
    """Scores (NaN where no part was found) and the straightened pictures."""
    ims, ms, ok = [], [], []
    for f in frames:
        loc = ref.part_mask(f)
        if loc is None:
            ok.append(False)
            continue
        s = model.straighten(f, loc)
        ims.append(s["image"])
        ms.append(s["mask"])
        ok.append(True)
    sc = np.full(len(frames), np.nan)
    maps = None
    if ims:
        v, maps = model.score_canonical(ims, ms, backbone)
        sc[np.array(ok)] = v
    return sc, maps


def train(cfg, root, shape, face, backbone=None, log=print):
    folder = os.path.join(root, "data", "good", shape, face)
    paths = list_images(folder)
    if len(paths) < 6:
        raise ValueError("Only %d good pictures in %s; collect more first (collect-good)." % (len(paths), folder))
    tr, hold = split(paths, float(cfg.get("holdout_fraction", 0.2)))
    ref = empty_ref(root)
    backbone = backbone or features.make(cfg, root)
    if getattr(backbone, "warning", None):
        log("WARNING: " + backbone.warning)
    log("Training %s/%s on %d good pictures (%d held back to set the threshold), features: %s" % (
        shape, face, len(tr), len(hold), backbone.name))
    m = NormalModel(shape, face, cfg)
    canon = m.prepare(read(tr), ref)
    m.fit(canon, backbone)
    out = os.path.join(root, "models", shape, face)
    m.meta["train_files"] = [os.path.basename(p) for p in tr]
    m.meta["holdout_files"] = [os.path.basename(p) for p in hold]
    m.save(out)
    log("Symmetry of the outline: %s; model saved in %s" % (
        "round" if m.order == 0 else "%d-fold" % m.order, out))
    th = calibrate(cfg, root, shape, face, model=m, backbone=backbone, log=log)
    return dict(model=out, n_train=len(tr), n_holdout=len(hold), order=m.order, threshold=th)


def calibrate(cfg, root, shape, face, model=None, backbone=None, log=print):
    folder = os.path.join(root, "models", shape, face)
    m = model or NormalModel.load(folder)
    backbone = backbone or features.make(cfg, root)
    ref = empty_ref(root)
    gdir = os.path.join(root, "data", "good", shape, face)
    hold = [os.path.join(gdir, f) for f in m.meta.get("holdout_files", [])]
    hold = [p for p in hold if os.path.exists(p)]
    good, _ = score_frames(m, ref, read(hold), backbone)
    bdir = os.path.join(root, "data", "damaged", shape, face)
    bpaths = list_images(bdir)
    bad, _ = score_frames(m, ref, read(bpaths), backbone) if bpaths else (np.array([]), None)
    missed = [os.path.basename(p) for p, s in zip(bpaths, bad) if not np.isfinite(s)]
    th = thmod.from_scores(good[np.isfinite(good)], bad[np.isfinite(bad)],
                           float(cfg.get("target_false_reject", 0.01)),
                           float(cfg.get("cost_bad_accepted", 10.0)), float(cfg.get("cost_good_rejected", 1.0)))
    th.update(shape=shape, face=face, good_scores=[float(x) for x in good],
              damaged_scores={os.path.basename(p): float(s) for p, s in zip(bpaths, bad)},
              damaged_no_part=missed)
    with open(os.path.join(folder, "threshold.json"), "w") as f:
        json.dump(th, f, indent=2, default=float)
    log(th["text"])
    return th


# ------------------------------------------------------------------------------------- learning curve
def paint_damage(frame, mask, rng):
    """Stand-in damage painted on a GOOD picture, for the learning curve before any damaged parts exist:
    a scratch, a dark blob, or a bite out of the edge (backdrop colour). A proxy, and labelled as one."""
    f = frame.copy()
    ys, xs = np.nonzero(cv2.erode(mask, np.ones((25, 25), np.uint8)))
    if len(xs) == 0:
        ys, xs = np.nonzero(mask)
    k = rng.integers(len(xs))
    x, y = int(xs[k]), int(ys[k])
    kind = ["scratch", "blob", "bite"][rng.integers(3)]
    if kind == "scratch":
        a = rng.uniform(0, math.pi)
        L = rng.uniform(25, 50)
        p1 = (int(x - L / 2 * math.cos(a)), int(y - L / 2 * math.sin(a)))
        p2 = (int(x + L / 2 * math.cos(a)), int(y + L / 2 * math.sin(a)))
        layer = np.zeros(mask.shape, np.uint8)
        cv2.line(layer, p1, p2, 255, 2, cv2.LINE_AA)
        d = (layer.astype(np.float32) / 255 * (mask > 0))[..., None] * rng.choice([-55, 55])
        f = np.clip(f.astype(np.float32) + d, 0, 255).astype(np.uint8)
    elif kind == "blob":
        layer = np.zeros(mask.shape, np.float32)
        cv2.circle(layer, (x, y), int(rng.uniform(7, 12)), 1.0, -1, cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (0, 0), 2) * (mask > 0)
        f = np.clip(f.astype(np.float32) * (1 - 0.35 * layer[..., None]), 0, 255).astype(np.uint8)
    else:
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea).reshape(-1, 2)
        px, py = c[rng.integers(len(c))]
        bite = np.zeros(mask.shape, np.uint8)
        cv2.circle(bite, (int(px), int(py)), int(rng.uniform(8, 12)), 1, -1)
        back = np.median(frame[mask == 0], axis=0)
        f[bite > 0] = back.astype(np.uint8)
    return f, kind


def auroc(good, bad):
    g, b = np.asarray(good, float), np.asarray(bad, float)
    if len(g) == 0 or len(b) == 0:
        return float("nan")
    return float(((b[:, None] > g[None, :]).sum() + 0.5 * (b[:, None] == g[None, :]).sum()) / (len(g) * len(b)))


def separation(good, bad):
    """Gap between the damaged and the good scores in units of the good scores' spread:
    (lowest damaged - highest good) / spread of good. Positive = every damaged part above every good one."""
    g, b = np.asarray(good, float), np.asarray(bad, float)
    sd = g.std() + 1e-9
    return float((b.min() - g.max()) / sd)


def learning_curve(cfg, root, shape, face, sizes=(5, 10, 20, 40, 80), repeats=3, backbone=None, log=print,
                   seed=0):
    """Train on n good pictures (n in sizes, `repeats` random draws each) and score the same held-out good
    and damaged pictures every time. Damaged: data/damaged/<shape>/<face>/ if present, else damage painted
    onto held-out good pictures (proxy). Returns (rows, info)."""
    gdir = os.path.join(root, "data", "good", shape, face)
    paths = list_images(gdir)
    pool, hold = split(paths, max(0.2, float(cfg.get("holdout_fraction", 0.2))), seed)
    ref = empty_ref(root)
    backbone = backbone or features.make(cfg, root)
    hold_frames = read(hold)
    bpaths = list_images(os.path.join(root, "data", "damaged", shape, face))
    rng = np.random.default_rng(seed + 1)
    if bpaths:
        bad_frames, bad_kind = read(bpaths), "real damaged parts (%d)" % len(bpaths)
    else:
        bad_frames = []
        for f in hold_frames:
            loc = ref.part_mask(f)
            if loc is not None:
                bad_frames.append(paint_damage(f, loc.mask, rng)[0])
        bad_kind = "damage painted on held-out good pictures (proxy, %d)" % len(bad_frames)
    sizes = [n for n in sizes if n <= len(pool)]
    rows = []
    for n in sizes:
        for r in range(repeats if n < len(pool) else 1):
            pick = sorted(np.random.default_rng(seed * 1000 + n * 10 + r).choice(len(pool), n, replace=False))
            m = NormalModel(shape, face, cfg)
            canon = m.prepare(read([pool[i] for i in pick]), ref)
            m.fit(canon, backbone)
            g, _ = score_frames(m, ref, hold_frames, backbone)
            b, _ = score_frames(m, ref, bad_frames, backbone)
            g, b = g[np.isfinite(g)], b[np.isfinite(b)]
            th = thmod.from_scores(g, None, float(cfg.get("target_false_reject", 0.01)))
            row = dict(n=n, repeat=r, auroc=auroc(g, b), separation=separation(g, b),
                       caught_at_threshold=float((b > th["threshold"]).mean()),
                       good_p50=float(np.median(g)), good_max=float(g.max()), bad_min=float(b.min()))
            rows.append(row)
            log("n=%3d draw %d: AUROC %.3f, separation %+.2f, damaged caught at the good-only threshold %.0f %%" % (
                n, r + 1, row["auroc"], row["separation"], 100 * row["caught_at_threshold"]))
    info = dict(shape=shape, face=face, damaged=bad_kind, n_holdout=len(hold_frames), pool=len(pool),
                backbone=backbone.name, suggestion=suggest_n(rows))
    return rows, info


def suggest_n(rows, metric="caught_at_threshold"):
    """The smallest n from which the curve has levelled off: its median, and the median at EVERY larger n,
    is within the draw-to-draw noise of the best median. Noise = the median (over n) of the spread between
    random draws at one n. The data decides; if only the largest n qualifies, the curve may still be
    climbing (or is too noisy to tell): collect more pictures or use more draws."""
    ns = sorted({r["n"] for r in rows})
    vals = {n: [r[metric] for r in rows if r["n"] == n] for n in ns}
    med = {n: float(np.median(v)) for n, v in vals.items()}
    spreads = [float(np.ptp(v)) for v in vals.values() if len(v) > 1]
    noise = float(np.median(spreads)) if spreads else 0.0
    best = max(med.values()) if med else None
    ok = [n for i, n in enumerate(ns) if all(med[m] >= best - noise - 1e-9 for m in ns[i:])]
    if ok and ok[0] != ns[-1]:
        n = ok[0]
        text = ("Enough at %d good pictures: from there on the median %s stays within %.3f (the spread "
                "between random draws) of the best, %.3f." % (n, metric, noise, best))
        if n == ns[0] and noise >= best - min(med.values()):
            text += (" The whole curve is flat within that spread, so this measure cannot tell the sizes apart;"
                     " more damaged pictures (or --repeats) make it finer.")
        return dict(n=n, metric=metric, value=med[n], noise=noise, text=text)
    return dict(n=None, metric=metric, value=med[ns[-1]] if ns else None, noise=noise,
                text="Not settled by %s pictures (or too noisy to tell): collect more good pictures, or run the "
                     "curve with more --repeats." % (ns[-1] if ns else "?"))
