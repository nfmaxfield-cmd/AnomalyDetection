"""Training, threshold calibration and the learning curve, from the picture folders.

Changelog
  0.1.0  2026-11-12  first version.
  0.2.0  2026-11-12  audit fixes: M8 at least min_holdout (20) pictures held back and the count checked BEFORE
                     a model is saved; M5 every training picture is scored against the others (leave one
                     out) and the ones that look unlike the rest are listed by name; M6 warnings for parts
                     that are small in the picture or hard to tell from the backdrop; H6 the empty reference
                     used for training is kept with the model; H2 the threshold gets the expected defect
                     rate, the false-reject cap and file names; M7 the learning curve judges "enough" by the
                     good-only threshold settling (needs no damaged parts) and says when its damage is painted.

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


def split(paths, holdout_fraction, seed=0, min_holdout=20):
    """Fixed shuffle so the same pictures are held out every time; at least min_holdout held out (audit M8:
    with 12 held-out parts the false-reject rate can be several times the target), never more than half."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(paths))
    k = int(round(len(paths) * holdout_fraction))
    k = max(k, min(int(min_holdout), len(paths) // 2))
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


MIN_CONTRAST = 3.0      # [chosen] part / backdrop difference below 3x the camera noise counts as "hard to see"; not DOE
MIN_DETAIL = 1.0        # [fixed] a crop smaller than the network's 224 px input is enlarged: no detail is added


def setup_warnings(m, cfg):
    """Plain warnings about the camera set-up, from the training pictures (audit M6)."""
    out = []
    if m.meta.get("train_no_part"):
        out.append("no part found in %d training picture(s): %s. If the parts are dark, the black backdrop hides "
                   "them: use a light backdrop for dark parts." % (len(m.meta["train_no_part"]),
                                                                  ", ".join(map(str, m.meta["train_no_part"][:8]))))
    c = [x for x in m.meta.get("train_contrast", []) if x is not None]
    if c and float(np.median(c)) < MIN_CONTRAST:
        out.append("the part is hard to tell from the backdrop (contrast %.1f, want at least %.1f): dark parts need a "
                   "light backdrop (SETUP-GUIDE step 3)." % (float(np.median(c)), MIN_CONTRAST))
    out_px = int(cfg.get("canonical_px", 224))
    if m.crop_px and m.crop_px < MIN_DETAIL * out_px:
        fw = (m.meta.get("frame_size") or [0])[0]
        out.append("the part is small in the picture (%d px across%s): small defects will be missed. Lower the camera "
                   "or use a camera height per size class so each part fills at least a third of the picture's width."
                   % (m.crop_px, ", %.0f %% of the width" % (100.0 * m.crop_px / fw) if fw else ""))
    return out


def train(cfg, root, shape, face, backbone=None, log=print):
    folder = os.path.join(root, "data", "good", shape, face)
    paths = list_images(folder)
    tr, hold = split(paths, float(cfg.get("holdout_fraction", 0.2)), min_holdout=int(cfg.get("min_holdout", 20)))
    if len(paths) < 6 or len(hold) < 3:
        raise ValueError("Only %d good pictures in %s; collect more first (collect-good): at least 6, better 50-100 "
                         "so that 20 can be held back to set the threshold." % (len(paths), folder))
    ref = empty_ref(root)
    backbone = backbone or features.make(cfg, root)
    if getattr(backbone, "warning", None):
        log("WARNING: " + backbone.warning)
    log("Training %s/%s on %d good pictures (%d held back to set the threshold), features: %s" % (
        shape, face, len(tr), len(hold), backbone.name))
    if len(hold) < int(cfg.get("min_holdout", 20)):
        log("NOTE: only %d pictures held back (want %d): the threshold is rough until more good pictures are "
            "collected." % (len(hold), int(cfg.get("min_holdout", 20))))
    m = NormalModel(shape, face, cfg)
    names = [os.path.basename(p) for p in tr]
    canon = m.prepare(read(tr), ref, names=names)
    m.fit(canon, backbone)
    out = os.path.join(root, "models", shape, face)
    kept = [names[k] for k in m.kept_index]
    m.meta["train_files"] = kept
    m.meta["holdout_files"] = [os.path.basename(p) for p in hold]
    loo = m.self_scores(canon, backbone)
    m.meta["train_loo_scores"] = [float(x) for x in loo] if loo is not None else None
    from . import lens as lensmod
    lc = lensmod.load_for(cfg, root)
    m.meta["lens_id"] = lc.id if lc is not None else None              # pictures in data/ were corrected with it
    m.meta["setup_warnings"] = setup_warnings(m, cfg)
    m.save(out)
    ref.save(os.path.join(out, "empty_at_training.npz"))       # the empty spot this model was trained with (H6)
    log("Symmetry of the outline: %s; model saved in %s" % (
        "round" if m.order == 0 else "%d-fold" % m.order, out))
    for w in m.meta["setup_warnings"]:
        log("WARNING: " + w)
    th = calibrate(cfg, root, shape, face, model=m, backbone=backbone, log=log)
    odd = []
    if loo is not None:
        odd = [dict(name=n, score=float(x)) for n, x in zip(kept, loo) if x > th["threshold"]]
        if odd:
            log("CHECK: %d training picture(s) look unlike the other good pictures (each scored against the others "
                "is above the threshold): %s. Open them: a damaged part, an empty spot or another part in the good "
                "folder should be deleted, then train again." % (len(odd), ", ".join(
                    "%s (%.2f)" % (o["name"], o["score"]) for o in sorted(odd, key=lambda o: -o["score"])[:10])))
    return dict(model=out, n_train=len(kept), n_holdout=len(hold), order=m.order, threshold=th,
                odd_training_pictures=odd, setup_warnings=m.meta["setup_warnings"])


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
                           float(cfg.get("cost_bad_accepted", 10.0)), float(cfg.get("cost_good_rejected", 1.0)),
                           expected_defect_rate=float(cfg.get("expected_defect_rate", 0.05)),
                           max_false_reject=cfg.get("max_false_reject"),
                           good_names=[os.path.basename(p) for p, x in zip(hold, good) if np.isfinite(x)],
                           bad_names=[os.path.basename(p) for p, x in zip(bpaths, bad) if np.isfinite(x)])
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
    pool, hold = split(paths, max(0.2, float(cfg.get("holdout_fraction", 0.2))), seed,
                       min_holdout=int(cfg.get("min_holdout", 20)))
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
        bad_kind = "damage painted on held-out good pictures (stand-in, %d)" % len(bad_frames)
        log("NOTE: no damaged pictures yet, so damage is PAINTED on held-out good pictures as a stand-in (strong "
            "contrast: easier to catch than real damage). \"Enough\" is judged by the good-only threshold settling, "
            "which needs no damaged parts.")
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
                       caught_at_threshold=float((b > th["threshold"]).mean()), thr_good=float(th["threshold"]),
                       good_p50=float(np.median(g)), good_max=float(g.max()), bad_min=float(b.min()))
            rows.append(row)
            log("n=%3d draw %d: good-only threshold %.2f, AUROC %.3f, separation %+.2f, damaged caught at that "
                "threshold %.0f %%" % (n, r + 1, row["thr_good"], row["auroc"], row["separation"],
                                       100 * row["caught_at_threshold"]))
    info = dict(shape=shape, face=face, damaged=bad_kind, n_holdout=len(hold_frames), pool=len(pool),
                backbone=backbone.name, suggestion=suggest_n(rows, "thr_good", higher_is_better=False),
                suggestion_damaged=suggest_n(rows))
    return rows, info


METRIC_WORDS = {"caught_at_threshold": "share of damaged parts caught",
                "thr_good": "good-only threshold"}


def suggest_n(rows, metric="caught_at_threshold", higher_is_better=True):
    """The smallest n from which the curve has levelled off: its median, and the median at EVERY larger n,
    is within the draw-to-draw noise of the best median. Noise = the median (over n) of the spread between
    random draws at one n. The data decides; if only the largest n qualifies, the curve may still be
    climbing (or is too noisy to tell): collect more pictures or use more draws.
    metric "thr_good" (higher_is_better=False): the good-only threshold falls as more good pictures show the
    normal spread, and settles once enough are seen; this needs no damaged parts (audit M7)."""
    sign = 1.0 if higher_is_better else -1.0
    words = METRIC_WORDS.get(metric, metric)
    ns = sorted({r["n"] for r in rows})
    vals = {n: [sign * r[metric] for r in rows if r["n"] == n] for n in ns}
    med = {n: float(np.median(v)) for n, v in vals.items()}
    spreads = [float(np.ptp(v)) for v in vals.values() if len(v) > 1]
    noise = float(np.median(spreads)) if spreads else 0.0
    best = max(med.values()) if med else None
    ok = [n for i, n in enumerate(ns) if all(med[m] >= best - noise - 1e-9 for m in ns[i:])]
    if ok and ok[0] != ns[-1]:
        n = ok[0]
        text = ("Enough at %d good pictures: from there on the median %s stays within %.3f (the spread "
                "between random draws) of the best, %.3f." % (n, words, noise, sign * best))
        if n == ns[0] and noise >= best - min(med.values()):
            text += (" The whole curve is flat within that spread, so this measure cannot tell the sizes apart;"
                     " more pictures (or --repeats) make it finer.")
        return dict(n=n, metric=metric, value=sign * med[n], noise=noise, text=text)
    return dict(n=None, metric=metric, value=sign * med[ns[-1]] if ns else None, noise=noise,
                text="Not settled by %s pictures (or too noisy to tell): collect more good pictures, or run the "
                     "curve with more --repeats." % (ns[-1] if ns else "?"))
