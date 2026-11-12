"""The "normal" model for one shape and one face: a PatchCore-style memory of what good parts look like.

Changelog
  0.1.0  2026-11-12  first version: local (same-spot) and global (classic PatchCore) memory banks.

Training (good parts only):
  1. every training picture is straightened (straighten.py) against a reference picture of a good part,
     which is itself refined once: the first picture, then the average of all straightened pictures;
  2. small extra rotations are added (augmentation, settings: train_rotations_deg);
  3. patch features (features.py) of every picture go into the memory bank.

Scoring one picture: each 8 x 8 px cell's feature is compared with the memory; the distance to the
nearest remembered feature is that cell's anomaly. The anomaly map is smoothed a little, kept to the
part (plus a small margin, so a chipped edge still counts), and its maximum is the picture's score.

Two kinds of memory (setting "bank"):
  local   (default) a cell is compared only with the SAME spot (+/- local_window cells) on the good
          parts. Possible because every part is straightened to the same position; it notices "a
          normal-looking feature in the wrong place" (e.g. a hole where none should be).
  global  classic PatchCore: one pool of all good features, thinned to a coreset (greedy farthest-
          point selection). Does not depend on straightening being right; misses misplaced features.
The choice was made from the synthetic results (see the build report); real parts may differ.

Saved as models/<shape>/<face>/model.npz plus meta.json (readable).
"""
import datetime as dt
import json
import os

import cv2
import numpy as np

from . import __version__
from . import straighten as st

PROJ_DIM = 128        # [chosen] features are projected to this many numbers (random projection): 3x less
                      #          memory and work with distances kept to a few %; DOE-able (64/128/256)
MARGIN_CELLS = 2      # [chosen] cells around the part outline still scored (chips, missing corners)


def _proj_matrix(c_in, dim, seed=0):
    if c_in <= dim:
        return None
    g = np.random.default_rng(seed)
    return (g.normal(0, 1, (c_in, dim)) / np.sqrt(dim)).astype(np.float32)


def _gauss(amap, sigma):
    if sigma <= 0:
        return amap
    return cv2.GaussianBlur(amap.astype(np.float32), (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


def coreset(x, k, seed=0, proj_dim=64):
    """Greedy farthest-point selection of k rows of x (PatchCore's coreset), distances measured in a
    random projection to proj_dim numbers for speed."""
    n = len(x)
    if k >= n:
        return np.arange(n)
    p = _proj_matrix(x.shape[1], proj_dim, seed + 1)
    z = x @ p if p is not None else x
    rng = np.random.default_rng(seed)
    sel = [int(rng.integers(n))]
    d = ((z - z[sel[0]]) ** 2).sum(1)
    for _ in range(k - 1):
        i = int(np.argmax(d))
        sel.append(i)
        d = np.minimum(d, ((z - z[i]) ** 2).sum(1))
    return np.array(sel)


def _nn_dist(q, bank, chunk=4096):
    """Distance from each row of q to its nearest row of bank (Euclidean)."""
    bb = (bank ** 2).sum(1)
    out = np.empty(len(q), np.float32)
    for k in range(0, len(q), chunk):
        qq = q[k:k + chunk]
        d = (qq ** 2).sum(1)[:, None] + bb[None, :] - 2 * qq @ bank.T
        out[k:k + chunk] = np.sqrt(np.maximum(d.min(1), 0))
    return out


class NormalModel:
    def __init__(self, shape, face, cfg):
        self.shape, self.face = shape, face
        self.cfg = dict(cfg)
        self.bank_type = cfg.get("bank", "local")
        self.window = int(cfg.get("local_window", 1))
        self.sigma = float(cfg.get("smooth_sigma", 2.0))
        self.out_px = int(cfg.get("canonical_px", 224))
        self.order = None
        self.crop_px = None
        self.ref_img = None
        self.ref_mask = None
        self.region = None
        self.proj = None
        self.mu = None
        self.sd = None
        self.bank = None
        self.backbone_name = None
        self.meta = {}

    # -------------------------------------------------------------------------------------- training
    def prepare(self, frames, empty_ref, backbone=None):
        """Straighten training frames. Returns list of (canonical image, canonical mask)."""
        locs = []
        for f in frames:
            loc = empty_ref.part_mask(f)
            if loc is None:
                continue
            locs.append((f, loc))
        if not locs:
            raise ValueError("no part found in any training picture")
        sym_cfg = (self.cfg.get("symmetry") or {}).get(self.shape)
        syms = [st.outline_symmetry(l.contour, l.centre, sym_cfg) for _, l in locs]
        orders = [s["order"] for s in syms]
        self.order = int(max(set(orders), key=orders.count)) if sym_cfg is None else int(sym_cfg)
        syms = [st.outline_symmetry(l.contour, l.centre, self.order) for _, l in locs]
        rmax = float(np.median([s["max_radius"] for s in syms]))
        self.crop_px = int(round(2 * rmax * float(self.cfg.get("crop_margin", 1.15))))
        span, step = float(self.cfg.get("fine_align_deg", 5.0)), float(self.cfg.get("fine_align_step_deg", 0.5))
        first = st.straighten(locs[0][0], locs[0][1], None, self.crop_px, self.out_px, self.order, sym=syms[0])
        ref, rmask = first["image"], first["mask"]
        for _ in range(2):          # straighten against the reference, then against their average
            outs = [st.straighten(f, l, ref, self.crop_px, self.out_px, self.order, span, step, sym=s,
                                  ref_mask=rmask) for (f, l), s in zip(locs, syms)]
            ref = np.mean([o["image"].astype(np.float32) for o in outs], axis=0).astype(np.uint8)
            rmask = (np.mean([o["mask"] for o in outs], axis=0) > 0.5).astype(np.uint8)
        self.ref_img, self.ref_mask = ref, rmask
        self.meta["content_symmetry"] = st.content_symmetry(ref, self.order, rmask)
        self.meta["align_corr"] = [o["corr"] for o in outs]
        return [(o["image"], o["mask"]) for o in outs]

    def augment(self, canon):
        out = []
        for img, m in canon:
            out.append(img)
            for a in self.cfg.get("train_rotations_deg", []) or []:
                M = cv2.getRotationMatrix2D((self.out_px / 2.0, self.out_px / 2.0), float(a), 1.0)
                out.append(cv2.warpAffine(img, M, (self.out_px, self.out_px), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REPLICATE))
        return out

    def fit(self, canon, backbone):
        """canon: list of (canonical image, mask) from prepare()."""
        imgs = self.augment(canon)
        f = backbone(imgs)                                     # N, C, h, w
        self.backbone_name = backbone.name
        n, c, h, w = f.shape
        self.mu = f.mean(axis=(0, 2, 3)).astype(np.float32)
        self.sd = (f.std(axis=(0, 2, 3)) + 1e-6).astype(np.float32)
        self.proj = _proj_matrix(c, PROJ_DIM)
        z = self._embed(f)                                     # N, h, w, D
        masks = np.stack([m for _, m in canon]).astype(np.float32)
        region = cv2.resize(masks.max(0), (w, h), interpolation=cv2.INTER_AREA) > 0.01
        self.region = cv2.dilate(region.astype(np.uint8), np.ones((2 * MARGIN_CELLS + 1,) * 2, np.uint8)) > 0
        if self.bank_type == "local":
            self.bank = z.astype(np.float16)
        else:
            flat = z[:, self.region].reshape(-1, z.shape[-1])
            k = max(64, int(len(flat) * float(self.cfg.get("coreset_fraction", 0.1))))
            self.bank = flat[coreset(flat, k)].astype(np.float16)
        self.meta.update(dict(n_train=len(canon), n_train_augmented=len(imgs), bank=self.bank_type,
                              backbone=self.backbone_name, trained=dt.datetime.now().isoformat(timespec="seconds"),
                              qc_version=__version__, order=self.order, crop_px=self.crop_px))
        return self

    def _embed(self, f):
        z = (f - self.mu[None, :, None, None]) / self.sd[None, :, None, None]
        z = z.transpose(0, 2, 3, 1)                             # N, h, w, C
        if self.proj is not None:
            z = z @ self.proj
        return z.astype(np.float32)

    # -------------------------------------------------------------------------------------- scoring
    def straighten(self, frame, loc):
        span, step = float(self.cfg.get("fine_align_deg", 5.0)), float(self.cfg.get("fine_align_step_deg", 0.5))
        return st.straighten(frame, loc, self.ref_img, self.crop_px, self.out_px, self.order, span, step,
                             ref_mask=self.ref_mask)

    def anomaly_maps(self, canon_imgs, canon_masks, backbone):
        """Cell anomaly maps (N, h, w) for straightened pictures."""
        f = backbone(canon_imgs)
        z = self._embed(f)
        n, h, w, d = z.shape
        out = np.zeros((n, h, w), np.float32)
        if self.bank_type == "local":
            pad, pn = self._local_cache()                        # M, h+2r, w+2r, d and its squared norms
            r = self.window
            for i in range(n):
                zi = z[i]
                zn = (zi ** 2).sum(-1)
                best = np.full((h, w), np.inf, np.float32)
                for dy in range(2 * r + 1):
                    for dx in range(2 * r + 1):
                        b = pad[:, dy:dy + h, dx:dx + w]                       # M, h, w, d
                        dot = np.einsum("mhwd,hwd->mhw", b, zi, optimize=True)
                        d2 = (pn[:, dy:dy + h, dx:dx + w] - 2 * dot).min(0) + zn
                        best = np.minimum(best, d2)
                out[i] = np.sqrt(np.maximum(best, 0))
        else:
            bank = self._global_cache()
            for i in range(n):
                out[i] = _nn_dist(z[i].reshape(-1, d), bank).reshape(h, w)
        for i in range(n):
            reg = self.region.copy()
            if canon_masks is not None and canon_masks[i] is not None:
                cm = cv2.resize(canon_masks[i].astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0.01
                reg |= cv2.dilate(cm.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            out[i] = _gauss(out[i], self.sigma) * reg
        return out

    def _local_cache(self):
        if getattr(self, "_lc", None) is None:
            r = self.window
            b = self.bank.astype(np.float32)
            pad = np.pad(b, ((0, 0), (r, r), (r, r), (0, 0)), mode="edge")
            self._lc = (pad, (pad ** 2).sum(-1))
        return self._lc

    def _global_cache(self):
        if getattr(self, "_gc", None) is None:
            self._gc = self.bank.astype(np.float32)
        return self._gc

    def score_canonical(self, canon_imgs, canon_masks, backbone):
        maps = self.anomaly_maps(canon_imgs, canon_masks, backbone)
        return maps.reshape(len(maps), -1).max(1), maps

    def heatmap(self, amap):
        """Cell map -> full canonical-size map (float, same units as the score)."""
        return cv2.resize(amap, (self.out_px, self.out_px), interpolation=cv2.INTER_LINEAR)

    # -------------------------------------------------------------------------------------- files
    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        np.savez_compressed(os.path.join(folder, "model.npz"), bank=self.bank, mu=self.mu, sd=self.sd,
                            proj=self.proj if self.proj is not None else np.zeros((0, 0), np.float32),
                            region=self.region, ref_img=self.ref_img, ref_mask=self.ref_mask)
        meta = dict(self.meta, shape=self.shape, face=self.face, order=self.order, crop_px=self.crop_px,
                    out_px=self.out_px, bank=self.bank_type, window=self.window, sigma=self.sigma,
                    backbone=self.backbone_name, cfg={k: v for k, v in self.cfg.items() if not k.startswith("_")})
        with open(os.path.join(folder, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=float)
        return folder

    @classmethod
    def load(cls, folder):
        with open(os.path.join(folder, "meta.json")) as f:
            meta = json.load(f)
        m = cls(meta["shape"], meta["face"], meta.get("cfg", {}))
        z = np.load(os.path.join(folder, "model.npz"))
        m.bank, m.mu, m.sd = z["bank"], z["mu"], z["sd"]
        m.proj = z["proj"] if z["proj"].size else None
        m.region, m.ref_img, m.ref_mask = z["region"], z["ref_img"], z["ref_mask"]
        m.order, m.crop_px, m.out_px = meta["order"], meta["crop_px"], meta["out_px"]
        m.bank_type, m.window, m.sigma = meta["bank"], meta["window"], meta["sigma"]
        m.backbone_name = meta["backbone"]
        m.meta = meta
        return m
