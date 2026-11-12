"""The "normal" model for one shape and one face: a PatchCore-style memory of what good parts look like.

Changelog
  0.1.0  2026-11-12  first version: local (same-spot) and global (classic PatchCore) memory banks.
  0.2.0  2026-11-12  audit fixes: memory laid out cell-first so the same-spot comparison is one batched
                     matrix product per neighbour (M4: ~9x faster, same scores); shifted training copies
                     (M1, setting train_shifts_px); the backbone file's checksum is stored and checked
                     (M4: a model never runs with a different network than it was trained with);
                     training statistics kept for the station's checks (area, sharpness, alignment,
                     M2 / M9); leave-one-out scores of the training pictures (M5).

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

Saved as models/<shape>/<face>/model.npz plus meta.json (readable). meta.json also holds what the
training pictures looked like (part area, sharpness, alignment match, picture size): the station compares
every new picture with these, so its limits come from the training pictures, not from typed-in numbers.
"""
import datetime as dt
import json
import os

import cv2
import numpy as np

from . import __version__
from . import capture
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


def flatten_light(img, mask):
    """Remove a straight brightness slope across the part (light brighter on one side, fixed in the room, so
    it turns with the part's angle in the straightened picture). A plane is fitted to the log-brightness
    inside the part and divided out; the overall level and anything not a straight slope (rings, marks,
    damage) stay. Setting flatten_light."""
    if mask is None or mask.sum() < 50:
        return img
    g = np.log(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) + 8.0)
    m = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    if m.sum() < 50:
        m = mask > 0
    ys, xs = np.nonzero(m)
    c = img.shape[0] / 2.0
    A = np.stack([xs - c, ys - c, np.ones_like(xs)], 1).astype(np.float32)
    coef, *_ = np.linalg.lstsq(A, g[ys, xs], rcond=None)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float32)
    gain = np.exp(-(coef[0] * (xx - c) + coef[1] * (yy - c)))
    return np.clip(img.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)


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
    def prepare(self, frames, empty_ref, backbone=None, names=None):
        """Straighten training frames. Returns list of (canonical image, canonical mask). Pictures where
        no part is found are skipped and listed in meta["train_no_part"] (by name when names are given)."""
        locs = []
        self.kept_index = []
        no_part = []
        for k, f in enumerate(frames):
            loc = empty_ref.part_mask(f)
            if loc is None:
                no_part.append(names[k] if names else k)
                continue
            locs.append((f, loc))
            self.kept_index.append(k)
        self.meta["train_no_part"] = no_part
        if not locs:
            raise ValueError("no part found in any training picture. If the parts are dark, the black backdrop "
                             "hides them: use a light backdrop for dark parts (SETUP-GUIDE step 4); otherwise "
                             "check that the empty picture (collect-empty) was taken with the spot empty.")
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
        raw = locs
        for it in range(2):          # straighten against the reference, then against their average
            outs = [st.straighten(f, l, ref, self.crop_px, self.out_px, self.order, span, step, sym=s,
                                  ref_mask=rmask) for (f, l), s in zip(locs, syms)]
            ref = np.mean([o["image"].astype(np.float32) for o in outs], axis=0).astype(np.uint8)
            rmask = (np.mean([o["mask"] for o in outs], axis=0) > 0.5).astype(np.uint8)
            if it == 0 and self.cfg.get("light_field", False):
                # light fixed in the room (a slope, a reflection) found from the training parts themselves,
                # then divided out of every picture before it is straightened again
                self.light = learn_light_field([f for f, _ in raw], outs, ref, rmask)
                locs = [(self.correct(f), l) for f, l in raw]
        self.ref_img, self.ref_mask = ref, rmask
        self.meta["content_symmetry"] = st.content_symmetry(ref, self.order, rmask)
        self.meta["align_corr"] = [o["corr"] for o in outs]
        # what the training pictures looked like: the station's checks compare new pictures with these
        self.meta["train_area"] = [float(l.area) for _, l in locs]
        self.meta["train_sharpness"] = [capture.sharpness(f, l.filled) for f, l in locs]
        self.meta["train_contrast"] = [part_contrast(empty_ref, f, l) for f, l in locs]
        self.meta["frame_size"] = [int(locs[0][0].shape[1]), int(locs[0][0].shape[0])]
        return [(o["image"], o["mask"]) for o in outs]

    def copies_per_picture(self):
        return 1 + len(self.cfg.get("train_rotations_deg", []) or []) + 4 * len(self.cfg.get("train_shifts_px", []) or [])

    def augment(self, canon):
        """Each training picture plus slightly turned copies (train_rotations_deg) and copies shifted by
        +/-s px left/right and up/down (train_shifts_px): the memory then tolerates the small turn and
        shift left over after straightening (audit M1: 2 px of shift used to reject good parts)."""
        out = []
        c = self.out_px / 2.0
        for img, m in canon:
            out.append(img)
            for a in self.cfg.get("train_rotations_deg", []) or []:
                M = cv2.getRotationMatrix2D((c, c), float(a), 1.0)
                out.append(cv2.warpAffine(img, M, (self.out_px, self.out_px), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REPLICATE))
            for sft in self.cfg.get("train_shifts_px", []) or []:
                for dx, dy in ((-sft, 0), (sft, 0), (0, -sft), (0, sft)):
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    out.append(cv2.warpAffine(img, M, (self.out_px, self.out_px), flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_REPLICATE))
        return out

    def _flat(self, imgs, masks):
        if not self.cfg.get("flatten_light", False):
            return list(imgs)
        return [flatten_light(i, m) for i, m in zip(imgs, masks)]

    def fit(self, canon, backbone):
        """canon: list of (canonical image, mask) from prepare()."""
        canon = list(zip(self._flat([c[0] for c in canon], [c[1] for c in canon]), [c[1] for c in canon]))
        imgs = self.augment(canon)
        # standardisation from the original pictures (the turned / shifted copies have the same statistics);
        # then every picture's features are embedded in chunks, so the raw 384-number features of all copies
        # are never held at once (audit M4: memory; 60 pictures x 7 copies used to need about 2 GB)
        f0 = backbone([c[0] for c in canon])                   # N0, C, h, w
        self.backbone_name = backbone.name
        _, c, h, w = f0.shape
        self.mu = f0.mean(axis=(0, 2, 3)).astype(np.float32)
        self.sd = (f0.std(axis=(0, 2, 3)) + 1e-6).astype(np.float32)
        self.proj = _proj_matrix(c, PROJ_DIM)
        del f0
        z = np.concatenate([self._embed(backbone(imgs[k:k + 32])) for k in range(0, len(imgs), 32)])  # N, h, w, D
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
                              backbone=self.backbone_name, backbone_sha256=getattr(backbone, "sha256", None),
                              trained=dt.datetime.now().isoformat(timespec="seconds"),
                              qc_version=__version__, order=self.order, crop_px=self.crop_px))
        self._lc = self._gc = None
        return self

    def check_backbone(self, backbone):
        """None when the backbone is the one the model was trained with; otherwise the reason (text).
        A model trained with one network and run with another gives meaningless scores (audit M4: the
        8-bit and the normal ResNet-18 file both used to be called onnx-resnet18)."""
        want = self.meta.get("backbone_sha256")
        have = getattr(backbone, "sha256", None)
        if self.backbone_name and backbone.name != self.backbone_name:
            return "model trained with features %s, but %s is loaded" % (self.backbone_name, backbone.name)
        if want and have and want != have:
            return ("model trained with a different network file (checksum %s..., loaded %s...): retrain, or "
                    "put the original backbone file back" % (want[:12], have[:12]))
        return None

    def _embed(self, f):
        z = (f - self.mu[None, :, None, None]) / self.sd[None, :, None, None]
        z = z.transpose(0, 2, 3, 1)                             # N, h, w, C
        if self.proj is not None:
            z = z @ self.proj
        return z.astype(np.float32)

    # -------------------------------------------------------------------------------------- scoring
    def correct(self, frame):
        """Divide out the learned light field (setting light_field); unchanged when there is none."""
        L = getattr(self, "light", None)
        if L is None or L.shape[:2] != frame.shape[:2]:
            return frame
        return np.clip(frame.astype(np.float32) / L[..., None], 0, 255).astype(np.uint8)

    def straighten(self, frame, loc):
        frame = self.correct(frame)
        span, step = float(self.cfg.get("fine_align_deg", 5.0)), float(self.cfg.get("fine_align_step_deg", 0.5))
        return st.straighten(frame, loc, self.ref_img, self.crop_px, self.out_px, self.order, span, step,
                             ref_mask=self.ref_mask)

    def _local_maps(self, z, exclude=None):
        """Same-spot nearest distances, (N, h, w). The memory is laid out cell-first (h+2r, w+2r, M, d), so
        for each of the (2r+1)^2 neighbour offsets one batched matrix product compares every cell with its
        M remembered features (audit M4: 83 ms instead of 720 ms per picture, same numbers).
        exclude: for picture i, a list of memory rows (M indices) to leave out (leave-one-out scores)."""
        n, h, w, d = z.shape
        B, BN = self._local_cache()
        r = self.window
        out = np.zeros((n, h, w), np.float32)
        for i in range(n):
            zi = np.ascontiguousarray(z[i])
            zn = (zi ** 2).sum(-1)
            best = np.full((h, w), np.inf, np.float32)
            for dy in range(2 * r + 1):
                for dx in range(2 * r + 1):
                    dot = np.matmul(B[dy:dy + h, dx:dx + w], zi[..., None])[..., 0]       # h, w, M
                    d2 = BN[dy:dy + h, dx:dx + w] - 2 * dot
                    if exclude is not None and exclude[i] is not None:
                        d2[..., exclude[i]] = np.inf
                    best = np.minimum(best, d2.min(-1))
            out[i] = np.sqrt(np.maximum(best + zn, 0))
        return out

    def anomaly_maps(self, canon_imgs, canon_masks, backbone, exclude=None):
        """Cell anomaly maps (N, h, w) for straightened pictures."""
        if canon_masks is not None:
            canon_imgs = self._flat(canon_imgs, canon_masks)
        f = backbone(canon_imgs)
        z = self._embed(f)
        n, h, w, d = z.shape
        out = np.zeros((n, h, w), np.float32)
        if self.bank_type == "local":
            out = self._local_maps(z, exclude)
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
        """The memory, cell-first: (h+2r, w+2r, M, d) float32 and its squared norms (h+2r, w+2r, M)."""
        if getattr(self, "_lc", None) is None:
            r = self.window
            b = self.bank.astype(np.float32).transpose(1, 2, 0, 3)                 # h, w, M, d
            pad = np.ascontiguousarray(np.pad(b, ((r, r), (r, r), (0, 0), (0, 0)), mode="edge"))
            self._lc = (pad, (pad ** 2).sum(-1))
        return self._lc

    def self_scores(self, canon, backbone):
        """Leave-one-out score of every training picture: compared with the memory WITHOUT its own rows
        (itself and its turned / shifted copies). A training picture that scores high looks unlike all
        the others: a damaged part, an empty spot or a mix-up that slipped into the good folder (audit M5).
        Local bank only (returns None for the global bank)."""
        if self.bank_type != "local" or len(canon) < 3:
            return None
        k = self.copies_per_picture()
        excl = [np.arange(i * k, (i + 1) * k) for i in range(len(canon))]
        maps = self.anomaly_maps([c[0] for c in canon], [c[1] for c in canon], backbone, exclude=excl)
        return maps.reshape(len(maps), -1).max(1)

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
        L = getattr(self, "light", None)
        np.savez_compressed(os.path.join(folder, "model.npz"), bank=self.bank, mu=self.mu, sd=self.sd,
                            proj=self.proj if self.proj is not None else np.zeros((0, 0), np.float32),
                            region=self.region, ref_img=self.ref_img, ref_mask=self.ref_mask,
                            light=L.astype(np.float16) if L is not None else np.zeros((0, 0), np.float16))
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
        m.light = z["light"].astype(np.float32) if "light" in z.files and z["light"].size else None
        m.order, m.crop_px, m.out_px = meta["order"], meta["crop_px"], meta["out_px"]
        m.bank_type, m.window, m.sigma = meta["bank"], meta["window"], meta["sigma"]
        m.backbone_name = meta["backbone"]
        m.meta = meta
        return m


LIGHT_SMOOTH = 6.0      # [chosen] px; smoothing of the learned light field (keeps it to light, not part detail)


def learn_light_field(frames, outs, ref, rmask):
    """Light that is fixed in the room, learned from the good training pictures: each picture is divided by
    the straightened average part turned back to where that part lay; averaged over many parts at many
    angles, what is left is the light at each spot of the picture (a slope, a reflection), not the parts'
    own markings. Returns a (H, W) gain field, 1.0 where no part ever lay."""
    h, w = frames[0].shape[:2]
    num = np.zeros((h, w), np.float32)
    den = np.zeros((h, w), np.float32)
    rg = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    inner = cv2.erode(rmask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32)
    for f, o in zip(frames, outs):
        Minv = cv2.invertAffineTransform(o["M"])
        pred = cv2.warpAffine(rg, Minv, (w, h), flags=cv2.INTER_LINEAR)
        m = cv2.warpAffine(inner, Minv, (w, h), flags=cv2.INTER_NEAREST) > 0.5
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ok = m & (pred > 20)
        num[ok] += g[ok] / pred[ok]
        den[ok] += 1
    have = den >= max(3, 0.1 * len(frames))
    ratio = np.where(have, num / np.maximum(den, 1), 0)
    wsum = cv2.GaussianBlur(have.astype(np.float32), (0, 0), LIGHT_SMOOTH)
    L = cv2.GaussianBlur(ratio, (0, 0), LIGHT_SMOOTH) / np.maximum(wsum, 1e-3)
    L = np.where(wsum > 0.3, L, 1.0).astype(np.float32)
    if have.any():
        L /= float(np.median(L[have]))
    return np.clip(L, 0.5, 2.0)


def part_contrast(empty_ref, frame, loc):
    """How clearly the part stands out from the backdrop: the median difference inside the part divided by
    the camera's own noise on the empty spot (empty_ref.noise_level). Dark parts on the black backdrop come
    out low (audit M6); below about 1 the part cannot be found at all."""
    d = empty_ref.difference(frame)
    inside = d[loc.filled > 0]
    return float(np.median(inside) / max(1e-6, empty_ref.noise_level)) if inside.size else 0.0
