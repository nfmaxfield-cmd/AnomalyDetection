"""The inspection station: camera + empty reference + one model per shape and face + log + decisions.

Changelog
  0.1.0  2026-11-12  first version.

Folders (relative to the working folder, `root`):
  data/empty/            empty-spot pictures            data/good/<shape>/<face>/   good parts
  data/damaged/<shape>/<face>/   deliberately damaged parts (for the threshold)
  models/empty.npz       the empty reference             models/<shape>/<face>/      model + threshold
  runs/<time>/           the decision log (log.py)

    st = Station(cfg, root=".", simulate=True)
    st.take_empty()                                        # empty reference (and drift check vs the old one)
    msg = st.inspect("p1", "square", "top")                # grab burst, judge, log, message for the arm
    msg = st.inspect_frame(frame, "square", "top", "p1")   # judge a picture you already have
"""
import datetime as dt
import glob
import json
import os

import cv2
import numpy as np

from . import __version__
from . import background as bgmod
from . import capture
from . import features
from . import log as logmod
from .model import NormalModel


def now():
    return dt.datetime.now().isoformat(timespec="milliseconds")


def heat_colour(heat, threshold):
    """Heat map as colours: yellow at half the threshold, red at the threshold, dark red above."""
    t = threshold if threshold else float(heat.max()) + 1e-6
    v = np.clip((heat / t - 0.5) / 0.75, 0, 1)                 # 0.5 x threshold -> 0, 1.25 x threshold -> 1
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_AUTUMN)[:, 0]
    return lut[(255 - v * 255).astype(np.uint8)], v


def overlay(img, heat, threshold, alpha=0.7):
    """The picture unchanged where the part looks like the good ones (below half the threshold); colour
    fades in towards the threshold and is strongest above it."""
    col, v = heat_colour(heat, threshold)
    a = (alpha * v)[..., None]
    return (img.astype(np.float32) * (1 - a) + col.astype(np.float32) * a).astype(np.uint8)


class Station:
    def __init__(self, cfg, root=".", simulate=False, camera=None, log_name=None, mode="inspect"):
        self.cfg = cfg
        self.root = root
        self.simulate = simulate
        self.mode = mode
        self.camera = camera
        self.models = {}
        self.thresholds = {}
        self.parts = {}
        self.since_empty = 0
        self.empty_time = None
        self.empty = None
        self.log = logmod.DecisionLog(os.path.join(root, cfg.get("log_dir", "runs")), log_name)
        self._backbone = None
        self.detector = None
        p = self.path("models", "empty.npz")
        if os.path.exists(p):
            self.empty = bgmod.EmptyReference.load(p)
            self.empty_time = dt.datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds")

    # ------------------------------------------------------------------------------------- helpers
    def path(self, *p):
        return os.path.join(self.root, *p)

    @property
    def backbone(self):
        if self._backbone is None:
            self._backbone = features.make(self.cfg, self.root)
            w = getattr(self._backbone, "warning", None)
            if w:
                self.log.event("warning", text=w)
        return self._backbone

    def cam(self):
        if self.camera is None:
            self.camera = capture.open_camera(self.cfg, self.simulate)
        return self.camera

    def model(self, shape, face):
        key = (shape, face)
        if key not in self.models:
            folder = self.path("models", shape, face)
            if not os.path.exists(os.path.join(folder, "model.npz")):
                return None
            self.models[key] = NormalModel.load(folder)
            tp = os.path.join(folder, "threshold.json")
            self.thresholds[key] = json.load(open(tp)) if os.path.exists(tp) else None
        return self.models[key]

    def locate(self, frame):
        if self.cfg.get("locate_with_detector") and self.cfg.get("detector_weights"):
            from . import locate_detector
            if self.detector is None:
                self.detector = locate_detector.DetectorLocator(self.cfg["detector_weights"])
            near = self.detector.centre(frame)
            if near is not None:
                return self.empty.part_mask(frame, near=near, search_radius=self.detector.search_radius)
        return self.empty.part_mask(frame)

    # ------------------------------------------------------------------------------------- empty
    def take_empty(self, frames=None):
        n = int(self.cfg.get("empty_frames", 10))
        if frames is None:
            cam = self.cam()
            if isinstance(cam, capture.SimCamera):
                cam.clear()
                frames = cam.burst(n)
            else:
                # spread over a few seconds so slow flicker and noise are in the reference's limits
                import time
                frames = []
                for _ in range(n):
                    frames.append(cam.frame())
                    time.sleep(float(self.cfg.get("empty_spacing_s", 0.3)))
        new = bgmod.EmptyReference.from_frames(frames, float(self.cfg.get("mask_noise_quantile", 0.999)))
        drift = None
        if self.empty is not None:
            drift = self.empty.drift(frames[len(frames) // 2])
            if drift["drift"]:
                self.log.event("drift", source="empty", **drift)
        os.makedirs(self.path("models"), exist_ok=True)
        new.save(self.path("models", "empty.npz"))
        d = self.path("data", "empty")
        os.makedirs(d, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(os.path.join(d, "empty_%s.png" % stamp), frames[len(frames) // 2])
        self.empty = new
        self.empty_time = now()
        self.since_empty = 0
        self.log.event("empty", frames=len(frames), noise_level=new.noise_level, drift=drift)
        out = dict(ok=True, frames=len(frames), noise_level=new.noise_level, time=self.empty_time,
                   drift=drift["drift"] if drift else False,
                   drift_reasons=drift["reasons"] if drift else [])
        if drift and drift["drift"]:
            out["warning"] = ("The empty spot looks different from the last empty picture (%s). Check the "
                              "backdrop, light and camera; models trained before the change may need "
                              "retraining." % "; ".join(drift["reasons"]))
        return out

    def health(self):
        models = sorted("%s/%s" % (s, f) for s, f in
                        [tuple(p.split(os.sep)[-3:-1]) for p in glob.glob(self.path("models", "*", "*", "model.npz"))])
        return dict(ok=self.empty is not None, version=__version__, mode=self.mode, models=models,
                    empty_time=self.empty_time, parts_since_empty=self.since_empty, empty_due=self.empty_due(),
                    simulate=self.simulate, backbone=getattr(self._backbone, "name", None))

    def empty_due(self):
        every = int(self.cfg.get("empty_every_parts", 0) or 0)
        return self.empty is None or (every > 0 and self.since_empty >= every)

    # ------------------------------------------------------------------------------------- capture
    def grab(self, shape=None, face=None):
        """Burst after "paused": the sharpest still frame. Returns (frame, loc, info)."""
        cam = self.cam()
        frames = cam.burst(int(self.cfg.get("burst_frames", 8)), float(self.cfg.get("burst_settle_s", 0)))
        loc0 = self.locate(frames[-1])
        mask = loc0.filled if loc0 is not None else None
        i, sharp = capture.pick_sharpest(frames, mask)
        info = dict(sharpness=sharp[i], burst=len(frames), picked=i)
        if self.cfg.get("still_check", True) and mask is not None:
            ok, mv, lim = capture.is_still(frames, i, self.empty.still_noise, mask)
            info.update(still=ok, motion=mv, motion_limit=lim)
        frame = frames[i]
        return frame, self.locate(frame), info

    # ------------------------------------------------------------------------------------- inspect
    def _base(self, part_id, shape, face):
        return dict(part_id=part_id, shape=shape, face=face, time=now(), version=__version__, warnings=[],
                    reason="", needs_attention=False, score=None, threshold=None, heatmap_path=None,
                    image_path=None)

    def inspect(self, part_id, shape, face):
        if self.empty is None:
            return self._unsure(self._base(part_id, shape, face), "no empty picture yet: send \"empty\" first")
        if isinstance(self.cam(), capture.SimCamera) and self.cam().part is None:
            # simulator without a fake arm putting parts down: invent a good part
            self.cam().next_part(shape, face)
        frame, loc, info = self.grab(shape, face)
        return self.inspect_frame(frame, shape, face, part_id, info, loc=loc)

    def _unsure(self, msg, reason):
        msg.update(decision=self.cfg.get("unsure_goes_to", "reject"), reason=reason, needs_attention=True)
        return self._finish(msg)

    def _finish(self, msg):
        pid = msg.get("part_id")
        faces = self.parts.setdefault(pid, {})
        faces[msg.get("face")] = msg["decision"]
        if "reject" in faces.values():
            msg["part_decision"] = "reject"
        elif len(faces) >= 2:
            msg["part_decision"] = "accept"
        else:
            msg["part_decision"] = "pending"
        self.since_empty += 1
        msg["empty_due"] = self.empty_due()
        self.log.decision(msg)
        return msg

    def inspect_frame(self, frame, shape, face, part_id=None, info=None, loc=None):
        part_id = part_id or "part%d" % (len(self.parts) + 1)
        msg = self._base(part_id, shape, face)
        info = info or {}
        msg["sharpness"] = info.get("sharpness")
        if self.cfg.get("save_raw_frames", True):
            p = self.log.image_path(part_id, face, "frame", "jpg")
            cv2.imwrite(p, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            msg["image_path"] = p
        m = self.model(shape, face)
        if m is None:
            return self._unsure(msg, "no model for %s/%s: run  python -m qc train --shape %s --face %s" % (
                shape, face, shape, face))
        if self.empty is None:
            return self._unsure(msg, "no empty picture yet")
        if loc is None:
            loc = self.locate(frame)
        if loc is None:
            return self._unsure(msg, "no part found at the spot")
        if info.get("still") is False:
            msg["warnings"].append("part was still moving (%.1f, limit %.1f)" % (info["motion"], info["motion_limit"]))
        drift = self.empty.drift(frame, exclude=loc.filled)
        if drift["drift"]:
            msg["warnings"].append("background drift: " + "; ".join(drift["reasons"]))
            self.log.event("drift", source="part", part_id=part_id, **drift)
        s = m.straighten(frame, loc)
        scores, maps = m.score_canonical([s["image"]], [s["mask"]], self.backbone)
        score = float(scores[0])
        th = self.thresholds.get((shape, face))
        thr = float(th["threshold"]) if th else None
        msg.update(score=score, threshold=thr, angle_deg=s["angle_deg"],
                   model="%s/%s %s" % (shape, face, m.meta.get("trained", "")))
        heat = m.heatmap(maps[0])
        hp = self.log.image_path(part_id, face, "heat", "png")
        cv2.imwrite(hp, np.hstack([s["image"], overlay(s["image"], heat, thr)]))
        msg["heatmap_path"] = hp
        # where in the full picture: the hottest spot mapped back through the straightening
        yx = np.unravel_index(int(np.argmax(heat)), heat.shape)
        Minv = cv2.invertAffineTransform(s["M"])
        fx, fy = Minv @ np.array([yx[1], yx[0], 1.0])
        msg["hot_spot_px"] = [round(float(fx), 1), round(float(fy), 1)]
        if self.cfg.get("px_to_mm") and self.cfg.get("mm_per_px"):
            from . import calib
            msg["hot_spot_size_mm"] = calib.hot_spot_size_mm(heat, thr, m.crop_px / float(m.out_px),
                                                             float(self.cfg["mm_per_px"]))
        wp = self.log.image_path(part_id, face, "where", "jpg")
        big = cv2.warpAffine(heat, Minv, (frame.shape[1], frame.shape[0]))
        cv2.imwrite(wp, overlay(frame, big, thr, 0.35))
        if thr is None:
            return self._unsure(msg, "no threshold for %s/%s yet: run calibrate-threshold" % (shape, face))
        msg["decision"] = "reject" if score > thr else "accept"
        if drift["drift"] or info.get("still") is False:
            msg["needs_attention"] = True
        return self._finish(msg)

    # ------------------------------------------------------------------------------------- collect
    def collect(self, part_id, shape, face, frame=None):
        """Save a good part's picture for training (collect-good). The arm gets "accept" back."""
        if frame is None:
            if isinstance(self.cam(), capture.SimCamera) and self.cam().part is None:
                self.cam().next_part(shape, face)
            frame, loc, info = self.grab(shape, face)
            if info.get("still") is False:
                return dict(ok=False, decision="accept", part_id=part_id, needs_attention=True,
                            reason="part was moving; picture not kept")
        d = self.path("data", "good", shape, face)
        os.makedirs(d, exist_ok=True)
        name = "%s_%s.png" % (dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f"), logmod.safe_name(part_id or "p"))
        cv2.imwrite(os.path.join(d, name), frame)
        n = len(glob.glob(os.path.join(d, "*.png")))
        self.log.event("collect", part_id=part_id, shape=shape, face=face, file=name, count=n)
        return dict(ok=True, decision="accept", part_id=part_id, shape=shape, face=face, collected=n,
                    file=os.path.join(d, name))
