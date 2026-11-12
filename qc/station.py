"""The inspection station: camera + empty reference + one model per shape and face + log + decisions.

Changelog
  0.1.0  2026-11-12  first version.
  0.2.0  2026-11-12  audit fixes: H1 a picture with a warning (part moving, backdrop or light changed, blurrier
                     than training, part at the frame edge, outline or face unlike the training pictures)
                     is never "accept": it answers unsure_goes_to with needs_attention and the reason, and
                     "recheck": true when asking again may help (switch warnings_make_unsure); H5 part
                     records restart when a finished part's id comes again, a repeated face replaces the
                     earlier answer, and faces without an id pair up; H6 "empty" refuses a picture with
                     something part-sized in view (force to override) and warns while the empty spot
                     differs from the one the model was trained with; M2 focus check; M4 a model never
                     runs with another backbone; M5 collect refuses empty, moving, edge and drifted
                     pictures; M9 mix-up checks; L2 empty_every_parts counts parts; L5 thresholds reload
                     when the file changes, log pictures are never overwritten.

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
from . import lens as lensmod
from . import model as modelmod
from . import features
from . import log as logmod


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


class CorrectedCamera:
    """Wraps a camera: every picture is undistorted with the checkerboard calibration (qc/lens.py)."""

    def __init__(self, cam, lens):
        self.cam, self.lens = cam, lens
        self.warnings = getattr(cam, "warnings", [])
        self.settings_report = getattr(cam, "settings_report", None)

    def __getattr__(self, k):
        return getattr(self.cam, k)

    def frame(self):
        return self.lens.undistort(self.cam.frame())

    def burst(self, n=8, settle_s=0.0):
        return [self.lens.undistort(f) for f in self.cam.burst(n, settle_s)]


def spread_limits(values):
    """Lower and upper limits from the training pictures themselves: the lowest and highest value seen,
    widened by the training pictures' own spread on that side (min - (median - min), max + (max - median)).
    No typed-in number: a feature that varies a lot in training gets wide limits."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if len(v) < 3:
        return None, None
    med = float(np.median(v))
    return float(v.min() - (med - v.min())), float(v.max() + (v.max() - med))


def touches_edge(mask):
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def picture_checks(m, frame, loc, s=None, sharp=None):
    """Checks whose limits come from the model's training pictures (meta.json). Returns a list of
    (kind, text, recheck): recheck = asking again after a moment may help."""
    out = []
    meta = m.meta or {}
    if touches_edge(loc.filled):
        out.append(("edge", "part touches the edge of the picture: put it back on the spot", True))
    lo, hi = spread_limits(meta.get("train_area", []))
    if lo is not None and not (lo <= loc.area <= hi):
        out.append(("area", "part size %.0f px is outside the training pictures' range (%.0f-%.0f): another shape, "
                    "two parts, or something on the part?" % (loc.area, lo, hi), False))
    if sharp is not None:
        lo, _ = spread_limits(meta.get("train_sharpness", []))
        if lo is not None and sharp < lo:
            out.append(("focus", "picture blurrier than every training picture (sharpness %.0f, limit %.0f): check the "
                        "focus, the part's height, and that the part had stopped" % (sharp, lo), True))
    if s is not None and s.get("corr") is not None:
        lo, _ = spread_limits(meta.get("align_corr", []))
        if lo is not None and s["corr"] < lo:
            out.append(("face", "this face does not match the trained %s/%s face (alignment %.2f, training lowest "
                        "%.2f): wrong face up, wrong shape, or a large defect" % (m.shape, m.face, s["corr"], lo), False))
    return out


class Station:
    def __init__(self, cfg, root=".", simulate=False, camera=None, log_name=None, mode="inspect"):
        self.cfg = cfg
        self.root = root
        self.simulate = simulate
        self.mode = mode
        self.camera = camera
        self.models = {}
        self.thresholds = {}
        self._th_mtime = {}
        self.train_empty = {}
        self._vs_train = {}
        self.parts = {}
        self._last_auto = None
        self._auto_n = 0
        self.since_empty = 0
        self.empty_time = None
        self.empty = None
        self.log = logmod.DecisionLog(os.path.join(root, cfg.get("log_dir", "runs")), log_name)
        self._backbone = None
        self.detector = None
        self.camera_warnings = []
        self.lens = lensmod.load_for(cfg, root)          # checkerboard calibration (None: not calibrated / off)
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
            if self.lens is not None:
                self.camera = CorrectedCamera(self.camera, self.lens)
            self.camera_warnings = list(getattr(self.camera, "warnings", []) or [])
            if self.camera_warnings:
                self.log.event("camera", warnings=self.camera_warnings,
                               settings=getattr(self.camera, "settings_report", None))
        return self.camera

    def _sim(self):
        """The simulated camera (inside the lens wrapper, if any), or None for a real one."""
        c = self.cam()
        c = getattr(c, "cam", c) if isinstance(c, CorrectedCamera) else c
        return c if isinstance(c, capture.SimCamera) else None

    def model(self, shape, face):
        key = (shape, face)
        folder = self.path("models", shape, face)
        if key not in self.models:
            if not os.path.exists(os.path.join(folder, "model.npz")):
                return None
            self.models[key] = modelmod.NormalModel.load(folder)
            ep = os.path.join(folder, "empty_at_training.npz")
            self.train_empty[key] = bgmod.EmptyReference.load(ep) if os.path.exists(ep) else None
        tp = os.path.join(folder, "threshold.json")
        mt = os.path.getmtime(tp) if os.path.exists(tp) else None
        if self._th_mtime.get(key, "unset") != mt:          # read again when calibrate-threshold rewrote it (L5)
            self.thresholds[key] = json.load(open(tp)) if mt is not None else None
            self._th_mtime[key] = mt
        return self.models[key]

    def vs_training(self, key):
        """Text when the current empty reference differs from the one the model was trained with, else ''."""
        ref = self.train_empty.get(key)
        if ref is None or self.empty is None:
            return ""
        ck = (key, self.empty_time)
        if ck not in self._vs_train:
            d = ref.drift(np.clip(self.empty.mean, 0, 255).astype(np.uint8))
            self._vs_train[ck] = ("the empty spot differs from when the %s/%s model was trained (%s): if this was not "
                                  "a deliberate change, put things back; otherwise retrain" % (
                                      key[0], key[1], "; ".join(d["reasons"]))) if d["drift"] else ""
        return self._vs_train[ck]

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
    def _training_refs(self):
        refs = []
        for p in glob.glob(self.path("models", "*", "*", "empty_at_training.npz")):
            try:
                refs.append(bgmod.EmptyReference.load(p))
            except Exception:       # noqa: BLE001
                pass
        return refs

    def take_empty(self, frames=None, force=False):
        """A new empty reference. Refused (the old one kept) when something part-sized is in view compared
        with the previous empty reference (or, the first time, with a model's training-time one): audit H6.
        force=True takes it anyway (after a deliberate change of backdrop, light or camera)."""
        n = int(self.cfg.get("empty_frames", 10))
        if frames is None:
            cam = self.cam()
            if self._sim() is not None:
                self._sim().clear()
                frames = cam.burst(n)
            else:
                # spread over a few seconds so slow flicker and noise are in the reference's limits
                import time
                frames = []
                for _ in range(n):
                    frames.append(cam.frame())
                    time.sleep(float(self.cfg.get("empty_spacing_s", 0.3)))
        mid = frames[len(frames) // 2]
        olds = [self.empty] if self.empty is not None else self._training_refs()[:1]
        if olds and not force and getattr(olds[0], "lens_id", None) == (self.lens.id if self.lens else None):
            seen = olds[0].part_mask(mid)
            if seen is not None:
                where = (round(seen.centre[0]), round(seen.centre[1]))
                txt = ("Not taken: something of %.0f px is in view at x=%d, y=%d compared with the previous empty picture. "
                       "Clear the spot (part, hand, arm, tools) and send \"empty\" again. If you changed the backdrop, "
                       "light or camera on purpose, send it with \"force\": true (and retrain)." % (seen.area, where[0], where[1]))
                self.log.event("empty_refused", area=seen.area, where=where)
                return dict(ok=False, refused=True, reason=txt, area=seen.area, where=where,
                            time=self.empty_time, empty_due=self.empty_due())
        new = bgmod.EmptyReference.from_frames(frames, float(self.cfg.get("mask_noise_quantile", 0.999)))
        new.lens_id = self.lens.id if self.lens is not None else None
        drift = None
        if self.empty is not None:
            drift = self.empty.drift(mid)
            if drift["drift"]:
                self.log.event("drift", source="empty", **drift)
        os.makedirs(self.path("models"), exist_ok=True)
        new.save(self.path("models", "empty.npz"))
        d = self.path("data", "empty")
        os.makedirs(d, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(os.path.join(d, "empty_%s.png" % stamp), mid)
        self.empty = new
        self.empty_time = now()
        self.since_empty = 0
        self.log.event("empty", frames=len(frames), noise_level=new.noise_level, drift=drift, forced=bool(force))
        out = dict(ok=True, frames=len(frames), noise_level=new.noise_level, time=self.empty_time,
                   drift=drift["drift"] if drift else False,
                   drift_reasons=drift["reasons"] if drift else [], forced=bool(force))
        warn = []
        if drift and drift["drift"]:
            warn.append("The empty spot looks different from the last empty picture (%s). Check the backdrop, light "
                        "and camera." % "; ".join(drift["reasons"]))
        for key in sorted(self.models):
            t = self.vs_training(key)
            if t:
                warn.append(t)
        if warn:
            out["warning"] = " ".join(warn)
        return out

    def health(self):
        models = sorted("%s/%s" % (s, f) for s, f in
                        [tuple(p.split(os.sep)[-3:-1]) for p in glob.glob(self.path("models", "*", "*", "model.npz"))])
        return dict(ok=self.empty is not None, version=__version__, mode=self.mode, models=models,
                    empty_time=self.empty_time, parts_since_empty=self.since_empty, empty_due=self.empty_due(),
                    simulate=self.simulate, backbone=getattr(self._backbone, "name", None),
                    camera_warnings=self.camera_warnings)

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

    # ------------------------------------------------------------------------------------- part records
    def _record(self, part_id, face):
        """The part this face belongs to (audit H5). A record is finished after faces_per_part faces or a
        reject by score; the same id after that starts a NEW part. A face sent again for an unfinished part
        replaces the earlier answer (a recheck). Without an id, a face joins the last unfinished part that
        has no answer for that face yet, else starts a new one with an id made up here."""
        n = int(self.cfg.get("faces_per_part", 2))

        def done(r):
            return len(r["faces"]) >= n or r["final"]
        if not part_id:
            r = self.parts.get(self._last_auto)
            if r is not None and not done(r) and face not in r["faces"]:
                return self._last_auto
            self._auto_n += 1
            part_id = "auto%04d" % self._auto_n
            self._last_auto = part_id
        r = self.parts.get(part_id)
        if r is None or done(r):
            self.parts[part_id] = dict(faces={}, final=False)
            self.since_empty += 1                    # empty_every_parts counts parts, not faces (L2)
            if len(self.parts) > 5000:               # keep memory bounded on long runs
                for k in list(self.parts)[:1000]:
                    self.parts.pop(k, None)
        else:
            r["faces"].pop(face, None)               # recheck of this face: the new answer replaces the old one
        return part_id

    # ------------------------------------------------------------------------------------- inspect
    def _base(self, part_id, shape, face):
        return dict(part_id=part_id, shape=shape, face=face, time=now(), version=__version__, warnings=[],
                    reason="", needs_attention=False, recheck=False, score=None, threshold=None, heatmap_path=None,
                    image_path=None)

    def inspect(self, part_id, shape, face):
        if self.empty is None:
            pid = self._record(part_id, face)
            return self._unsure(self._base(pid, shape, face), "no empty picture yet: send \"empty\" first")
        if self._sim() is not None and self._sim().part is None:
            # simulator without a fake arm putting parts down: invent a good part
            self._sim().next_part(shape, face)
        frame, loc, info = self.grab(shape, face)
        return self.inspect_frame(frame, shape, face, part_id, info, loc=loc)

    def _unsure(self, msg, reason, recheck=False):
        msg.update(decision=self.cfg.get("unsure_goes_to", "reject"), reason=reason, needs_attention=True,
                   recheck=bool(recheck))
        return self._finish(msg)

    def _finish(self, msg):
        pid = msg.get("part_id")
        rec = self.parts.setdefault(pid, dict(faces={}, final=False))
        rec["faces"][msg.get("face")] = msg["decision"]
        if msg["decision"] == "reject" and not msg.get("needs_attention"):
            rec["final"] = True                      # rejected by its score: this part is decided
        faces = rec["faces"]
        if "reject" in faces.values():
            msg["part_decision"] = "reject"
        elif len(faces) >= int(self.cfg.get("faces_per_part", 2)):
            msg["part_decision"] = "accept"
        else:
            msg["part_decision"] = "pending"
        msg["empty_due"] = self.empty_due()
        self.log.decision(msg)
        return msg

    def _face_tag(self, part_id, face):
        """face, or face-2, face-3 ... when this part's face was already logged (a recheck or a reused id):
        log pictures are never overwritten (L5)."""
        tag, k = face, 1
        while os.path.exists(self.log.image_path(part_id, tag, "heat", "png")) or \
                os.path.exists(self.log.image_path(part_id, tag, "frame", "jpg")):
            k += 1
            tag = "%s-%d" % (face, k)
        return tag

    def inspect_frame(self, frame, shape, face, part_id=None, info=None, loc=None):
        part_id = self._record(part_id, face)
        msg = self._base(part_id, shape, face)
        tag = self._face_tag(part_id, face)
        info = info or {}
        msg["sharpness"] = info.get("sharpness")
        if self.cfg.get("save_raw_frames", True):
            p = self.log.image_path(part_id, tag, "frame", "jpg")
            cv2.imwrite(p, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            msg["image_path"] = p
        m = self.model(shape, face)
        if m is None:
            return self._unsure(msg, "no model for %s/%s: run  python -m qc train --shape %s --face %s" % (
                shape, face, shape, face))
        bad_bb = m.check_backbone(self.backbone)
        if bad_bb:
            return self._unsure(msg, bad_bb)
        if self.empty is None:
            return self._unsure(msg, "no empty picture yet")
        lid = self.lens.id if self.lens is not None else None
        if getattr(self.empty, "lens_id", None) != lid:
            return self._unsure(msg, "the camera calibration changed since the empty picture: send \"empty\" again")
        if m.meta.get("lens_id", None) != lid and (m.meta.get("lens_id") or lid):
            return self._unsure(msg, "the %s/%s model was trained with another camera calibration (or none): retrain "
                                "it (python -m qc train --shape %s --face %s)" % (shape, face, shape, face))
        if loc is None:
            loc = self.locate(frame)
        if loc is None:
            return self._unsure(msg, "no part found at the spot (if the part is dark, the black backdrop may hide it)",
                                recheck=True)
        attn = []                                    # (text, recheck): reasons this picture can't be trusted
        if info.get("still") is False:
            attn.append(("part was still moving (%.1f, limit %.1f)" % (info["motion"], info["motion_limit"]), True))
        drift = self.empty.drift(frame, exclude=loc.filled)
        if drift["drift"]:
            attn.append(("background drift: " + "; ".join(drift["reasons"]), False))
            self.log.event("drift", source="part", part_id=part_id, **drift)
        vt = self.vs_training((shape, face))
        if vt:
            msg["warnings"].append(vt)
        sharp = capture.sharpness(frame, loc.filled)
        msg["sharpness"] = sharp
        s = m.straighten(frame, loc)
        for kind, text, rc in picture_checks(m, frame, loc, s, sharp if self.cfg.get("focus_check", True) else None):
            attn.append((text, rc))
        scores, maps = m.score_canonical([s["image"]], [s["mask"]], self.backbone)
        score = float(scores[0])
        th = self.thresholds.get((shape, face))
        thr = float(th["threshold"]) if th else None
        msg.update(score=score, threshold=thr, angle_deg=s["angle_deg"],
                   model="%s/%s %s" % (shape, face, m.meta.get("trained", "")))
        heat = m.heatmap(maps[0])
        hp = self.log.image_path(part_id, tag, "heat", "png")
        cv2.imwrite(hp, np.hstack([s["image"], overlay(s["image"], heat, thr)]))
        msg["heatmap_path"] = hp
        # where in the full picture: the hottest spot mapped back through the straightening
        yx = np.unravel_index(int(np.argmax(heat)), heat.shape)
        Minv = cv2.invertAffineTransform(s["M"])
        fx, fy = Minv @ np.array([yx[1], yx[0], 1.0])
        msg["hot_spot_px"] = [round(float(fx), 1), round(float(fy), 1)]
        mmpp = None
        if self.cfg.get("px_to_mm") and self.cfg.get("mm_per_px"):
            mmpp = float(self.cfg["mm_per_px"])                      # calibrate-mm (kept, switch px_to_mm)
        elif self.lens is not None and self.lens.mm_per_px:
            mmpp = float(self.lens.mm_per_px)                        # checkerboard card at the spot
        if mmpp:
            from . import calib
            msg["mm_per_px"] = mmpp
            msg["hot_spot_size_mm"] = calib.hot_spot_size_mm(heat, thr, m.crop_px / float(m.out_px), mmpp)
        wp = self.log.image_path(part_id, tag, "where", "jpg")
        big = cv2.warpAffine(heat, Minv, (frame.shape[1], frame.shape[0]))
        cv2.imwrite(wp, overlay(frame, big, thr, 0.35))
        if thr is None:
            return self._unsure(msg, "no threshold for %s/%s yet: run calibrate-threshold" % (shape, face))
        msg["warnings"] += [t for t, _ in attn]
        if attn:
            msg.update(needs_attention=True, reason="; ".join(t for t, _ in attn),
                       recheck=all(rc for _, rc in attn))
        if score > thr:
            msg["decision"] = "reject"
        elif attn and self.cfg.get("warnings_make_unsure", True):
            msg["decision"] = self.cfg.get("unsure_goes_to", "reject")      # never "accept" with a warning (H1)
        else:
            msg["decision"] = "accept"
        return self._finish(msg)

    # ------------------------------------------------------------------------------------- collect
    def collect(self, part_id, shape, face, frame=None, loc=None, info=None):
        """Save a good part's picture for training (collect-good). The arm gets "accept" back. Refused (not
        saved, ok false, with the reason) when no part is found, the part touches the frame edge, it was
        still moving, or the backdrop changed (audit M5)."""
        if frame is None:
            if self._sim() is not None and self._sim().part is None:
                self._sim().next_part(shape, face)
            frame, loc, info = self.grab(shape, face)
        info = info or {}

        def refuse(reason):
            self.log.event("collect_refused", part_id=part_id, shape=shape, face=face, reason=reason)
            return dict(ok=False, decision="accept", part_id=part_id, shape=shape, face=face, needs_attention=True,
                        reason=reason + "; picture not kept")
        if self.empty is None:
            return refuse("no empty picture yet: send \"empty\" first")
        if loc is None:
            loc = self.locate(frame)
        if loc is None:
            return refuse("no part found at the spot")
        if info.get("still") is False:
            return refuse("part was moving")
        if touches_edge(loc.filled):
            return refuse("part touches the edge of the picture")
        drift = self.empty.drift(frame, exclude=loc.filled)
        if drift["drift"]:
            return refuse("the backdrop or light changed (%s)" % "; ".join(drift["reasons"]))
        d = self.path("data", "good", shape, face)
        os.makedirs(d, exist_ok=True)
        name = "%s_%s.png" % (dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f"), logmod.safe_name(part_id or "p"))
        cv2.imwrite(os.path.join(d, name), frame)
        n = len(glob.glob(os.path.join(d, "*.png")))
        self.log.event("collect", part_id=part_id, shape=shape, face=face, file=name, count=n)
        return dict(ok=True, decision="accept", part_id=part_id, shape=shape, face=face, collected=n,
                    file=os.path.join(d, name))
