"""Getting pictures: the webcam (settings locked), a simulated camera, and picking the sharpest frame.

Changelog
  0.1.0  2026-11-12  first version.
  0.2.0  2026-11-12  audit fixes: H4 the webcam is opened with a backend chosen per system (DirectShow on
                     Windows, V4L2 on Linux / Raspberry Pi), a reader thread always holds the NEWEST frame so a
                     burst never contains frames that waited in the camera's buffer (old direct reads kept
                     behind the switch camera_reader_thread, with a buffer drain), every locked setting is
                     read back and a warning names any that did not hold, gain is locked too; L4 a failed
                     read is retried and the camera reopened; the stillness check also compares frames
                     2+ apart, so a slow slide (1 px per frame) is no longer called still.

When the arm says "paused", a short burst of frames is grabbed and the sharpest one is kept (variance of
the Laplacian, measured on the part only, so the black backdrop never decides). The kept frame must also
be still: it is compared with its neighbours in the burst (next to it, and 2 or more frames away), and if
the part moved more than the camera's own noise allows, the result says so (the arm should wait a moment
and ask again).

    cam = open_camera(cfg)                  # webcam with locked exposure / white balance / focus / gain
    cam.settings_report, cam.warnings       # each setting: asked, read back, held?
    cam = SimCamera()                       # synthetic pictures (qc/sim.py); cam.next_part(...) sets the part
    frames = cam.burst(8)
    i, sharp = pick_sharpest(frames, mask)  # index and the sharpness of every frame
"""
import sys
import threading
import time

import cv2
import numpy as np

from . import sim


def sharpness(frame, mask=None):
    """Variance of the Laplacian (higher = sharper), on the part's bounding box when a mask is given."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if mask is not None and mask.any():
        ys, xs = np.nonzero(mask)
        y0, y1, x0, x1 = max(0, ys.min() - 4), ys.max() + 5, max(0, xs.min() - 4), xs.max() + 5     # [chosen] 4 px margin; not DOE
        g = g[y0:y1, x0:x1]
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def pick_sharpest(frames, mask=None):
    s = [sharpness(f, mask) for f in frames]
    return int(np.argmax(s)), s


def _change(a, b, mask=None):
    """Mean absolute change of the part area between two blurred frames, overall brightness flicker of
    the light divided out first (it is not movement)."""
    sel = mask > 0 if mask is not None and mask.any() else slice(None)
    b = b * (a[sel].mean() / max(1e-6, b[sel].mean()))
    d = np.abs(a - b).max(axis=2)
    if mask is not None and mask.any():
        d = d[mask > 0]
    return float(d.mean())


SLIDE_MIN_BURST = 6     # [chosen] the slide check needs a burst this long (shorter: only the jolt check); not DOE
STILL_FACTOR = 3.0      # [chosen] a jolt: change to both neighbours above 3x the noise unit; DOE-able
SLIDE_FACTOR = 1.5      # [chosen] a slide: change to every frame 2+ away above 1.5x the noise unit (a steady
                        #          slide gives 2x or more, a still part about 1x); DOE-able


def _blurred(frames):
    return [cv2.GaussianBlur(f.astype(np.float32), (0, 0), 1.0) for f in frames]


def motion_parts(frames, i, mask=None):
    """(near, far, floor) around frame i: near = change to the more similar neighbour; far = change to the
    most similar frame at least 2 away (bursts of SLIDE_MIN_BURST or more, else = near); floor = the smallest
    change between two neighbouring frames of the burst (the part's own noise in this burst: a bright part
    is noisier than the dark empty backdrop)."""
    if len(frames) < 2:
        return 0.0, 0.0, 0.0
    g = _blurred(frames)
    near = min(_change(g[i], g[j], mask) for j in (i - 1, i + 1) if 0 <= j < len(frames))
    far = near
    if len(frames) >= SLIDE_MIN_BURST:
        far = min(_change(g[i], g[j], mask) for j in range(len(frames)) if abs(j - i) >= 2)
    floor = min(_change(g[k], g[k + 1], mask) for k in range(len(frames) - 1))
    return near, far, floor


def motion(frames, i, mask=None):
    """The first build's measure (kept): change to the more similar neighbour."""
    return motion_parts(frames, i, mask)[0]


def is_still(frames, i, still_noise, mask=None):
    """Still unless (a) a jolt: frame i differs from both neighbours by more than STILL_FACTOR x the noise
    unit, or (b) a slide: it differs from every frame 2 or more away by more than SLIDE_FACTOR x that unit
    (audit H1: a slide of 1 px per frame used to pass). The unit is the larger of the empty scene's
    frame-to-frame change and this burst's own floor (a bright part is noisier than the dark backdrop). Returns (still, measure, limit) for the check nearest to failing."""
    near, far, floor = motion_parts(frames, i, mask)
    # the burst's own floor is only trusted when the burst is long enough for the slide check: in a shorter
    # burst a steady slide would set its own floor
    unit = max(float(still_noise), floor) if len(frames) >= SLIDE_MIN_BURST else float(still_noise)
    lim_near = STILL_FACTOR * unit
    lim_far = SLIDE_FACTOR * unit
    if near / lim_near >= far / lim_far:
        return near <= lim_near and far <= lim_far, near, lim_near
    return near <= lim_near and far <= lim_far, far, lim_far


# ------------------------------------------------------------------------------------------- webcam
BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "v4l2": cv2.CAP_V4L2,
            "avfoundation": cv2.CAP_AVFOUNDATION, "any": cv2.CAP_ANY}
# what each backend takes for "manual exposure", and its exposure scale (for the camera-setup slider)
EXPOSURE = {
    "dshow": dict(manual=(0.25, 1), lo=-13, hi=0, unit="2^value seconds"),
    "msmf": dict(manual=(0.25, 1), lo=-13, hi=0, unit="2^value seconds"),
    "v4l2": dict(manual=(1, 0.25), lo=1, hi=2000, unit="0.1 ms"),
    "avfoundation": dict(manual=(), lo=None, hi=None, unit=None),
    "any": dict(manual=(1, 0.25), lo=-13, hi=2000, unit="driver units"),
}
READ_TOL = dict(exposure=0.5, gain=1.0, wb=50.0, focus=1.0)     # [chosen] read-back rounding tolerance; not DOE


def pick_backend(name="auto", platform=None):
    """The OpenCV capture backend to use: DirectShow on Windows (Media Foundation often ignores manual
    exposure and focus), V4L2 on Linux and the Raspberry Pi, AVFoundation on a Mac (which cannot set
    exposure or focus at all: a warning says so)."""
    if name and name != "auto":
        if name not in BACKENDS:
            raise ValueError("camera_backend %r: use auto, %s" % (name, ", ".join(BACKENDS)))
        return name
    platform = platform or sys.platform
    if platform.startswith("win"):
        return "dshow"
    if platform.startswith("linux"):
        return "v4l2"
    if platform == "darwin":
        return "avfoundation"
    return "any"


class OpenCVCamera:
    """A USB webcam through OpenCV with automatic exposure, gain, white balance and focus turned off, every
    setting read back after it is set, and (by default) a reader thread that always holds the newest frame.
    Which numbers a driver accepts differs between webcams; `python -m qc camera-check` tests them."""

    def __init__(self, cfg, capture_factory=None):
        self.cfg = cfg
        self.backend = pick_backend(cfg.get("camera_backend", "auto"))
        self._factory = capture_factory or cv2.VideoCapture
        self.warnings = []
        self.settings_report = []
        self.locked = {}
        self._open()
        self._lock = threading.Condition()
        self._latest = None          # (frame, seq, time)
        self._seq = 0
        self._stop = threading.Event()
        self._error = None
        self._thread = None
        if cfg.get("camera_reader_thread", True):
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()
            self._wait_new(0, timeout=5.0)

    # --------------------------------------------------------------------------- open and lock
    def _open(self):
        idx = int(self.cfg.get("camera_index", 0))
        self.cap = self._factory(idx, BACKENDS[self.backend])
        if not self.cap.isOpened():
            raise RuntimeError("Camera %s did not open (backend %s). Is it plugged in, and closed in other "
                               "programs? (setting camera_backend can choose another backend)" % (idx, self.backend))
        c = self.cap
        c.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.cfg.get("frame_width", 640)))
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.cfg.get("frame_height", 480)))
        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # not every backend honours it: the reader thread is the real fix
        self.size = (int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.settings_report = []
        self.warnings = []
        want = (int(self.cfg.get("frame_width", 640)), int(self.cfg.get("frame_height", 480)))
        if self.size != want and all(self.size):
            self.warnings.append("asked for %dx%d pictures, the camera gives %dx%d" % (want + self.size))
        if self.backend == "avfoundation":
            self.warnings.append("macOS: OpenCV cannot turn off automatic exposure or focus here. Use the station "
                                 "on Windows or Linux (SETUP-GUIDE step 1).")
        if self.cfg.get("lock_camera_settings", True):
            self._lock_settings()
        for _ in range(5):              # let the first frames (often dark) pass
            c.read()

    def _setting(self, name, prop, value, tol):
        """Set prop to value (None: keep the camera's current value, written back so it holds), read it back."""
        c = self.cap
        before = c.get(prop)
        asked = before if value is None else float(value)
        c.set(prop, asked)
        got = c.get(prop)
        ok = abs(got - asked) <= tol
        self.settings_report.append(dict(name=name, asked=asked, read_back=got, held=bool(ok),
                                         kept_current=value is None))
        if not ok:
            self.warnings.append("%s: asked %g, the camera reports %g (the setting did not hold)" % (name, asked, got))
        self.locked[name] = got
        return ok

    def _lock_settings(self):
        c = self.cap
        ex = EXPOSURE[self.backend]
        held = False
        for v in ex["manual"]:          # "manual exposure" is a different number per backend: try, read back
            c.set(cv2.CAP_PROP_AUTO_EXPOSURE, v)
            got = c.get(cv2.CAP_PROP_AUTO_EXPOSURE)
            if abs(got - v) < 1e-3:
                held = True
                break
        self.settings_report.append(dict(name="auto exposure off", asked="manual", read_back=c.get(cv2.CAP_PROP_AUTO_EXPOSURE),
                                         held=held, kept_current=False))
        if not held and ex["manual"]:
            self.warnings.append("auto exposure: could not switch it off (the camera reports %g); pictures may "
                                 "change brightness by themselves" % c.get(cv2.CAP_PROP_AUTO_EXPOSURE))
        self._setting("exposure", cv2.CAP_PROP_EXPOSURE, self.cfg.get("exposure"), READ_TOL["exposure"])
        self._setting("gain", cv2.CAP_PROP_GAIN, self.cfg.get("gain"), READ_TOL["gain"])
        c.set(cv2.CAP_PROP_AUTO_WB, 0)
        awb = c.get(cv2.CAP_PROP_AUTO_WB)
        self.settings_report.append(dict(name="auto white balance off", asked=0, read_back=awb, held=awb == 0,
                                         kept_current=False))
        if awb != 0:
            self.warnings.append("auto white balance: could not switch it off (the camera reports %g)" % awb)
        self._setting("white balance", cv2.CAP_PROP_WB_TEMPERATURE, self.cfg.get("white_balance_k"), READ_TOL["wb"])
        c.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        af = c.get(cv2.CAP_PROP_AUTOFOCUS)
        self.settings_report.append(dict(name="autofocus off", asked=0, read_back=af, held=af == 0, kept_current=False))
        if af != 0:
            self.warnings.append("autofocus: could not switch it off (the camera reports %g)" % af)
        self._setting("focus", cv2.CAP_PROP_FOCUS, self.cfg.get("focus"), READ_TOL["focus"])
        self.locked.update(auto_exposure=c.get(cv2.CAP_PROP_AUTO_EXPOSURE), auto_wb=awb, autofocus=af)

    def reopen(self):
        """Release and open the camera again with the same settings (after it stopped giving pictures)."""
        try:
            self.cap.release()
        except Exception:      # noqa: BLE001
            pass
        self._open()

    # --------------------------------------------------------------------------- reading
    def _read(self):
        """One frame straight from OpenCV; retried, then the camera is reopened once (audit L4)."""
        for attempt in range(6):
            ok, f = self.cap.read()
            if ok and f is not None:
                return f
            if attempt == 2:
                self.reopen()
            time.sleep(0.05)
        raise RuntimeError("The camera stopped giving pictures (tried again and reopened it). Check the cable.")

    def _reader(self):
        while not self._stop.is_set():
            try:
                f = self._read()
            except RuntimeError as e:
                with self._lock:
                    self._error = e
                    self._lock.notify_all()
                time.sleep(0.5)
                continue
            with self._lock:
                self._seq += 1
                self._error = None
                self._latest = (f, self._seq, time.time())
                self._lock.notify_all()

    def _wait_new(self, after_seq, timeout=3.0):
        """The first frame whose number is above after_seq (taken after the request)."""
        end = time.time() + timeout
        with self._lock:
            while self._latest is None or self._latest[1] <= after_seq:
                left = end - time.time()
                if left <= 0:
                    raise RuntimeError(str(getattr(self, "_error", None) or "The camera stopped giving pictures."))
                self._lock.wait(left)
            return self._latest

    def drain(self):
        """Without the reader thread: throw away frames that waited in the camera's buffer. Buffered frames
        come back at once; a live one takes about one frame period, so reading stops at the first slow one."""
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        period = 1.0 / max(1.0, min(fps, 120.0))
        for _ in range(30):
            t0 = time.time()
            self.cap.grab()
            if time.time() - t0 > 0.5 * period:
                break

    def frame(self):
        if self._thread is None:
            return self._read()
        with self._lock:
            seq = self._latest[1] if self._latest else 0
        return self._wait_new(seq)[0]

    def burst(self, n=8, settle_s=0.0):
        """n NEW frames, all taken after this call (none from before the arm's "paused")."""
        if settle_s:
            time.sleep(settle_s)
        if self._thread is None:
            self.drain()
            return [self._read() for _ in range(n)]
        with self._lock:
            seq = self._latest[1] if self._latest else 0
        seq += 1                         # the frame being exposed during the call may have started before it
        out = []
        while len(out) < n:
            f, seq, _ = self._wait_new(seq)
            out.append(f)
        return out

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)
        self.cap.release()


class SimCamera:
    """Synthetic camera for --simulate and tests: shows the empty spot until next_part() puts a part
    there, and goes back to empty after clear(). scene: "hard" (default) or "easy" (qc/sim.py)."""

    def __init__(self, seed=0, drift=None, scene=None):
        self.seed = seed
        self.count = 0
        self.part = None
        self.drift = drift
        self.scene = scene

    def next_part(self, shape, face, defect=None, seed=None, angle=None, severity=1.0):
        self.part = dict(shape=shape, face=face, defect=defect, seed=self.seed * 100003 + self.count if seed is None else seed,
                         angle=angle, severity=severity)
        self.count += 1

    def clear(self):
        self.part = None

    def frame(self):
        self.count += 1
        if self.part is None:
            return sim.empty_frame(seed=self.seed * 7 + self.count, drift=self.drift, scene=self.scene)
        p = self.part
        return sim.render(p["shape"], p["face"], p["defect"], seed=p["seed"], angle=p["angle"], drift=self.drift,
                          severity=p.get("severity", 1.0), scene=self.scene)["frame"]

    def burst(self, n=8, settle_s=0.0):
        if self.part is None:
            return [self.frame() for _ in range(n)]
        p = self.part
        frames, _ = sim.burst(p["shape"], p["face"], p["defect"], seed=p["seed"], n=n, angle=p["angle"],
                              scene=self.scene, drift=self.drift, severity=p.get("severity", 1.0))
        return frames

    def close(self):
        pass


def open_camera(cfg, simulate=False, seed=0):
    return SimCamera(seed, scene=cfg.get("sim_scene")) if simulate else OpenCVCamera(cfg)
