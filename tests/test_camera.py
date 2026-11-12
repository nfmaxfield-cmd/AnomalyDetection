"""The webcam path without a webcam (audit H4, L4): a fake OpenCV capture that buffers old frames, ignores
some settings, or stops giving pictures. Backend choice per system, setting read-back, the reader thread
and the buffer drain, retry and reopen, and the day-one camera check.

Changelog
  0.1.0  2026-11-12  first version (audit fixes).
"""
import threading
import time

import cv2
import numpy as np
import pytest

from qc import camcheck, capture


class FakeCap:
    """Pretend cv2.VideoCapture. Frames are 'made' at fps; up to `buffer` unread frames wait in a queue and
    read() returns the OLDEST waiting one (as UVC drivers do). Each frame's brightness is the scene value at
    the moment it was made. Settings in `ignore` are not kept (get returns the old value)."""
    opened = []

    def __init__(self, idx, api, fps=50.0, buffer=4, ignore=(), fail_reads=0, scene=None):
        self.api, self.fps, self.buffer, self.ignore = api, fps, buffer, set(ignore)
        self.props = {cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 480, cv2.CAP_PROP_FPS: fps,
                      cv2.CAP_PROP_AUTO_EXPOSURE: 3, cv2.CAP_PROP_EXPOSURE: 150, cv2.CAP_PROP_GAIN: 64,
                      cv2.CAP_PROP_AUTO_WB: 1, cv2.CAP_PROP_WB_TEMPERATURE: 4600, cv2.CAP_PROP_AUTOFOCUS: 1,
                      cv2.CAP_PROP_FOCUS: 20}
        self.t0 = time.time()
        self.next = 0
        self.fail_reads = fail_reads
        self.qleft = 0
        self.jump_to = None
        self.scene = scene if scene is not None else {"v": 50}
        self.history = [(self.t0, self.scene["v"])]
        FakeCap.opened.append(self)

    def isOpened(self):
        return True

    def set(self, p, v):
        if p not in self.ignore:
            self.props[p] = float(v)
        return True

    def get(self, p):
        return float(self.props.get(p, 0.0))

    def set_scene(self, v):
        self.scene["v"] = v
        self.history.append((time.time(), v))

    def _value_at(self, t):
        v = self.history[0][1]
        for tt, vv in self.history:
            if tt <= t:
                v = vv
        return v

    def grab(self):
        now = time.time()
        made = int((now - self.t0) * self.fps)                   # frames made so far
        if self.qleft > 0:                                       # still reading out an old, full queue
            self.qleft -= 1
        elif made - self.next > self.buffer:                     # queue full: its `buffer` OLDEST frames wait,
            self.qleft = self.buffer - 1                         # newer ones were dropped
            self.jump_to = made
        elif self.next >= made:                                  # nothing waiting: wait for the next frame
            time.sleep(max(0.0, self.t0 + (self.next + 1) / self.fps - now))
        self._last = self.next
        self.next += 1
        if self.qleft == 0 and self.jump_to is not None:         # queue read out: next frame is live again
            self.next = max(self.next, self.jump_to)
            self.jump_to = None
        return True

    def retrieve(self):
        t = self.t0 + self._last / self.fps
        return True, np.full((48, 64, 3), self._value_at(t), np.uint8)

    def read(self):
        if self.fail_reads > 0:
            self.fail_reads -= 1
            return False, None
        self.grab()
        return self.retrieve()

    def release(self):
        pass


def cfg(**kw):
    c = dict(camera_index=0, camera_backend="v4l2", frame_width=640, frame_height=480, lock_camera_settings=True,
             exposure=None, gain=None, white_balance_k=None, focus=None, camera_reader_thread=True)
    c.update(kw)
    return c


def test_backend_is_chosen_per_system():
    assert capture.pick_backend("auto", "win32") == "dshow"
    assert capture.pick_backend("auto", "linux") == "v4l2"
    assert capture.pick_backend("auto", "darwin") == "avfoundation"
    assert capture.pick_backend("msmf", "win32") == "msmf"
    with pytest.raises(ValueError):
        capture.pick_backend("nonsense")


def test_settings_are_read_back_and_a_setting_that_does_not_hold_is_named():
    cam = capture.OpenCVCamera(cfg(exposure=80, focus=40, camera_reader_thread=False),
                               capture_factory=lambda i, api: FakeCap(i, api, ignore={cv2.CAP_PROP_FOCUS}))
    rep = {r["name"]: r for r in cam.settings_report}
    assert rep["exposure"]["held"] and rep["exposure"]["read_back"] == 80
    assert rep["auto exposure off"]["held"]
    assert not rep["focus"]["held"]
    assert any(w.startswith("focus") for w in cam.warnings)
    assert rep["gain"]["kept_current"] and rep["gain"]["held"]           # gain locked at its current value
    cam.close()


def test_mac_backend_warns_that_exposure_cannot_be_set():
    cam = capture.OpenCVCamera(cfg(camera_backend="avfoundation", camera_reader_thread=False),
                               capture_factory=lambda i, api: FakeCap(i, api))
    assert any("macOS" in w for w in cam.warnings)
    cam.close()


def _stale_test(cam, fake):
    """Hand in view, a pause, hand away, then a burst: how many burst frames still show the hand?"""
    fake.set_scene(200)               # hand in view
    time.sleep(0.5)                   # nobody reads: the camera's queue fills with hand frames
    fake.set_scene(50)                # hand away; "paused" arrives now
    burst = cam.burst(6)
    return sum(int(f.mean() > 120) for f in burst)


def test_direct_reads_without_drain_return_old_frames():
    """What the first build did: the first frames of a burst were made before the request (the problem)."""
    fake = FakeCap(0, 0, buffer=4)
    cam = capture.OpenCVCamera(cfg(camera_reader_thread=False), capture_factory=lambda i, api: fake)
    cam.drain = lambda: None          # the first build had no drain
    assert _stale_test(cam, fake) > 0
    cam.close()


def test_drain_and_reader_thread_give_only_new_frames():
    fake = FakeCap(0, 0, buffer=4)
    cam = capture.OpenCVCamera(cfg(camera_reader_thread=False), capture_factory=lambda i, api: fake)
    assert _stale_test(cam, fake) == 0
    cam.close()
    fake = FakeCap(0, 0, buffer=4)
    cam = capture.OpenCVCamera(cfg(camera_reader_thread=True), capture_factory=lambda i, api: fake)
    assert _stale_test(cam, fake) == 0
    cam.close()


def test_failed_reads_are_retried_and_the_camera_reopened():
    made = []

    def factory(i, api):
        f = FakeCap(i, api, fail_reads=0 if made else 0)
        made.append(f)
        return f
    cam = capture.OpenCVCamera(cfg(camera_reader_thread=False), capture_factory=factory)
    made[-1].fail_reads = 4           # more failures than the quick retries: the camera is reopened
    f = cam.frame()
    assert f is not None and len(made) >= 2
    cam.close()


def test_camera_that_never_recovers_raises_a_plain_error():
    class Dead(FakeCap):
        def read(self):
            return False, None
    cam = None
    with pytest.raises(RuntimeError, match="stopped giving pictures"):
        cam = capture.OpenCVCamera(cfg(camera_reader_thread=False), capture_factory=lambda i, api: Dead(i, api))
        cam.frame()


def test_camera_check_runs_and_reports(tmp_path):
    fakes = []
    scene = {"v": 50}

    def factory(i, api):
        f = FakeCap(i, api, ignore={cv2.CAP_PROP_WB_TEMPERATURE}, scene=scene)
        fakes.append(f)
        return f
    steps = iter(["hand", "away", "hand", "away", "clear", "sheet"])

    def ask(prompt):
        k = next(steps)
        v = {"hand": 200, "away": 50, "clear": 50, "sheet": 50}[k]
        if k == "away":
            time.sleep(0.3)               # a person takes a moment to move the hand away
        for f in fakes:
            f.set_scene(v)
        if k == "hand":
            time.sleep(0.3)
        return ""
    out = []
    r = camcheck.run(cfg(white_balance_k=5000), ask=ask, out=out.append, capture_factory=factory, root=str(tmp_path))
    assert r["stale"]["reader thread (what the station uses)"] == []
    assert r["stale"]["plain reads, no drain (the first build)"]          # the fake buffers: shown as a finding
    assert any("white balance" in p for p in r["problems"])
    assert abs(r["auto_exposure_shift"]) < 3 * r["auto_exposure_noise"] + 1e-6
    assert "RESULT" in out[-2] and (tmp_path / "runs").exists()


def test_stale_and_brightness_helpers():
    hand = np.full((10, 10, 3), 200, np.uint8)
    empty = np.full((10, 10, 3), 50, np.uint8)
    assert camcheck.stale_frames(hand, [hand, empty, empty], empty) == [0]
    reg = np.ones((10, 10), bool)
    assert camcheck.brightness_shift([empty], [empty + 10], reg) == pytest.approx(10)
