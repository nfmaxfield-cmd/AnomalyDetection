"""Day-one camera check: does the webcam do what the station needs?  python -m qc camera-check

Changelog
  0.1.0  2026-11-12  first version (audit H4): settings read back, settings after reopening the camera,
                     old buffered frames in a burst, automatic exposure / gain still active.

Four checks, each with a plain verdict and what to do:
  1. Settings: every locked setting (auto exposure off, exposure, gain, white balance, focus) is set and read
     back. A setting the camera does not keep is named.
  2. Reopen: the camera is closed and opened again; some drivers forget their settings on reopening.
  3. Old frames: you hold a hand under the camera, then take it away and press Enter. The burst taken after
     Enter must not show the hand. It is run twice: with the reader thread (what the station uses) and
     with the first build's plain reads (shows whether this camera keeps old frames in a buffer).
  4. Automatic exposure: you cover half the view with a white sheet. If the uncovered half gets darker or
     brighter, the camera is still adjusting exposure or gain by itself.
The result is printed and saved as runs/camera_check_<time>.json (send it to Nick if anything fails).
"""
import datetime as dt
import json
import os

import cv2
import numpy as np

from . import capture


def _grey(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) if f.ndim == 3 else f.astype(np.float32)


def noise_between(frames):
    """Typical mean absolute change between two frames of a still scene (the unit for the checks below)."""
    if len(frames) < 2:
        return 1.0
    d = [float(np.abs(_grey(a) - _grey(b)).mean()) for a, b in zip(frames[:-1], frames[1:])]
    return max(0.5, float(np.median(d)))


def stale_frames(with_hand, burst, after):
    """Indices of burst frames that look more like the picture WITH the hand than like the scene after the
    hand was taken away (`after`): frames that waited in a buffer."""
    h, a = _grey(with_hand), _grey(after)
    out = []
    for i, f in enumerate(burst):
        g = _grey(f)
        if np.abs(g - h).mean() < np.abs(g - a).mean():
            out.append(i)
    return out


def brightness_shift(before, after, region):
    """Mean brightness change in `region` (a boolean mask: the part of the view NOT covered by the sheet)."""
    return float(np.mean([_grey(f)[region].mean() for f in after]) - np.mean([_grey(f)[region].mean() for f in before]))


def run(cfg, ask=input, out=print, capture_factory=None, root="."):
    res = dict(time=dt.datetime.now().isoformat(timespec="seconds"), backend=capture.pick_backend(cfg.get("camera_backend", "auto")),
               problems=[])
    out("Camera check, camera %s, backend %s" % (cfg.get("camera_index", 0), res["backend"]))
    # 1. settings
    cam = capture.OpenCVCamera(dict(cfg, camera_reader_thread=True), capture_factory=capture_factory)
    res["size"] = list(cam.size)
    res["settings"] = cam.settings_report
    out("\n1. Settings (asked -> what the camera reports):")
    for r in cam.settings_report:
        out("   %-24s %-10s -> %-10s %s" % (r["name"], r["asked"] if isinstance(r["asked"], str) else "%g" % r["asked"],
                                           "%g" % r["read_back"], "held" if r["held"] else "DID NOT HOLD"))
    for w in cam.warnings:
        res["problems"].append(w)
    first = {r["name"]: r["read_back"] for r in cam.settings_report}
    cam.close()
    # 2. reopen
    cam = capture.OpenCVCamera(dict(cfg, camera_reader_thread=True), capture_factory=capture_factory)
    again = {r["name"]: r["read_back"] for r in cam.settings_report}
    changed = [k for k in first if k in again and abs(first[k] - again[k]) > 1e-6]
    res["reopen_changed"] = changed
    out("\n2. Closed and opened again: %s" % ("settings unchanged" if not changed else
                                              "CHANGED: %s" % ", ".join(changed)))
    if changed:
        res["problems"].append("settings changed after reopening the camera: %s" % ", ".join(changed))
    # 3. old frames, with the reader thread and with plain reads
    n = int(cfg.get("burst_frames", 8))
    res["stale"] = {}
    for label, c in (("reader thread (what the station uses)", cam), ("plain reads, no drain (the first build)", None)):
        if c is None:
            c = capture.OpenCVCamera(dict(cfg, camera_reader_thread=False), capture_factory=capture_factory)
        ask("\n3. Old frames, %s: hold your hand under the camera, keep it there, and press Enter ..." % label)
        if c is not cam:
            c.drain()                     # the reference picture of the hand itself must be a fresh one
        hand = c.frame()
        ask("   Now take your hand fully out of view and press Enter straight away ...")
        burst = c.burst(n) if c is cam else [c._read() for _ in range(n)]
        after = c.burst(2)[-1]
        stale = stale_frames(hand, burst, after)
        res["stale"][label] = stale
        out("   burst of %d after Enter: %s" % (n, "no old frames" if not stale else
                                                "frames %s still show the hand (old, buffered)" % stale))
        if stale and c is cam:
            res["problems"].append("the station's burst contained old frames %s: tell Nick" % stale)
        if c is not cam:
            c.close()
    # 4. automatic exposure / gain
    ask("\n4. Clear the view (spot empty) and press Enter ...")
    before = cam.burst(5)
    ask("   Lay a WHITE sheet of paper over the LEFT half of the view only, and press Enter ...")
    after = cam.burst(5)
    h, w = before[0].shape[:2]
    right = np.zeros((h, w), bool)
    right[:, int(w * 0.75):] = True
    shift = brightness_shift(before, after, right)
    noise = noise_between(before)
    res["auto_exposure_shift"] = shift
    res["auto_exposure_noise"] = noise
    held = abs(shift) <= 3 * noise           # [chosen] 3x the frame-to-frame change; not DOE
    out("   uncovered right quarter changed by %+.1f grey levels (still-scene change %.1f): %s" % (
        shift, noise, "exposure held" if held else "the camera is STILL ADJUSTING exposure or gain by itself"))
    if not held:
        res["problems"].append("automatic exposure or gain is still active (brightness moved %+.1f)" % shift)
    cam.close()
    d = os.path.join(root, cfg.get("log_dir", "runs"))
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "camera_check_%s.json" % dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    with open(p, "w") as f:
        json.dump(res, f, indent=2, default=float)
    res["file"] = p
    out("\nRESULT: %s" % ("all checks passed" if not res["problems"] else
                          "%d problem(s):\n  - %s" % (len(res["problems"]), "\n  - ".join(res["problems"]))))
    out("Saved: %s" % p)
    return res
