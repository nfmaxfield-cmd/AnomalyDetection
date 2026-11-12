"""Clock line-up (SWITCH clock_lineup, off by default): match recorded camera frames to an arm log
that was written with a different clock.

Changelog
  0.1.0  2026-11-12  first version. Adapted from alvik-fleet-capstone tools/recognition/outside.py
                     estimate_time_offset() (there: robot speed from its log vs. speed seen by the camera;
                     here: "arm says paused" vs. "picture is still").

Not needed for live runs: there the arm asks and the camera answers at once, on one computer. It is for
offline work, e.g. a video recorded by a phone or a second computer while the arm wrote its own log of
"paused" / "moving" times. The two clocks can differ by seconds. The offset is found from the data:

  1. motion in the video over time (mean frame-to-frame change),
  2. the arm's state over time (1 = moving, 0 = paused),
  3. the shift of the arm's times that makes the two agree best (correlation), searched over +/- search_s.

    off, corr = estimate_offset(frame_times, frames_or_motion, arm_events, search_s=5)
    windows = paused_windows(arm_events, off)           # [(t_start, t_end)] on the camera's clock
    keep = frames_in_windows(frame_times, windows, margin_s=0.2)
"""
import numpy as np


def motion_signal(frames):
    """Mean absolute change between consecutive frames (first value repeated)."""
    m = [0.0]
    for a, b in zip(frames, frames[1:]):
        m.append(float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()))
    m[0] = m[1] if len(m) > 1 else 0.0
    return np.array(m)


def arm_state(events, t):
    """events: [(t, "paused"|"moving")] sorted; returns 1.0 where moving, 0.0 where paused, at times t."""
    ev_t = np.array([e[0] for e in events], float)
    ev_s = np.array([0.0 if e[1] == "paused" else 1.0 for e in events])
    idx = np.searchsorted(ev_t, t, side="right") - 1
    out = np.where(idx >= 0, ev_s[np.clip(idx, 0, None)], 1.0)
    return out


def estimate_offset(frame_times, motion, events, search_s=5.0, step_s=0.02):
    """Offset to ADD to the arm's times to put them on the camera's clock, and the correlation reached."""
    t = np.asarray(frame_times, float)
    mo = np.asarray(motion, float)
    best = (None, -2.0)
    for off in np.arange(-search_s, search_s + 1e-9, step_s):
        s = arm_state([(e[0] + off, e[1]) for e in events], t)
        if s.std() < 1e-9 or mo.std() < 1e-9:
            continue
        c = float(np.corrcoef(s, mo)[0, 1])
        if c > best[1]:
            best = (float(off), c)
    return best


def paused_windows(events, offset=0.0):
    out = []
    start = None
    for t, s in events:
        if s == "paused" and start is None:
            start = t + offset
        elif s != "paused" and start is not None:
            out.append((start, t + offset))
            start = None
    if start is not None:
        out.append((start, float("inf")))
    return out


def frames_in_windows(frame_times, windows, margin_s=0.2):
    """Indices of frames well inside a paused window (margin_s after it starts and before it ends)."""
    t = np.asarray(frame_times, float)
    keep = np.zeros(len(t), bool)
    for a, b in windows:
        keep |= (t >= a + margin_s) & (t <= b - margin_s)
    return np.nonzero(keep)[0]
