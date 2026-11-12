#!/usr/bin/env python3
"""Pictures for the docs and build reports, all from computer-generated scenes (qc/sim.py).

Changelog
  0.1.0  2026-11-12  first version.

    python tools/doc_pictures.py docs/screenshots/2026-11-12
    python tools/doc_pictures.py <out> --terminal some_output.txt --terminal-name run_simulate_terminal.png
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from qc import background as bgmod  # noqa: E402
from qc import capture, sim  # noqa: E402
from qc import straighten as st  # noqa: E402


def label(img, text, y=18, scale=0.5):
    out = img.copy()
    cv2.putText(out, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def caption(img, lines, h=None):
    h = h or 22 * len(lines) + 10
    bar = np.full((h, img.shape[1], 3), 255, np.uint8)
    for k, t in enumerate(lines):
        cv2.putText(bar, t, (8, 20 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    return np.vstack([img, bar])


def straighten_before_after(ref, out):
    top, bottom = [], []
    R = RM = None
    for s in range(6):
        r = sim.render("square", "bottom", seed=300 + s)
        loc = ref.part_mask(r["frame"])
        o = st.straighten(r["frame"], loc, R, 300, 224, None, ref_mask=RM)
        if R is None:
            R, RM = o["image"], o["mask"]
        x, y = int(loc.centre[0]), int(loc.centre[1])
        crop = r["frame"][y - 150:y + 150, x - 150:x + 150]
        top.append(label(cv2.resize(crop, (180, 180)), "as placed"))
        bottom.append(label(cv2.resize(o["image"], (180, 180)), "turned %+.0f deg" % o["angle_deg"]))
    img = np.vstack([np.hstack(top), np.hstack(bottom)])
    img = caption(img, ["Straightening: six simulated square parts (bottom face, 'A7' stamp) at random angles (top row)",
                        "and after straightening (bottom row). The outline gives the angle up to 90 deg; the stamp picks",
                        "which of the four. Data: qc/sim.py, seeds 300-305."])
    cv2.imwrite(os.path.join(out, "straighten_before_after_square_bottom.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])


def drift_examples(ref, out):
    tiles = []
    e = sim.empty_frame(seed=11)
    d = ref.drift(e)
    tiles.append(label(cv2.resize(e, (256, 192)), "empty: %s" % ("DRIFT" if d["drift"] else "ok")))
    for k in sim.DRIFTS:
        f = sim.empty_frame(seed=12, drift=k)
        d = ref.drift(f)
        g = f.copy()
        if d["drift"]:
            cv2.circle(g, d["where"], 28, (0, 0, 255), 3)
        tiles.append(label(cv2.resize(g, (256, 192)), "%s: %s" % (k, "DRIFT" if d["drift"] else "ok")))
    img = np.hstack(tiles)
    img = caption(img, ["Background drift check on simulated empty frames: normal, then four kinds of change. The red ring",
                        "marks the biggest change found (pictures brightened 3x here so the black backdrop is visible).",
                        "Data: qc/sim.py; limits from 10 empty frames (leave-one-out), qc/background.py."])
    img[:192] = np.clip(img[:192].astype(np.int32) * 3, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out, "drift_check_empty_frames.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])


def burst_pick(ref, out):
    frames, settled = sim.burst("hexagon", "top", seed=4)
    loc = ref.part_mask(frames[-1])
    i, s = capture.pick_sharpest(frames, loc.filled)
    tiles = []
    for k, f in enumerate(frames):
        x, y = int(loc.centre[0]), int(loc.centre[1])
        t = cv2.resize(f[y - 130:y + 130, x - 130:x + 130], (150, 150))
        ok, mv, lim = capture.is_still(frames, k, ref.still_noise, loc.filled)
        t = label(t, "sharp %.0f" % s[k], 16, 0.45)
        t = label(t, "%s %.1f" % ("still" if ok else "moving", mv), 140, 0.45)
        if k == i:
            cv2.rectangle(t, (1, 1), (148, 148), (0, 200, 0), 3)
        tiles.append(t)
    img = np.hstack(tiles)
    img = caption(img, ["A burst of 8 frames after 'paused' (simulated): the first ones still slide and blur. 'sharp' =",
                        "variance of the Laplacian on the part; 'still/moving' = change vs. the nearest frame (limit %.1f, from" % (
                            capture.STILL_FACTOR * ref.still_noise),
                        "the empty frames). Green box = the frame kept. Data: qc/sim.py burst, seed 4."])
    cv2.imwrite(os.path.join(out, "burst_sharpest_frame_pick.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])


def terminal(text_path, out_path, title, head=22, tail=8, width=125):
    """Terminal output as a picture: the first `head` and last `tail` lines, long lines wrapped."""
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    raw = [l for l in open(text_path).read().rstrip().splitlines() if l.strip()]
    if len(raw) > head + tail:
        raw = raw[:head] + ["   ... (%d lines left out) ..." % (len(raw) - head - tail)] + raw[-tail:]
    lines = []
    for l in raw:
        lines += textwrap.wrap(l, width, subsequent_indent="    ") or [""]
    h = 0.16 * (len(lines) + 2) + 0.3
    fig = plt.figure(figsize=(11, h))
    fig.patch.set_facecolor("#1e1e1e")
    fig.text(0.01, 1 - 0.12 / h, title, color="#9cdcfe", family="monospace", fontsize=9, va="top")
    fig.text(0.01, 1 - 0.45 / h, "\n".join(lines), color="#e8e8e8", family="monospace", fontsize=8, va="top",
             linespacing=1.35)
    fig.savefig(out_path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--terminal", default=None)
    ap.add_argument("--terminal-name", default="terminal.png")
    ap.add_argument("--terminal-title", default="$ python -m qc run --simulate")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.terminal:
        terminal(a.terminal, os.path.join(a.out, a.terminal_name), a.terminal_title)
        return 0
    ref = bgmod.EmptyReference.from_frames([sim.empty_frame(seed=100 + i) for i in range(10)])
    straighten_before_after(ref, a.out)
    drift_examples(ref, a.out)
    burst_pick(ref, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
