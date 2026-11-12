#!/usr/bin/env python3
"""Pictures for the audit-fixes report (docs/build-reports/2026-11-12-audit-fixes.md), all computer-generated.

Changelog
  0.1.0  2026-11-12  first version.

    python tools/audit_pictures.py docs/screenshots/2026-11-12 [--h2 h2_table.json] [--probes old.json new.json]
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from qc import background as bgmod  # noqa: E402
from qc import capture, charts, lens, sim  # noqa: E402

sys.path.insert(0, HERE)
from doc_pictures import caption, label  # noqa: E402


def scenes(out):
    """The same parts in the easy and the hard scene, and one defect at the three severities (hard)."""
    tiles = []
    for scene in ("easy", "hard"):
        row = []
        for shape, face, seed in (("square", "bottom", 11), ("disc", "top", 12), ("hexagon", "top", 13)):
            r = sim.render(shape, face, seed=seed, scene=scene)
            cx, cy = [int(round(v)) for v in r["centre"]]
            y0, x0 = int(np.clip(cy - 130, 0, sim.H - 260)), int(np.clip(cx - 130, 0, sim.W - 260))
            row.append(label(r["frame"][y0:y0 + 260, x0:x0 + 260], "%s scene, %s/%s" % (scene, shape, face)))
        tiles.append(np.hstack(row))
    row = []
    for name, sev in sim.SEVERITIES:
        r = sim.render("square", "top", "scratch", seed=21, severity=sev, scene="hard")
        cx, cy = [int(round(v)) for v in r["centre"]]
        y0, x0 = int(np.clip(cy - 130, 0, sim.H - 260)), int(np.clip(cx - 130, 0, sim.W - 260))
        row.append(label(r["frame"][y0:y0 + 260, x0:x0 + 260], "hard, scratch, %s (x%.2f)" % (name, sev)))
    tiles.append(np.hstack(row))
    pic = caption(np.vstack(tiles), [
        "Top: the first build's easy scene. Middle: the hard scene (now the default): light brighter on the right and a",
        "reflection, both fixed in the room; more part-to-part variation; faint handling marks on good parts; noisier",
        "JPEG camera; parts placed less precisely. Bottom: one scratch at normal, subtle and faint severity (hard scene).",
        "Data: qc/sim.py, computer-generated."])
    cv2.imwrite(os.path.join(out, "sim_easy_vs_hard_scene.jpg"), pic, [cv2.IMWRITE_JPEG_QUALITY, 85])


def stillness_chart(out):
    """Old and new stillness check against slide speed, both scenes."""
    plt = charts._plt()
    speeds = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    fig.subplots_adjust(bottom=0.3, top=0.86, left=0.1, right=0.97)
    colours = {("easy", "old"): charts.MUTED, ("easy", "new"): charts.BLUE, ("hard", "old"): charts.YELLOW,
               ("hard", "new"): charts.ORANGE}
    for scene in ("easy", "hard"):
        ref = bgmod.EmptyReference.from_frames([sim.empty_frame(seed=100 + i, scene=scene) for i in range(10)])
        old, new = [], []
        for v in speeds:
            o, n = [], []
            for seed in range(3):
                fr = [sim.render("square", "top", seed=600 + seed, angle=30, centre=(320 + v * k, 240), motion_px=v,
                                 noise_seed=seed * 31 + k, scene=scene)["frame"] for k in range(8)]
                loc = ref.part_mask(fr[-1])
                i, _ = capture.pick_sharpest(fr, loc.filled)
                o.append(capture.motion(fr, i, loc.filled) / (capture.STILL_FACTOR * ref.still_noise))
                ok, m, lim = capture.is_still(fr, i, ref.still_noise, loc.filled)
                n.append(m / lim)
            old.append(np.median(o))
            new.append(np.median(n))
        ax.plot(speeds, old, color=colours[(scene, "old")], marker="o", linewidth=2, linestyle="--",
                label="first build, %s scene" % scene)
        ax.plot(speeds, new, color=colours[(scene, "new")], marker="o", linewidth=2, label="now, %s scene" % scene)
    ax.axhline(1.0, color=charts.INK, linewidth=1.2)
    ax.text(speeds[-1], 1.02, "above this line = called moving", ha="right", va="bottom", fontsize=9, color=charts.INK)
    ax.set_xlabel("part sliding (pixels per frame) during an 8-frame burst")
    ax.set_ylabel("motion measure / its limit")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Stillness check: how far each check's motion measure is from its limit", fontsize=11,
                 color=charts.INK, x=0.01, ha="left")
    charts._footer(fig, "computer-generated bursts (qc/sim.py), square/top part sliding at a steady speed, median of 3 "
                        "bursts per speed; qc/capture.py first build and now.",
                   "The first build called a part sliding 1 pixel per frame still; the new check calls it moving "
                   "from about half a pixel per frame on, and a still part stays well below its limit in both scenes.")
    fig.savefig(os.path.join(out, "stillness_check_slow_slide.png"), dpi=110)
    plt.close(fig)


def h2_chart(table, out):
    plt = charts._plt()
    t = json.load(open(table))
    names = list(t)
    short = ["clearly damaged", "damage not on\nthis face (5.4)", "faint (5.9)", "below the good\nmedian (5.25)"]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5.2))
    fig.subplots_adjust(bottom=0.32, top=0.86, left=0.1, right=0.97)
    ax.bar(x - 0.2, [100 * t[n]["old_fr"] for n in names], 0.38, color=charts.MUTED, label="first build's rule")
    ax.bar(x + 0.2, [100 * t[n]["new_fr"] for n in names], 0.38, color=charts.BLUE, label="now")
    for k, n in enumerate(names):
        for dx, v in ((-0.2, t[n]["old_fr"]), (0.2, t[n]["new_fr"])):
            ax.text(k + dx, 100 * v + 0.8, "%.1f %%" % (100 * v), ha="center", fontsize=9, color=charts.INK)
    ax.set_xticks(x)
    ax.set_xticklabels(short[:len(names)], fontsize=9)
    ax.set_ylabel("good parts rejected in the long run (%)")
    ax.set_xlabel("what the 10th of 10 damaged pictures looks like (9 are clearly damaged)")
    ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Threshold rule with damaged parts: one odd damaged picture", fontsize=11, color=charts.INK, x=0.01,
                 ha="left")
    charts._footer(fig, "20 held-out good scores drawn from a Gumbel distribution (location 5, scale 0.5) plus 10 damaged "
                        "scores, 200 draws, median shown; long-run share from the true distribution; qc/threshold.py "
                        "first build and now (expected defect rate 5 %, cost 10:1, cap 2 %).",
                   "One damaged picture that scores like a good part used to drag the line into the good parts "
                   "(17-46 % scrapped); now it is named for a person to check and the line stays at about 1 %.")
    fig.savefig(os.path.join(out, "threshold_rule_audit_cases.png"), dpi=110)
    plt.close(fig)


def speed_chart(out):
    """Same-spot comparison time, first build's layout vs cell-first, against memory size."""
    plt = charts._plt()
    rng = np.random.default_rng(0)
    h = w = 28
    d = 128
    sizes = [48 * 3, 80 * 3, 60 * 7, 80 * 7]
    t_old, t_new = [], []
    for M in sizes:
        bank = rng.normal(0, 1, (M, h, w, d)).astype(np.float32)
        z = rng.normal(0, 1, (h, w, d)).astype(np.float32)
        pad = np.pad(bank, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
        pn = (pad ** 2).sum(-1)
        B = np.ascontiguousarray(pad.transpose(1, 2, 0, 3))
        BN = np.ascontiguousarray(pn.transpose(1, 2, 0))

        def old():
            best = np.full((h, w), np.inf, np.float32)
            for dy in range(3):
                for dx in range(3):
                    dot = np.einsum("mhwd,hwd->mhw", pad[:, dy:dy + h, dx:dx + w], z, optimize=True)
                    best = np.minimum(best, (pn[:, dy:dy + h, dx:dx + w] - 2 * dot).min(0))
            return best

        def new():
            best = np.full((h, w), np.inf, np.float32)
            for dy in range(3):
                for dx in range(3):
                    dot = np.matmul(B[dy:dy + h, dx:dx + w], z[..., None])[..., 0]
                    best = np.minimum(best, (BN[dy:dy + h, dx:dx + w] - 2 * dot).min(-1))
            return best
        for fn, acc in ((old, t_old), (new, t_new)):
            fn()
            t0 = time.perf_counter()
            for _ in range(2):
                fn()
            acc.append(1000 * (time.perf_counter() - t0) / 2)
        del bank, pad, B
    fig, ax = plt.subplots(figsize=(8, 5.0))
    fig.subplots_adjust(bottom=0.3, top=0.86, left=0.1, right=0.97)
    ax.plot(sizes, t_old, color=charts.MUTED, marker="o", linewidth=2, label="first build (picture-first memory)")
    ax.plot(sizes, t_new, color=charts.BLUE, marker="o", linewidth=2, label="now (cell-first memory)")
    for s_, a, b in zip(sizes, t_old, t_new):
        ax.annotate("%.0f" % a, (s_, a), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)
        ax.annotate("%.0f" % b, (s_, b), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)
    ax.set_xlabel("pictures in memory (training pictures x copies: 48x3, 80x3, 60x7, 80x7)")
    ax.set_ylabel("ms per picture, memory comparison only")
    ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Same-spot memory comparison time per picture", fontsize=11, color=charts.INK, x=0.01, ha="left")
    charts._footer(fig, "random features of the real size (28 x 28 cells, 128 numbers), qc/model.py layouts, build "
                        "machine with 2 CPU cores while other jobs ran (upper values).",
                   "Laying the memory out cell-first makes the comparison several times faster at every size, which pays "
                   "for the shifted training copies (60 x 7 pictures).")
    fig.savefig(os.path.join(out, "memory_compare_speed.png"), dpi=110)
    plt.close(fig)


def calibration_pictures(out):
    png, _ = lens.make_card(os.path.join(out, "_card"))
    card = cv2.imread(png)
    card = cv2.resize(card, (card.shape[1] // 3, card.shape[0] // 3), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(out, "calibration_card_print.png"), caption(card, [
        "The printable calibration card (python -m qc make-card): 9 x 6 squares of 20 mm, a 100 mm check line,",
        "the square size printed. Shown at a third of its 300 dpi size. Data: qc/lens.py."]))
    for p in (png, os.path.join(out, "_card.pdf")):
        os.remove(p)
    views = [lens.sim_view(100 + k)[0] for k in range(12)]
    flat = lens.sim_view(999, flat=True)[0]
    r = lens.calibrate(views, (8, 5), 20.0, flat=flat)
    lc = lens.LensCorrection(r)
    f = lens.sim_view(998, flat=True, height_mm=240.0)[0][0]      # the card close up: it fills the picture
    g = lc.undistort(f)

    def lines(img):
        out_ = img.copy()
        c = lens.find_corners(img, (8, 5)).reshape(5, 8, 2)
        for row in (c[0], c[-1]):
            a, b = row[0], row[-1]
            cv2.line(out_, tuple(int(v) for v in a), tuple(int(v) for v in b), (0, 0, 255), 1, cv2.LINE_AA)
            for p in row:
                cv2.circle(out_, tuple(int(round(v)) for v in p), 2, (0, 200, 0), -1)
        return out_
    pic = np.hstack([label(lines(f), "as the lens sees it"), label(lines(g), "after calibration (undistorted)")])
    pic = caption(pic, [
        "Simulated webcam with barrel distortion, card held close so it fills the picture. Green: corners; red: straight line",
        "through the end corners of the first and last row. Before: the corners bow off the line; after: they lie on it.",
        "12 tilted views: reprojection error %.3f px (limit %.3f px); %.4f mm per pixel (true %.4f)." % (
            r["rms"], r["limit"], r["mm_per_px"], 400 / 560.0),
        "Data: qc/lens.py sim_view (computer-generated), calibrate, LensCorrection."])
    cv2.imwrite(os.path.join(out, "lens_calibration_before_after.jpg"), pic, [cv2.IMWRITE_JPEG_QUALITY, 88])


def text_picture(lines, out_path, title, width=125):
    from doc_pictures import terminal
    tmp = out_path + ".txt"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    terminal(tmp, out_path, title, head=80, tail=0, width=width)
    os.remove(tmp)


def probes_picture(old_p, new_p, out):
    old, new = json.load(open(old_p)), json.load(open(new_p))
    L = ["%-38s | %-30s | %s" % ("case (good part unless named)", "first build", "now"), "-" * 118]
    for a, b in zip(old, new):
        def fmt(r):
            if r["case"].startswith("collect"):
                return "picture not kept, attention" if r.get("ok") is False else "picture kept"
            s = r.get("decision") or ("not kept" if r.get("ok") is False else "kept")
            if r.get("part") not in (None, "") and r.get("case", "").startswith(("p1", "NEW", "no part")):
                s += ", part " + str(r["part"])
            if r.get("attention"):
                s += ", attention"
            if r.get("recheck"):
                s += ", recheck"
            return s
        L.append("%-38s | %-30s | %s" % (a["case"][:38], fmt(a)[:30], fmt(b)))
        if b.get("reason"):
            L.append("%-38s | %-30s |   %s" % ("", "", b["reason"].split(" (")[0].split(": put")[0][:48]))
    text_picture(L, os.path.join(out, "station_probes_before_after.png"),
                 "audit probes on the simulated station (easy scene, ResNet-18): first build vs now")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--h2", default=None)
    ap.add_argument("--probes", nargs=2, default=None)
    ap.add_argument("--only", default=None, help="comma list: scenes,still,h2,speed,lens,probes")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    only = set(a.only.split(",")) if a.only else None

    def want(k):
        return only is None or k in only
    if want("scenes"):
        scenes(a.out)
    if want("still"):
        stillness_chart(a.out)
    if want("h2") and a.h2:
        h2_chart(a.h2, a.out)
    if want("speed"):
        speed_chart(a.out)
    if want("lens"):
        calibration_pictures(a.out)
    if want("probes") and a.probes:
        probes_picture(a.probes[0], a.probes[1], a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
