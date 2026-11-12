#!/usr/bin/env python3
"""Detection results on the computer-generated set: shape x face x defect type -> caught / missed.

Changelog
  0.1.0  2026-11-12  first version.

    python tools/evaluate_synthetic.py --out docs/results/synthetic --backbone onnx --bank local
    python tools/evaluate_synthetic.py --quick            # fewer pictures, for a fast check

For every shape and face, all pictures made by qc/sim.py with fixed seeds (so every run is the same):
  train      N_TRAIN good parts                     -> the normal model
  holdout    N_HOLD good parts + 2 damaged per type -> the threshold (qc/threshold.py, from data)
  test       N_TEST good parts (false rejects) + N_DEF per defect type at normal severity, and the same
             again at "subtle" severity (half the size and contrast) -> caught / missed
Also: does the heat map's hottest spot fall on the defect (within LOC_TOL_PX of it)? and the background
drift check on empty frames and on part pictures, with and without each kind of drift.

Writes results.json, table.md and a score chart per run into --out.
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
from qc import charts, features, sim  # noqa: E402
from qc import config as cfgmod  # noqa: E402
from qc import threshold as thmod  # noqa: E402
from qc.model import NormalModel  # noqa: E402

LOC_TOL_PX = 12          # hottest spot within this many frame px of the defect counts as "points at it"


def heat_caption(pic, shape, face, defect, score, thr):
    """White strip under a heat-map example: what it shows and where it comes from."""
    lines = ["%s/%s with a simulated %s: left as placed, right straightened with the heat map" % (shape, face, defect),
             "(colour from half the threshold; red = at or above it). Score %.1f, threshold %.1f." % (score, thr),
             "Data: qc/sim.py, pretrained ResNet-18, local bank."]
    bar = np.full((22 * len(lines) + 8, pic.shape[1], 3), 255, np.uint8)
    for k, t in enumerate(lines):
        cv2.putText(bar, t, (6, 18 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
    return np.vstack([pic, bar])


def frames(shape, face, seeds, defect=None, severity=1.0, drift=None):
    return [sim.render(shape, face, defect, seed=s, severity=severity, drift=drift) for s in seeds]


def score(m, ref, items, bb):
    ims, ms, keep = [], [], []
    for it in items:
        loc = ref.part_mask(it["frame"])
        if loc is None:
            keep.append(None)
            continue
        s = m.straighten(it["frame"], loc)
        ims.append(s["image"])
        ms.append(s["mask"])
        keep.append(s)
    sc, maps = m.score_canonical(ims, ms, bb) if ims else (np.array([]), [])
    out, k = [], 0
    for it, s in zip(items, keep):
        if s is None:
            out.append(dict(score=float("inf"), hit=None, no_part=True))     # no part found -> rejected
            continue
        heat = m.heatmap(maps[k])
        hit = None
        if it["defect"]:
            yx = np.unravel_index(int(np.argmax(heat)), heat.shape)
            fx, fy = cv2.invertAffineTransform(s["M"]) @ np.array([yx[1], yx[0], 1.0])
            dm = cv2.dilate((it["defect_mask"] > 0.2).astype(np.uint8), np.ones((2 * LOC_TOL_PX + 1,) * 2, np.uint8))
            hit = bool(dm[int(np.clip(round(fy), 0, sim.H - 1)), int(np.clip(round(fx), 0, sim.W - 1))])
        out.append(dict(score=float(sc[k]), hit=hit, no_part=False, heat=heat, canon=s["image"]))
        k += 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "results", "synthetic"))
    ap.add_argument("--backbone", default="onnx")
    ap.add_argument("--bank", default="local")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--shapes", default=",".join(sim.SHAPES))
    ap.add_argument("--faces", default=",".join(sim.FACES))
    ap.add_argument("--examples", default=None, help="folder for heat-map example pictures")
    a = ap.parse_args(argv)
    n_train, n_hold, n_test, n_def = (20, 10, 15, 4) if a.quick else (60, 20, 40, 10)
    cfg = cfgmod.load(os.path.join(ROOT, "nonexistent.json"))
    cfg.update(backbone=a.backbone, bank=a.bank)
    bb = features.make(cfg, ROOT)
    tag = "%s-%s" % (bb.name, a.bank)
    out = os.path.join(a.out, tag)
    os.makedirs(out, exist_ok=True)
    ref = bgmod.EmptyReference.from_frames([sim.empty_frame(seed=100 + i) for i in range(10)])
    res = dict(backbone=bb.name, bank=a.bank, n_train=n_train, n_hold=n_hold, n_test=n_test, n_def=n_def,
               cells={}, timing={})
    groups_all = []
    for shape in a.shapes.split(","):
        for face in a.faces.split(","):
            key = "%s/%s" % (shape, face)
            base = (sim.SHAPES.index(shape) * 2 + sim.FACES.index(face)) * 100000
            t0 = time.time()
            m = NormalModel(shape, face, cfg)
            train = [f["frame"] for f in frames(shape, face, range(base, base + n_train))]
            m.fit(m.prepare(train, ref), bb)
            t_train = time.time() - t0
            hold = score(m, ref, frames(shape, face, range(base + 10000, base + 10000 + n_hold)), bb)
            cal_bad = []
            for j, d in enumerate(sim.DEFECTS):
                cal_bad += score(m, ref, frames(shape, face, range(base + 20000 + 100 * j, base + 20000 + 100 * j + 2), d), bb)
            th = thmod.from_scores([h["score"] for h in hold], [b["score"] for b in cal_bad if np.isfinite(b["score"])],
                                   cfg["target_false_reject"], cfg["cost_bad_accepted"], cfg["cost_good_rejected"])
            thr = th["threshold"]
            t1 = time.time()
            good = score(m, ref, frames(shape, face, range(base + 30000, base + 30000 + n_test)), bb)
            t_score = (time.time() - t1) / n_test
            cell = dict(threshold=thr, threshold_method=th["method"], order=m.order,
                        false_reject=sum(g["score"] > thr for g in good), n_good=len(good),
                        good_scores=[g["score"] for g in good], train_s=t_train, score_s=t_score, defects={})
            groups = [("good (test)", [g["score"] for g in good], charts.BLUE)]
            for sev_name, sev in (("normal", 1.0), ("subtle", 0.5)):
                for j, d in enumerate(sim.DEFECTS):
                    s0 = base + 40000 + (0 if sev == 1.0 else 50000) + 100 * j
                    items = frames(shape, face, range(s0, s0 + n_def), d, sev)
                    r = score(m, ref, items, bb)
                    caught = sum(x["score"] > thr for x in r)
                    hits = [x["hit"] for x in r if x["hit"] is not None and x["score"] > thr]
                    cell["defects"]["%s/%s" % (d, sev_name)] = dict(
                        caught=int(caught), n=len(r), located=int(sum(hits)), scores=[x["score"] for x in r],
                        no_part=int(sum(x["no_part"] for x in r)))
                    if sev_name == "normal":
                        groups.append((d, [min(x["score"], 4 * thr) for x in r], charts.ORANGE))
                    if a.examples and sev_name == "normal" and shape == "square" and face == "top" or \
                            (a.examples and sev_name == "normal" and d in ("scratch", "stain") and face == "top"):
                        x = r[0]
                        if not x["no_part"]:
                            from qc.station import overlay
                            os.makedirs(a.examples, exist_ok=True)
                            p = os.path.join(a.examples, "heatmap_%s_%s_%s.jpg" % (shape, face, d))
                            pic = np.hstack([items[0]["frame"][96:384, 176:464],
                                             cv2.resize(overlay(x["canon"], x["heat"], thr), (288, 288))])
                            cv2.imwrite(p, heat_caption(pic, shape, face, d, x["score"], thr))
            res["cells"][key] = cell
            groups_all.append((key, groups, thr))
            print("%-15s thr %.2f  false rejects %d/%d  %s  (train %.0fs, %.0f ms/picture)" % (
                key, thr, cell["false_reject"], len(good),
                " ".join("%s %d/%d" % (k.split("/")[0][:5] + ("*" if k.endswith("subtle") else ""), v["caught"], v["n"])
                         for k, v in cell["defects"].items()), t_train, 1000 * t_score), flush=True)
    # drift
    dr = dict(empty_normal=0, empty_n=30, parts_normal=0, parts_n=0, kinds={})
    for i in range(30):
        dr["empty_normal"] += ref.drift(sim.empty_frame(seed=5000 + i))["drift"]
    for key, cell in res["cells"].items():
        pass
    for shape in sim.SHAPES:
        for s in range(5):
            f = sim.render(shape, "top", seed=6000 + s)["frame"]
            loc = ref.part_mask(f)
            dr["parts_normal"] += ref.drift(f, exclude=loc.filled)["drift"]
            dr["parts_n"] += 1
    for k in sim.DRIFTS:
        e = sum(ref.drift(sim.empty_frame(seed=7000 + i, drift=k))["drift"] for i in range(10))
        p = 0
        for i in range(10):
            f = sim.render(sim.SHAPES[i % 3], "top", seed=8000 + i, drift=k)["frame"]
            loc = ref.part_mask(f)
            p += ref.drift(f, exclude=loc.filled if loc is not None else None)["drift"]
        dr["kinds"][k] = dict(empty_warned=int(e), empty_n=10, parts_warned=int(p), parts_n=10)
    res["drift"] = dr
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    write_table(res, os.path.join(out, "table.md"))
    for key, groups, thr in groups_all:
        if key in ("square/top", "disc/bottom", "hexagon/top"):
            charts.score_strip(
                groups, thr, "Anomaly scores, %s: good test parts vs each defect type (%s, %s bank)" % (key, bb.name, a.bank),
                "computer-generated pictures (qc/sim.py, fixed seeds), %d good test parts, %d per defect type; "
                "scores above 4x the threshold drawn at 4x" % (len(groups[0][1]), len(groups[1][1])),
                "Every dot right of the line is rejected: good parts (blue) should sit left of it, damaged ones right.",
                os.path.join(out, "scores_%s.png" % key.replace("/", "_")))
    print(open(os.path.join(out, "table.md")).read())
    return 0


def write_table(res, path):
    keys = list(res["cells"].keys())
    L = ["# Synthetic results: %s features, %s memory bank" % (res["backbone"], res["bank"]), "",
         "Computer-generated pictures (qc/sim.py, fixed seeds). Per shape/face: %d good pictures to train, "
         "%d good + 10 damaged (2 per type) to set the threshold, then %d good test parts and %d per defect "
         "type at each severity. Cells: caught / tested (heat map's hottest spot on the defect)." % (
             res["n_train"], res["n_hold"], res["n_test"], res["n_def"]), ""]
    defs = list(next(iter(res["cells"].values()))["defects"].keys())
    for sev in ("normal", "subtle"):
        L.append("**%s severity**%s" % (sev.capitalize(), "" if sev == "normal" else " (half the normal size and contrast)"))
        L.append("")
        cols = [d for d in defs if d.endswith(sev)]
        L.append("| shape / face | " + " | ".join(c.split("/")[0] for c in cols) + " | false rejects (good) |")
        L.append("|---" * (len(cols) + 2) + "|")
        for k in keys:
            c = res["cells"][k]
            L.append("| %s | " % k + " | ".join("%d/%d (%d)" % (c["defects"][d]["caught"], c["defects"][d]["n"],
                                                               c["defects"][d]["located"]) for d in cols) +
                     " | %d/%d |" % (c["false_reject"], c["n_good"]))
        tot = []
        for d in cols:
            tot.append("**%d/%d**" % (sum(res["cells"][k]["defects"][d]["caught"] for k in keys),
                                      sum(res["cells"][k]["defects"][d]["n"] for k in keys)))
        L.append("| **all shapes** | " + " | ".join(tot) + " | **%d/%d** |" % (
            sum(res["cells"][k]["false_reject"] for k in keys), sum(res["cells"][k]["n_good"] for k in keys)))
        L.append("")
    d = res["drift"]
    L += ["**Background drift check** (warnings / pictures)", "",
          "| case | empty spot | with a part |", "|---|---|---|",
          "| no drift (false alarms) | %d/%d | %d/%d |" % (d["empty_normal"], d["empty_n"], d["parts_normal"], d["parts_n"])]
    for k, v in d["kinds"].items():
        L.append("| %s | %d/%d | %d/%d |" % (k, v["empty_warned"], v["empty_n"], v["parts_warned"], v["parts_n"]))
    L += ["", "Timing on the build machine (2 CPU cores): training %.0f-%.0f s per shape/face, %.0f-%.0f ms per picture "
          "(straighten + score)." % (min(c["train_s"] for c in res["cells"].values()), max(c["train_s"] for c in res["cells"].values()),
                                      1000 * min(c["score_s"] for c in res["cells"].values()), 1000 * max(c["score_s"] for c in res["cells"].values()))]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
