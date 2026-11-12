"""The command-line flow with --simulate in a fresh folder: collect-empty, collect-good, collect-damaged,
train, calibrate-threshold, learning-curve, inspect, run. Hand-made features keep it quick.

Changelog
  0.1.0  2026-11-12  first version.
"""
import glob
import json
import os

import pytest

from qc import cli
from qc import train as trainmod


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("qcwork"))
    cfgp = os.path.join(root, "qc_config.json")
    with open(cfgp, "w") as f:
        json.dump(dict(backbone="handcrafted", burst_frames=4), f)
    base = ["--config", cfgp, "--root", root]
    assert cli.main(base + ["collect-empty", "--simulate"]) == 0
    assert cli.main(base + ["collect-good", "--simulate", "--shape", "square", "--face", "top", "--n", "14"]) == 0
    assert cli.main(base + ["collect-damaged", "--simulate", "--shape", "square", "--face", "top", "--n", "5"]) == 0
    assert cli.main(base + ["train", "--shape", "square", "--face", "top"]) == 0
    return root, base


def test_files_land_where_the_docs_say(work):
    root, _ = work
    assert os.path.exists(os.path.join(root, "models", "empty.npz"))
    assert len(glob.glob(os.path.join(root, "data", "good", "square", "top", "*.png"))) == 14
    assert len(glob.glob(os.path.join(root, "data", "damaged", "square", "top", "*.png"))) == 5
    th = json.load(open(os.path.join(root, "models", "square", "top", "threshold.json")))
    assert th["n_good"] >= 3 and th["n_bad"] == 5 and th["threshold"] > 0


def test_calibrate_and_inspect(work, capsys):
    root, base = work
    assert cli.main(base + ["calibrate-threshold", "--shape", "square", "--face", "top"]) == 0
    img = sorted(glob.glob(os.path.join(root, "data", "damaged", "square", "top", "*missing_corner*.png")))[0]
    capsys.readouterr()
    assert cli.main(base + ["inspect", img, "--shape", "square", "--face", "top"]) == 0
    msg = json.loads(capsys.readouterr().out)
    assert msg["decision"] == "reject" and msg["score"] > msg["threshold"]


def test_simulated_run_with_fake_arm(work, capsys):
    root, base = work
    assert cli.main(base + ["run", "--simulate", "--parts", "2", "--defect-rate", "0"]) == 0
    out = capsys.readouterr().out
    assert "Simulated run finished" in out
    tally = json.loads(out.split("Simulated run finished: ")[1].splitlines()[0])
    assert sum(tally.values()) == 2


def test_learning_curve_command(work):
    root, base = work
    assert cli.main(base + ["learning-curve", "--shape", "square", "--face", "top", "--sizes", "3,6",
                            "--repeats", "2"]) == 0
    assert os.path.exists(os.path.join(root, "models", "square", "top", "learning_curve.png"))
    assert os.path.exists(os.path.join(root, "models", "square", "top", "learning_curve.csv"))


def test_suggest_n_uses_the_spread_between_draws():
    rows = []
    for n, vals in ((5, [0.5, 0.6, 0.7]), (10, [0.85, 0.9, 0.95]), (20, [0.95, 0.97, 0.98]), (40, [0.96, 0.97, 0.98])):
        rows += [dict(n=n, caught_at_threshold=v) for v in vals]
    s = trainmod.suggest_n(rows)
    assert s["n"] == 20            # noise = median draw spread 0.065; from 20 on every median is within it of 0.97
    climbing = [dict(n=n, caught_at_threshold=v) for n, v in ((5, 0.2), (10, 0.5), (20, 0.8))]
    assert trainmod.suggest_n(climbing)["n"] is None
    dip = rows + [dict(n=80, caught_at_threshold=v) for v in (0.6, 0.6, 0.61)]      # a later dip: not settled at 10
    assert trainmod.suggest_n(dip)["n"] is None


def test_paint_damage_changes_only_the_part(empty_ref):
    import numpy as np
    from qc import sim
    f = sim.render("disc", "top", seed=3)["frame"]
    loc = empty_ref.part_mask(f)
    rng = np.random.default_rng(0)
    for _ in range(6):
        g, kind = trainmod.paint_damage(f, loc.mask, rng)
        changed = np.abs(g.astype(int) - f.astype(int)).max(2) > 0
        assert changed.any()
        grown = __import__("cv2").dilate(loc.filled, np.ones((31, 31), np.uint8))
        assert not (changed & (grown == 0)).any(), kind
