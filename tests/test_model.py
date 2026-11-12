"""Model training and scoring: damaged parts score higher than good ones; save/load; both banks;
the pretrained ResNet-18 when its file is present.

Changelog
  0.1.0  2026-11-12  first version.
"""
import os

import numpy as np
import pytest

from conftest import BACKBONE_ONNX, scores, trained
from qc import features, sim
from qc.model import NormalModel, coreset


def test_train_and_score_good_parts(square_model, empty_ref, hand):
    good = scores(square_model, empty_ref, [sim.render("square", "top", seed=31000 + i)["frame"] for i in range(6)], hand)
    assert np.isfinite(good).all() and (good > 0).all()
    assert square_model.order == 4
    assert square_model.meta["n_train"] == 16


@pytest.mark.parametrize("defect", ["chip", "missing_corner", "stain"])
def test_damaged_parts_score_higher_than_good_ones(square_model, empty_ref, hand, defect):
    good = scores(square_model, empty_ref, [sim.render("square", "top", seed=32000 + i)["frame"] for i in range(8)], hand)
    bad = scores(square_model, empty_ref, [sim.render("square", "top", defect, seed=33000 + i)["frame"] for i in range(4)], hand)
    assert bad.min() > good.max(), (good, bad)


def test_heat_map_points_at_the_defect(square_model, empty_ref, hand):
    import cv2
    r = sim.render("square", "top", "missing_corner", seed=34001)
    loc = empty_ref.part_mask(r["frame"])
    s = square_model.straighten(r["frame"], loc)
    sc, maps = square_model.score_canonical([s["image"]], [s["mask"]], hand)
    heat = square_model.heatmap(maps[0])
    y, x = np.unravel_index(np.argmax(heat), heat.shape)
    fx, fy = cv2.invertAffineTransform(s["M"]) @ np.array([x, y, 1.0])
    dm = cv2.dilate((r["defect_mask"] > 0.2).astype(np.uint8), np.ones((25, 25), np.uint8))
    assert dm[int(round(fy)), int(round(fx))]


def test_save_and_load_give_the_same_scores(square_model, empty_ref, hand, tmp_path):
    frames = [sim.render("square", "top", seed=35000 + i)["frame"] for i in range(3)]
    a = scores(square_model, empty_ref, frames, hand)
    square_model.save(str(tmp_path / "m"))
    m2 = NormalModel.load(str(tmp_path / "m"))
    b = scores(m2, empty_ref, frames, hand)
    assert np.allclose(a, b, rtol=1e-2)


def test_global_bank_also_separates(cfg, empty_ref, hand):
    m = trained("hexagon", "top", cfg, empty_ref, hand, n=12, bank="global")
    good = scores(m, empty_ref, [sim.render("hexagon", "top", seed=36000 + i)["frame"] for i in range(5)], hand)
    bad = scores(m, empty_ref, [sim.render("hexagon", "top", "chip", seed=36100 + i)["frame"] for i in range(3)], hand)
    assert bad.min() > good.max()


def test_coreset_picks_spread_out_points():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 0.01, (200, 4)), rng.normal(5, 0.01, (3, 4))]).astype(np.float32)
    idx = coreset(x, 2)
    assert {i >= 200 for i in idx} == {True, False}      # one from each cluster


def test_auto_backbone_falls_back_with_a_warning(tmp_path):
    bb = features.make({"backbone": "auto", "backbone_onnx": str(tmp_path / "missing.onnx")}, str(tmp_path))
    assert bb.name in ("handcrafted",) or bb.name.startswith("torch")
    if bb.name == "handcrafted":
        assert "get_backbone" in bb.warning


@pytest.mark.skipif(not os.path.exists(BACKBONE_ONNX), reason="pretrained backbone not downloaded "
                    "(python tools/get_backbone.py)")
def test_pretrained_resnet18_catches_a_scratch(cfg, empty_ref):
    bb = features.OnnxBackbone(BACKBONE_ONNX)
    f = bb([sim.render("disc", "top", seed=1)["frame"][128:352, 208:432]])
    assert f.shape == (1, 384, 28, 28)
    m = trained("square", "top", dict(cfg, backbone="onnx"), empty_ref, bb, n=12)
    good = scores(m, empty_ref, [sim.render("square", "top", seed=37000 + i)["frame"] for i in range(6)], bb)
    bad = scores(m, empty_ref, [sim.render("square", "top", "scratch", seed=37100 + i)["frame"] for i in range(3)], bb)
    assert bad.min() > good.max(), (good, bad)
