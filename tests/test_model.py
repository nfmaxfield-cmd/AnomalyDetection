"""Model training and scoring: damaged parts score higher than good ones; save/load; both banks;
the pretrained ResNet-18 when its file is present.

Changelog
  0.1.0  2026-11-12  first version.
  0.2.0  2026-11-12  audit fixes: cell-first memory gives the first build's scores (M4); shifted copies make
                     the memory tolerate leftover shift (M1); leave-one-out scores find a damaged picture in the
                     good folder (M5); light flattening removes a straight slope (H3 option).
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


def _old_local_maps(m, z):
    """The first build's comparison (memory laid out picture-first, einsum per neighbour)."""
    r = m.window
    b = m.bank.astype(np.float32)
    pad = np.pad(b, ((0, 0), (r, r), (r, r), (0, 0)), mode="edge")
    pn = (pad ** 2).sum(-1)
    n, h, w, d = z.shape
    out = np.zeros((n, h, w), np.float32)
    for i in range(n):
        zi, zn = z[i], (z[i] ** 2).sum(-1)
        best = np.full((h, w), np.inf, np.float32)
        for dy in range(2 * r + 1):
            for dx in range(2 * r + 1):
                dot = np.einsum("mhwd,hwd->mhw", pad[:, dy:dy + h, dx:dx + w], zi, optimize=True)
                best = np.minimum(best, (pn[:, dy:dy + h, dx:dx + w] - 2 * dot).min(0) + zn)
        out[i] = np.sqrt(np.maximum(best, 0))
    return out


def test_m4_cell_first_memory_gives_the_same_maps(square_model, empty_ref, hand):
    f = sim.render("square", "top", "stain", seed=38000)["frame"]
    s = square_model.straighten(f, empty_ref.part_mask(f))
    z = square_model._embed(hand([s["image"]]))
    assert np.allclose(square_model._local_maps(z), _old_local_maps(square_model, z), atol=1e-3)


def test_m1_shift_copies_tolerate_leftover_shift(cfg, empty_ref, hand):
    import cv2
    frames = [sim.render("square", "bottom", seed=38100 + i)["frame"] for i in range(12)]
    test = [sim.render("square", "bottom", seed=38200 + i)["frame"] for i in range(5)]
    out = {}
    for shifts in ([], [3]):
        m = NormalModel("square", "bottom", dict(cfg, train_shifts_px=shifts))
        m.fit(m.prepare(frames, empty_ref), hand)
        assert m.copies_per_picture() == 3 + 4 * len(shifts) and len(m.bank) == 12 * m.copies_per_picture()
        ims, ms = [], []
        for f in test:
            st = m.straighten(f, empty_ref.part_mask(f))
            M = np.float32([[1, 0, 3], [0, 1, 1]])          # 3 px of leftover shift after straightening
            ims.append(cv2.warpAffine(st["image"], M, (224, 224), borderMode=cv2.BORDER_REPLICATE))
            ms.append(cv2.warpAffine(st["mask"], M, (224, 224), flags=cv2.INTER_NEAREST))
        out[len(shifts)] = float(np.median(m.score_canonical(ims, ms, hand)[0]))
    assert out[1] < out[0]


def test_m5_leave_one_out_finds_a_damaged_picture_in_the_good_folder(cfg, empty_ref, hand):
    frames = [sim.render("square", "top", seed=38300 + i)["frame"] for i in range(12)]
    frames[5] = sim.render("square", "top", "stain", seed=38399)["frame"]
    m = NormalModel("square", "top", cfg)
    canon = m.prepare(frames, empty_ref)
    m.fit(canon, hand)
    loo = m.self_scores(canon, hand)
    assert int(np.argmax(loo)) == 5 and loo[5] > 1.5 * np.median(loo)


def test_flatten_light_removes_a_straight_slope():
    from qc.model import flatten_light
    img = np.full((224, 224, 3), 150, np.uint8)
    mask = np.zeros((224, 224), np.uint8)
    mask[40:184, 40:184] = 1
    xx = np.arange(224, dtype=np.float32)[None, :, None]
    sloped = np.clip(img * (1 + 0.1 * (xx - 112) / 112), 0, 255).astype(np.uint8)
    out = flatten_light(sloped, mask)
    inside = out[40:184, 40:184, 0].astype(float)
    assert inside[:, :10].mean() == pytest.approx(inside[:, -10:].mean(), abs=2)
    assert abs(sloped[40:184, 40:50].mean() - sloped[40:184, 174:184].mean()) > 15


def test_training_statistics_are_stored(square_model):
    for k in ("train_area", "train_sharpness", "align_corr", "train_contrast", "frame_size"):
        assert k in square_model.meta and square_model.meta[k]
    assert square_model.meta["backbone_sha256"] == "handcrafted-v1"
