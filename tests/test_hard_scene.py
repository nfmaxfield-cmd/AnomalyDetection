"""The harder simulated scene (audit H3) with the pretrained ResNet-18: it renders what it claims, the
station still separates good from clearly damaged parts there, and its own checks do not turn good parts
into "unsure" (each unsure good part is a good part scrapped).

Changelog
  0.1.0  2026-11-12  first version (audit fixes).
"""
import os

import numpy as np
import pytest

from conftest import BACKBONE_ONNX, scores
from qc import background as bgmod
from qc import capture, features, sim
from qc import threshold as thmod
from qc.model import NormalModel
from qc.station import picture_checks

needs_backbone = pytest.mark.skipif(not os.path.exists(BACKBONE_ONNX), reason="pretrained backbone not downloaded "
                                    "(python tools/get_backbone.py)")


def test_easy_scene_is_kept_behind_the_switch():
    a = sim.render("square", "top", seed=3, scene="easy")["frame"]
    b = sim.render("square", "top", seed=3, scene="hard")["frame"]
    assert a.shape == b.shape and np.abs(a.astype(int) - b.astype(int)).mean() > 2
    assert sim.render("square", "top", seed=3, scene="easy")["frame"].tobytes() == a.tobytes()   # repeatable
    with pytest.raises(ValueError):
        sim.render("square", "top", seed=3, scene="medium")


def test_hard_scene_has_its_factors():
    e = [sim.empty_frame(seed=i, scene="hard").astype(float) for i in range(4)]
    left, right = np.mean([x[200:280, 120:200].mean() for x in e]), np.mean([x[200:280, 440:520].mean() for x in e])
    assert right > left                                              # light brighter on one side, fixed
    noise_h = np.std(e[0][200:280, 300:340] - e[1][200:280, 300:340])
    ee = [sim.empty_frame(seed=i, scene="easy").astype(float) for i in range(2)]
    assert noise_h > np.std(ee[0][200:280, 300:340] - ee[1][200:280, 300:340])       # noisier camera
    cs = [sim.render("square", "top", seed=s, scene="hard")["centre"] for s in range(40)]
    assert np.std([c[0] for c in cs]) > 8                            # placed less precisely (sd 15 px)
    assert dict(sim.SEVERITIES)["faint"] == 0.25


@needs_backbone
def test_hard_scene_good_parts_pass_and_clear_damage_is_caught(cfg):
    bb = features.OnnxBackbone(BACKBONE_ONNX)
    ref = bgmod.EmptyReference.from_frames([sim.empty_frame(seed=100 + i, scene="hard") for i in range(10)])
    c = dict(cfg, backbone="onnx")
    m = NormalModel("square", "top", c)
    m.fit(m.prepare([sim.render("square", "top", seed=50000 + i, scene="hard")["frame"] for i in range(30)], ref), bb)
    hold = scores(m, ref, [sim.render("square", "top", seed=51000 + i, scene="hard")["frame"] for i in range(20)], bb)
    thr = thmod.from_scores(hold)["threshold"]
    good_frames = [sim.render("square", "top", seed=52000 + i, scene="hard")["frame"] for i in range(15)]
    good = scores(m, ref, good_frames, bb)
    bad = scores(m, ref, [sim.render("square", "top", d, seed=53000 + k, scene="hard")["frame"]
                          for k, d in enumerate(sim.DEFECTS * 2)], bb)
    assert (good > thr).sum() <= 1
    assert (bad > thr).sum() >= 8
    unsure = 0
    for f in good_frames:                                            # the station's checks on good parts
        loc = ref.part_mask(f)
        s = m.straighten(f, loc)
        unsure += bool(picture_checks(m, f, loc, s, capture.sharpness(f, loc.filled))) or ref.drift(f, exclude=loc.filled)["drift"]
    assert unsure <= 1


def test_hard_scene_bursts_are_still_and_slides_are_not():
    ref = bgmod.EmptyReference.from_frames([sim.empty_frame(seed=100 + i, scene="hard") for i in range(10)])
    for shape in sim.SHAPES:
        fr, settled = sim.burst(shape, "bottom", seed=3, scene="hard")
        loc = ref.part_mask(fr[-1])
        i, _ = capture.pick_sharpest(fr, loc.filled)
        assert capture.is_still(fr, i, ref.still_noise, loc.filled)[0]
    slide = [sim.render("square", "top", seed=7, angle=30, centre=(320 + k, 240), motion_px=1.0, noise_seed=k,
                        scene="hard")["frame"] for k in range(8)]
    loc = ref.part_mask(slide[-1])
    i, _ = capture.pick_sharpest(slide, loc.filled)
    assert not capture.is_still(slide, i, ref.still_noise, loc.filled)[0]
