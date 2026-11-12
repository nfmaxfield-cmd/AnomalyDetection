"""Background masking from the empty reference, and the drift warning.

Changelog
  0.1.0  2026-11-12  first version.
"""
import cv2
import numpy as np
import pytest

from conftest import iou
from qc import background as bgmod
from qc import sim


@pytest.mark.parametrize("shape", sim.SHAPES)
def test_mask_matches_the_true_outline_from_the_first_picture(empty_ref, shape):
    r = sim.render(shape, "top", seed=3)
    loc = empty_ref.part_mask(r["frame"])
    assert loc is not None
    truth = (r["part_mask"] > 0.5).astype(np.uint8)
    # holes (the backdrop showing through) are not part; the filled outline covers them
    assert iou(loc.mask, truth) > 0.95
    filled_truth = np.zeros_like(truth)
    cs, _ = cv2.findContours(truth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(filled_truth, cs, -1, 1, -1)
    assert iou(loc.filled, filled_truth) > 0.97
    assert abs(loc.centre[0] - r["centre"][0]) < 1.5 and abs(loc.centre[1] - r["centre"][1]) < 1.5


def test_empty_spot_has_no_part(empty_ref):
    assert empty_ref.part_mask(sim.empty_frame(seed=999)) is None


def test_limits_come_from_the_empty_frames(empty_ref):
    # noise level from leave-one-out differences: above the sensor noise, far below a part's contrast
    assert 3 < empty_ref.noise_level < 40
    assert empty_ref.n_frames == 10


def test_single_empty_frame_still_works():
    ref = bgmod.EmptyReference.from_frames([sim.empty_frame(seed=1)])
    assert ref.n_frames == 1
    assert ref.part_mask(sim.render("disc", "top", seed=2)["frame"]) is not None


def test_save_and_load(empty_ref, tmp_path):
    p = empty_ref.save(str(tmp_path / "e.npz"))
    r2 = bgmod.EmptyReference.load(p)
    assert abs(r2.noise_level - empty_ref.noise_level) < 1e-6
    assert np.abs(r2.mean - empty_ref.mean).max() < 0.1


def test_no_drift_warning_on_normal_frames(empty_ref):
    for i in range(10):
        assert not empty_ref.drift(sim.empty_frame(seed=300 + i))["drift"]
    for i, shape in enumerate(sim.SHAPES):
        f = sim.render(shape, "top", seed=400 + i)["frame"]
        loc = empty_ref.part_mask(f)
        assert not empty_ref.drift(f, exclude=loc.filled)["drift"]


@pytest.mark.parametrize("kind", sim.DRIFTS)
def test_drift_warning_on_empty_and_part_frames(empty_ref, kind):
    d = empty_ref.drift(sim.empty_frame(seed=500, drift=kind))
    assert d["drift"], d
    assert d["reasons"]
    f = sim.render("square", "top", seed=501, drift=kind)["frame"]
    loc = empty_ref.part_mask(f)
    assert loc is not None
    assert empty_ref.drift(f, exclude=loc.filled)["drift"]


def test_stray_object_is_located(empty_ref):
    d = empty_ref.drift(sim.empty_frame(seed=77, drift="stray_object"))
    x, y = d["where"]
    assert 60 < x < 240 and 360 < y < 470            # where sim.py puts the screw
