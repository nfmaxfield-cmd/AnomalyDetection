"""Straightening with symmetry: the outline's symmetry, the angle, and markings that break the symmetry.

Changelog
  0.1.0  2026-11-12  first version.
"""
import numpy as np
import pytest

from qc import sim
from qc import straighten as st

ORDERS = {"disc": 0, "square": 4, "hexagon": 6}


@pytest.mark.parametrize("shape", sim.SHAPES)
def test_symmetry_detected_from_the_outline(empty_ref, shape):
    for seed in range(3):
        r = sim.render(shape, "top", seed=seed)
        loc = empty_ref.part_mask(r["frame"])
        assert st.outline_symmetry(loc.contour, loc.centre)["order"] == ORDERS[shape]


def test_symmetry_can_be_configured(empty_ref):
    r = sim.render("square", "top", seed=1)
    loc = empty_ref.part_mask(r["frame"])
    assert st.outline_symmetry(loc.contour, loc.centre, order=2)["order"] == 2


def test_symmetry_still_found_on_a_damaged_part(empty_ref):
    r = sim.render("hexagon", "top", defect="missing_corner", seed=5)
    loc = empty_ref.part_mask(r["frame"])
    assert st.outline_symmetry(loc.contour, loc.centre)["order"] == 6


def _canonical_angles(shape, face, empty_ref, n=8):
    ref = rm = None
    out = []
    for s in range(n):
        r = sim.render(shape, face, seed=50 + s)
        loc = empty_ref.part_mask(r["frame"])
        o = st.straighten(r["frame"], loc, ref, 300, 224, None, ref_mask=rm)
        if ref is None:
            ref, rm = o["image"], o["mask"]
        out.append((o["angle_deg"] + r["angle"]) % 360)     # same for every part when straightening works
    return np.array(out)


def _spread(a, period):
    d = (a - a[0] + period / 2) % period - period / 2
    return float(np.abs(d).max())


@pytest.mark.parametrize("shape,face,period", [("square", "top", 90), ("hexagon", "top", 60),
                                               ("hexagon", "bottom", 60)])
def test_straightened_to_the_same_angle_up_to_symmetry(empty_ref, shape, face, period):
    assert _spread(_canonical_angles(shape, face, empty_ref), period) < 1.0


@pytest.mark.parametrize("shape", ["square", "disc"])
def test_a_stamp_on_the_face_decides_the_full_turn(empty_ref, shape):
    # square bottom: outline 4-fold, the "A7" stamp makes it 1-fold; disc bottom: round, the stamp decides
    assert _spread(_canonical_angles(shape, "bottom", empty_ref), 360) < 3.0


def test_fine_align_recovers_a_small_turn(empty_ref):
    r = sim.render("square", "top", seed=9, angle=10.0)
    loc = empty_ref.part_mask(r["frame"])
    sym = st.outline_symmetry(loc.contour, loc.centre)
    ref, _, _ = st.crop(r["frame"], None, loc.centre, sym["angle_deg"], 300, 224)
    off, _, _ = st.crop(r["frame"], None, loc.centre, sym["angle_deg"] + 3.0, 300, 224)
    da, dx, dy, c = st.fine_align(off, ref, 0.0, 5.0, 0.5)
    assert abs(da + 3.0) < 0.4 and c > 0.9


def test_content_symmetry_reports_the_stamp(empty_ref):
    out = []
    for face in ("top", "bottom"):
        r = sim.render("square", face, seed=11)
        loc = empty_ref.part_mask(r["frame"])
        o = st.straighten(r["frame"], loc, None, 300, 224, None)
        out.append(min(st.content_symmetry(o["image"], 4, o["mask"])))
    assert out[0] > out[1] + 0.1           # top face is symmetric, the stamped bottom is not
