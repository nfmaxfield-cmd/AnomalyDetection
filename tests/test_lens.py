"""Checkerboard calibration card (Nick, 12 Nov): the printable card, the lens fit on simulated distorted
pictures, its data-derived accept limit and checks, and undistortion + mm in the station.

Changelog
  0.1.0  2026-11-12  first version.
"""
import json
import os

import cv2
import numpy as np
import pytest

from qc import capture, lens, sim
from qc.station import CorrectedCamera, Station

INNER = (8, 5)


@pytest.fixture(scope="module")
def cal():
    views = [lens.sim_view(100 + k, INNER)[0] for k in range(12)]
    flat = lens.sim_view(999, INNER, flat=True)[0]
    return lens.calibrate(views, INNER, 20.0, flat=flat)


def test_card_files_and_true_size(tmp_path):
    png, pdf = lens.make_card(str(tmp_path / "card"), (9, 6), 20.0)
    img = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
    assert os.path.getsize(pdf) > 1000 and open(pdf, "rb").read(4) == b"%PDF"
    row = img[int(8 * 300 / 25.4) + 5]                     # a row through the first squares
    edges = np.nonzero(np.diff((row < 128).astype(int)))[0]
    assert np.median(np.diff(edges)) == pytest.approx(20 * 300 / 25.4, abs=1.5)     # 20 mm at 300 dpi
    assert lens.find_corners(cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4)), INNER) is not None
    with pytest.raises(ValueError):
        lens.make_card(str(tmp_path / "big"), (9, 6), 30.0)


def test_calibration_recovers_the_simulated_lens(cal):
    assert cal["ok"], cal["problems"]
    K = np.array(cal["camera_matrix"])
    assert K[0, 0] == pytest.approx(lens.SIM_K[0, 0], rel=0.02) and K[0, 2] == pytest.approx(lens.SIM_K[0, 2], abs=4)
    assert cal["mm_per_px"] == pytest.approx(400.0 / 560.0, rel=0.01)
    assert cal["rms"] <= cal["limit"] and cal["limit"] == pytest.approx(lens.JITTER_FACTOR * max(cal["jitter"], lens.JITTER_FLOOR))


def test_undistortion_straightens_lines(cal):
    f = lens.sim_view(999, INNER, flat=True)[0][0]
    lc = lens.LensCorrection(cal)

    def bow(c):
        g = c.reshape(INNER[1], INNER[0], 2)[0]
        p = np.polyfit(g[:, 0], g[:, 1], 1)
        return np.abs(g[:, 1] - np.polyval(p, g[:, 0])).max()
    assert bow(lens.find_corners(lc.undistort(f), INNER)) < 0.3 * bow(lens.find_corners(f, INNER))


def test_checks_refuse_too_few_and_too_alike_views():
    few = lens.calibrate([lens.sim_view(200 + k, INNER)[0] for k in range(5)], INNER, 20.0)
    assert not few["ok"] and "found in only" in few["problems"][0]
    same = lens.calibrate([lens.sim_view(300 + k, INNER, tilt_deg=12.0)[0] for k in range(9)], INNER, 20.0)
    assert not same["ok"] and any("tilted too alike" in p for p in same["problems"])


def test_a_bent_card_view_is_dropped():
    views = [lens.sim_view(400 + k, INNER)[0] for k in range(10)]
    yy, xx = np.mgrid[0:480, 0:640].astype(np.float32)
    bend = (xx + 3.0 * np.sin(yy / 40.0)).astype(np.float32)              # a wavy, bent card
    views[4] = [cv2.remap(f, bend, yy, cv2.INTER_LINEAR) for f in views[4]]
    r = lens.calibrate(views, INNER, 20.0)
    assert 4 in r["dropped"]


def test_station_undistorts_and_reports_mm(tmp_path, cal):
    root = str(tmp_path)
    lens.save(cal, root)
    st = Station({"lens_correction": True, "log_dir": "runs"}, root, simulate=True, camera=None)
    assert st.lens is not None and st.lens.mm_per_px == pytest.approx(cal["mm_per_px"])
    raw = sim.render("square", "top", seed=1, scene="easy")["frame"]

    class Cam:
        part = None

        def frame(self):
            return lens.distort(raw)

        def burst(self, n=8, settle_s=0):
            return [self.frame()] * n
    cc = CorrectedCamera(Cam(), st.lens)
    back = cc.frame()
    centre = (slice(140, 340), slice(220, 420))
    assert np.abs(back[centre].astype(int) - raw[centre].astype(int)).mean() < \
        0.5 * np.abs(lens.distort(raw)[centre].astype(int) - raw[centre].astype(int)).mean()
    off = Station({"lens_correction": False, "log_dir": "runs"}, root, simulate=True)
    assert off.lens is None


def test_model_trained_without_the_calibration_is_unsure(tmp_path, cal, cfg, empty_ref, hand):
    from conftest import trained
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "models"))
    lens.save(cal, root)
    lc = lens.LensCorrection.load(os.path.join(root, lens.FILE))
    empty_ref.lens_id = lc.id
    empty_ref.save(os.path.join(root, "models", "empty.npz"))
    empty_ref.lens_id = None
    trained("square", "top", cfg, empty_ref, hand).save(os.path.join(root, "models", "square", "top"))
    json.dump({"threshold": 5.0}, open(os.path.join(root, "models", "square", "top", "threshold.json"), "w"))
    st = Station(dict(cfg), root, simulate=True, camera=capture.SimCamera(seed=1, scene="easy"))
    st._backbone = hand
    m = st.inspect_frame(sim.render("square", "top", seed=2)["frame"], "square", "top", "x")
    assert m["decision"] == "reject" and "calibration" in m["reason"]
