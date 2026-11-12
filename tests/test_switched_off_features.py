"""Features kept behind switches (off by default): clock line-up, px-to-mm, the detector locator, the
robot presets in the studio, and the QC labelling page.

Changelog
  0.1.0  2026-11-12  first version.
"""
import json
import os
import sys
import threading
import urllib.request

import cv2
import numpy as np

from conftest import ROOT
from qc import calib, clock, sim
from qc import config as cfgmod


def test_switches_are_off_by_default():
    d = cfgmod.DEFAULTS
    assert d["locate_with_detector"] is False
    assert d["robot_features"] is False and d["clock_lineup"] is False and d["px_to_mm"] is False


def test_clock_lineup_finds_a_known_offset():
    rng = np.random.default_rng(0)
    t = np.arange(0, 60, 0.1)                                  # camera clock, 10 frames per second
    events, s = [], 0.0
    while s < 60:
        events.append((s, "moving"))
        s += rng.uniform(1.0, 2.5)
        events.append((s, "paused"))
        s += rng.uniform(1.5, 3.0)
    true_off = 2.34                                            # arm clock is 2.34 s behind the camera
    state = clock.arm_state([(e + true_off, k) for e, k in events], t)
    motion = state * 5 + rng.normal(0, 0.5, len(t))
    off, corr = clock.estimate_offset(t, motion, events, search_s=5, step_s=0.02)
    assert abs(off - true_off) < 0.11 and corr > 0.8
    keep = clock.frames_in_windows(t, clock.paused_windows(events, off), 0.2)
    assert len(keep) > 0 and state[keep].max() == 0


def test_px_to_mm_from_a_card_at_the_spot(empty_ref):
    f = sim.empty_frame(seed=5).copy()
    mm_per_px_true = 0.25
    w, h = 85.6 / mm_per_px_true, 54.0 / mm_per_px_true             # 342 x 216 px
    box = cv2.boxPoints(((320, 240), (w, h), 12.0)).astype(np.float32)
    cv2.fillConvexPoly(f, box.astype(np.int32), (230, 230, 230), cv2.LINE_AA)
    r = calib.from_card(empty_ref, f, 85.6, 54.0)
    assert abs(r["mm_per_px"] - mm_per_px_true) / mm_per_px_true < 0.02
    c = calib.from_corners([[0, 0], [400, 0], [400, 200], [0, 200]], 100.0, 50.0)
    assert abs(c["mm_per_px"] - 0.25) < 1e-6


def test_detector_locator_is_only_loaded_when_switched_on(empty_ref, tmp_path, cfg, monkeypatch):
    from qc import station as stmod

    class FakeLoc:
        search_radius = 150

        def __init__(self, w):
            pass

        def centre(self, frame):
            return (320.0, 240.0)

    import qc.locate_detector as ld
    monkeypatch.setattr(ld, "DetectorLocator", FakeLoc)
    os.makedirs(tmp_path / "models")
    empty_ref.save(str(tmp_path / "models" / "empty.npz"))
    st = stmod.Station(dict(cfg), str(tmp_path), simulate=True)
    f = sim.render("disc", "top", seed=2)["frame"]
    assert st.locate(f) is not None and st.detector is None
    st2 = stmod.Station(dict(cfg, locate_with_detector=True, detector_weights="x.onnx"), str(tmp_path), simulate=True)
    assert st2.locate(f) is not None and isinstance(st2.detector, FakeLoc)


def test_studio_offers_the_qc_preset_and_hides_the_robot_one(monkeypatch):
    sys.path.insert(0, os.path.join(ROOT, "tools", "recognition"))
    import importlib
    monkeypatch.delenv("QC_ROBOT_FEATURES", raising=False)
    import project
    project = importlib.reload(project)
    ids = [p["id"] for p in project.PRESETS]
    assert ids[0] == "qc" and "alvik" not in ids
    names = [c["name"] for c in project.QC_PRESET["classes"]]
    assert {"part", "scratch", "chip", "dent", "missing_corner", "stain"} <= set(names)
    monkeypatch.setenv("QC_ROBOT_FEATURES", "1")
    project = importlib.reload(project)
    assert "alvik" in [p["id"] for p in project.PRESETS]
    monkeypatch.delenv("QC_ROBOT_FEATURES")
    importlib.reload(project)


def test_qc_labelling_page_saves_labels(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "tools", "qclabel"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("qclabel_app", os.path.join(ROOT, "tools", "qclabel", "app.py"))
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    for i in range(2):
        cv2.imwrite(str(tmp_path / ("p%d.png" % i)), sim.render("square", "top", seed=i)["frame"])
    store = app.Store(str(tmp_path))
    srv = app.make_server(store, None, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = "http://127.0.0.1:%d" % srv.server_address[1]
        assert b"QC labelling" in urllib.request.urlopen(url + "/", timeout=10).read()
        lst = json.loads(urllib.request.urlopen(url + "/api/list", timeout=10).read())
        assert lst["files"] == ["p0.png", "p1.png"]
        body = json.dumps(dict(file="p1.png", label="defect", regions=[dict(x=1, y=2, w=30, h=20, type="chip")]))
        r = json.loads(urllib.request.urlopen(urllib.request.Request(url + "/api/label", body.encode()), timeout=10).read())
        assert r["ok"]
    finally:
        srv.shutdown()
        srv.server_close()
    saved = json.load(open(tmp_path / "qc_labels.json"))
    assert saved["p1.png"]["label"] == "defect" and saved["p1.png"]["regions"][0]["type"] == "chip"
    n = app.export(app.Store(str(tmp_path)), str(tmp_path / "root"), "square", "top")
    assert n["defect"] == 1 and os.path.exists(tmp_path / "root" / "data" / "damaged" / "square" / "top" / "p1.png")
