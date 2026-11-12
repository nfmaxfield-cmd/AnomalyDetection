"""The station end to end on simulated pictures: JSON hand-off over stdio, HTTP and files; the decision log;
part decisions over two faces; unsure cases; periodic empty pictures; drift warnings on parts.

Changelog
  0.1.0  2026-11-12  first version.
  0.2.0  2026-11-12  audit fixes: H1 (never "accept" with a warning; recheck), H5 (part ids), H6 (empty refused
                     with something in view; training-time empty), M2 (focus), M4 (backbone guard), M5 (collect
                     checks), M9 (mix-ups get a reason), L2 (parts counted), L3 (non-object JSON), L5 (threshold
                     reload, log pictures not overwritten).
"""
import csv
import io
import json
import os
import threading
import urllib.request

import pytest

from conftest import trained
from qc import capture, handoff, sim
from qc import threshold as thmod
from qc.station import Station

FIELDS = {"part_id", "shape", "face", "decision", "part_decision", "score", "threshold", "reason", "warnings",
          "needs_attention", "empty_due", "heatmap_path", "image_path", "time", "version"}


@pytest.fixture()
def station(tmp_path, cfg, empty_ref, hand):
    """A working folder with an empty reference and a square/top + square/bottom model."""
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "models"), exist_ok=True)
    empty_ref.save(os.path.join(root, "models", "empty.npz"))
    for face in ("top", "bottom"):
        m = trained("square", face, cfg, empty_ref, hand)
        folder = os.path.join(root, "models", "square", face)
        m.save(folder)
        from conftest import scores
        good = scores(m, empty_ref, [sim.render("square", face, seed=41000 + i)["frame"] for i in range(8)], hand)
        with open(os.path.join(folder, "threshold.json"), "w") as f:
            json.dump(thmod.from_scores(good), f, default=float)
    st = Station(dict(cfg, empty_every_parts=3), root, simulate=True, camera=capture.SimCamera(seed=5))
    st._backbone = hand
    return st


def test_stdio_handoff_messages(station):
    cam = station.camera
    lines = []
    cam.next_part("square", "top", seed=42001)
    lines.append(json.dumps(dict(cmd="paused", part_id="p1", shape="square", face="top")))
    out = io.StringIO()
    handoff.serve_stdio(station, io.StringIO("\n".join(lines) + "\n"), out)
    msg = json.loads(out.getvalue().strip().splitlines()[-1])
    assert FIELDS <= set(msg)
    assert msg["decision"] == "accept" and msg["part_decision"] == "pending"
    assert msg["score"] < msg["threshold"]
    assert os.path.exists(msg["heatmap_path"]) and os.path.exists(msg["image_path"])


def test_part_decision_needs_both_faces_and_any_reject_rejects(station):
    cam = station.camera
    cam.next_part("square", "top", seed=42002)
    a = handoff.handle(station, dict(cmd="paused", part_id="p2", shape="square", face="top"))
    cam.next_part("square", "bottom", seed=42003)
    b = handoff.handle(station, dict(cmd="paused", part_id="p2", shape="square", face="bottom"))
    assert a["part_decision"] == "pending" and b["part_decision"] == "accept"
    cam.next_part("square", "top", seed=42004)
    handoff.handle(station, dict(cmd="paused", part_id="p3", shape="square", face="top"))
    cam.next_part("square", "bottom", "missing_corner", seed=42005)
    c = handoff.handle(station, dict(cmd="paused", part_id="p3", shape="square", face="bottom"))
    assert c["decision"] == "reject" and c["part_decision"] == "reject"
    assert c["score"] > c["threshold"]


def test_unsure_cases_go_to_reject_with_a_reason(station):
    station.camera.clear()                     # nothing at the spot
    station.camera.part = None
    m = station.inspect_frame(sim.empty_frame(seed=3), "square", "top", "p9")
    assert m["decision"] == "reject" and m["needs_attention"] and "no part" in m["reason"]
    m = station.inspect_frame(sim.render("disc", "top", seed=4)["frame"], "disc", "top", "p10")
    assert m["decision"] == "reject" and "no model" in m["reason"]


def test_empty_due_after_n_parts_and_empty_command(station):
    cam = station.camera
    msgs = []
    for i in range(3):
        cam.next_part("square", "top", seed=42100 + i)
        msgs.append(handoff.handle(station, dict(cmd="paused", part_id="q%d" % i, shape="square", face="top")))
    assert [m["empty_due"] for m in msgs] == [False, False, True]
    cam.clear()
    r = handoff.handle(station, dict(cmd="empty"))
    assert r["ok"] and not r["drift"]
    assert not station.health()["empty_due"]


def test_drift_on_a_part_picture_is_a_warning(station):
    f = sim.render("square", "top", seed=42200, drift="stray_object")["frame"]
    m = station.inspect_frame(f, "square", "top", "d1")
    assert any("drift" in w for w in m["warnings"]) and m["needs_attention"]


def test_decision_log_has_every_decision(station):
    cam = station.camera
    for i in range(2):
        cam.next_part("square", "top", seed=42300 + i)
        handoff.handle(station, dict(cmd="paused", part_id="L%d" % i, shape="square", face="top"))
    rows = [json.loads(l) for l in open(station.log.jsonl)]
    assert [r["part_id"] for r in rows] == ["L0", "L1"]
    with open(station.log.csv) as f:
        c = list(csv.DictReader(f))
    assert [r["part_id"] for r in c] == ["L0", "L1"] and c[0]["decision"] in ("accept", "reject")
    imgs = os.listdir(station.log.images)
    assert {"L0_top_frame.jpg", "L0_top_heat.png", "L0_top_where.jpg"} <= set(imgs)


def test_http_handoff(station):
    srv = handoff.make_http_server(station, "127.0.0.1", 0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        url = "http://127.0.0.1:%d" % srv.server_address[1]
        h = json.loads(urllib.request.urlopen(url + "/health", timeout=30).read())
        assert h["ok"] and "square/top" in h["models"]
        station.camera.next_part("square", "top", seed=42400)
        req = urllib.request.Request(url + "/paused", json.dumps(dict(part_id="h1", shape="square", face="top")).encode())
        m = json.loads(urllib.request.urlopen(req, timeout=60).read())
        assert m["part_id"] == "h1" and m["decision"] in ("accept", "reject")
        bad = urllib.request.Request(url + "/nonsense", b"{}")
        assert "error" in json.loads(urllib.request.urlopen(bad, timeout=30).read())
    finally:
        srv.shutdown()
        srv.server_close()


def test_file_handoff(station, tmp_path):
    inbox, outbox = str(tmp_path / "in"), str(tmp_path / "out")
    os.makedirs(inbox)
    station.camera.next_part("square", "top", seed=42500)
    with open(os.path.join(inbox, "req1.json"), "w") as f:
        json.dump(dict(cmd="paused", part_id="f1", shape="square", face="top"), f)
    stop = threading.Event()
    t = threading.Thread(target=handoff.serve_files, args=(station, inbox, outbox, 0.05, stop), daemon=True)
    t.start()
    import time
    for _ in range(600):
        if os.path.exists(os.path.join(outbox, "req1.json")):
            break
        time.sleep(0.05)
    stop.set()
    t.join(5)
    m = json.load(open(os.path.join(outbox, "req1.json")))
    assert m["part_id"] == "f1" and not os.path.exists(os.path.join(inbox, "req1.json"))


# ------------------------------------------------------------------------------------------- audit fixes
class FramesCam:
    """A camera that returns the given burst."""
    part = None

    def __init__(self, frames):
        self.frames = frames

    def burst(self, n=8, settle_s=0):
        return self.frames

    def frame(self):
        return self.frames[0]


def sliding(seed, px, n=8, defect=None):
    return [sim.render("square", "top", defect, seed=seed, angle=30, centre=(320 + px * k, 240), motion_px=px,
                       noise_seed=seed * 31 + k)["frame"] for k in range(n)]


def test_h1_moving_part_is_never_accepted_and_asks_for_a_recheck(station):
    station.camera = FramesCam(sliding(601, 1.0))
    m = station.inspect("mv1", "square", "top")
    assert m["decision"] == "reject" and m["needs_attention"] and m["recheck"]
    assert "moving" in m["reason"]


@pytest.mark.parametrize("drift", sim.DRIFTS)
def test_h1_changed_backdrop_with_a_good_part_is_never_accepted(station, drift):
    m = station.inspect_frame(sim.render("square", "top", seed=502, drift=drift)["frame"], "square", "top", "dr")
    assert m["decision"] == "reject" and m["needs_attention"] and m["reason"]


def test_h1_switch_off_gives_the_first_builds_behaviour(station):
    station.cfg["warnings_make_unsure"] = False
    m = station.inspect_frame(sim.render("square", "top", seed=503, drift="stray_object")["frame"], "square", "top", "x")
    assert m["needs_attention"] and m["decision"] == ("reject" if m["score"] > m["threshold"] else "accept")


def test_h5_reused_id_starts_a_new_part(station):
    a = station.inspect_frame(sim.render("square", "top", seed=510)["frame"], "square", "top", "p1")
    b = station.inspect_frame(sim.render("square", "bottom", "stain", seed=511)["frame"], "square", "bottom", "p1")
    assert a["decision"] == "accept" and b["part_decision"] == "reject"
    c = station.inspect_frame(sim.render("square", "top", seed=512)["frame"], "square", "top", "p1")
    assert c["decision"] == "accept" and c["part_decision"] == "pending"


def test_h5_faces_pair_up_without_an_id(station):
    a = station.inspect_frame(sim.render("square", "top", seed=513)["frame"], "square", "top", None)
    b = station.inspect_frame(sim.render("square", "bottom", seed=514)["frame"], "square", "bottom", None)
    assert a["part_id"] == b["part_id"] and b["part_decision"] == "accept"
    c = station.inspect_frame(sim.render("square", "top", seed=515)["frame"], "square", "top", None)
    assert c["part_id"] != a["part_id"] and c["part_decision"] == "pending"


def test_h5_a_recheck_replaces_the_earlier_answer(station):
    station.camera = FramesCam(sliding(620, 2.0))
    a = station.inspect("r1", "square", "top")
    assert a["decision"] == "reject" and a["recheck"]
    b = station.inspect_frame(sim.render("square", "top", seed=620)["frame"], "square", "top", "r1")
    assert b["decision"] == "accept" and b["part_decision"] == "pending"
    c = station.inspect_frame(sim.render("square", "bottom", seed=621)["frame"], "square", "bottom", "r1")
    assert c["part_decision"] == "accept"


def test_h6_empty_with_a_part_in_view_is_refused_and_the_old_one_kept(station):
    before = open(station.path("models", "empty.npz"), "rb").read()
    r = station.take_empty([sim.render("square", "top", seed=640, noise_seed=i)["frame"] for i in range(10)])
    assert not r["ok"] and r["refused"] and "in view" in r["reason"]
    assert open(station.path("models", "empty.npz"), "rb").read() == before
    m = station.inspect_frame(sim.render("square", "top", seed=641)["frame"], "square", "top", "after")
    assert m["score"] is not None and m["decision"] == "accept"
    r = station.take_empty([sim.empty_frame(seed=650 + i) for i in range(10)])
    assert r["ok"]


def test_h6_forced_empty_after_a_light_change_warns_against_the_training_empty(station, empty_ref):
    for face in ("top", "bottom"):
        empty_ref.save(station.path("models", "square", face, "empty_at_training.npz"))
    station.models.clear()
    station.model("square", "top")
    frames = [sim.empty_frame(seed=660 + i, drift="light_shift") for i in range(10)]
    r = station.take_empty(frames, force=True)
    assert r["ok"] and "trained" in r.get("warning", "")
    m = station.inspect_frame(sim.render("square", "top", seed=661, drift="light_shift")["frame"], "square", "top", "ls")
    assert any("trained" in w for w in m["warnings"])


def test_m2_blurred_picture_is_unsure(station):
    m = station.inspect_frame(sim.render("square", "top", seed=670, defocus=2.0)["frame"], "square", "top", "bl")
    assert m["decision"] == "reject" and m["needs_attention"] and "blurrier" in m["reason"]
    ok = station.inspect_frame(sim.render("square", "top", seed=671)["frame"], "square", "top", "sh")
    assert "blurrier" not in ok["reason"]


def test_m4_model_refuses_another_backbone(station):
    class Other(type(station.backbone)):
        sha256 = "something-else"
    station._backbone = Other()
    m = station.inspect_frame(sim.render("square", "top", seed=680)["frame"], "square", "top", "bb")
    assert m["decision"] == "reject" and "different network" in m["reason"]


def test_m5_collect_refuses_an_empty_spot_and_an_edge_part(station):
    station.mode = "collect"
    r = station.collect("c1", "square", "top", frame=sim.empty_frame(seed=690))
    assert not r["ok"] and "no part" in r["reason"]
    r = station.collect("c2", "square", "top", frame=sim.render("square", "top", seed=691, centre=(60, 240))["frame"])
    assert not r["ok"] and "edge" in r["reason"]
    r = station.collect("c3", "square", "top", frame=sim.render("square", "top", seed=692)["frame"])
    assert r["ok"] and os.path.exists(r["file"])


@pytest.mark.parametrize("shape,face", [("hexagon", "top"), ("disc", "top")])
def test_m9_wrong_shape_gets_a_reason(station, shape, face):
    m = station.inspect_frame(sim.render(shape, face, seed=700)["frame"], "square", "top", "ws")
    assert m["decision"] == "reject" and m["needs_attention"] and m["reason"]


def test_m9_good_parts_raise_no_checks(station):
    for i in range(6):
        m = station.inspect_frame(sim.render("square", "top", seed=710 + i)["frame"], "square", "top", "g%d" % i)
        assert not m["needs_attention"], m["reason"]


def test_l2_empty_every_parts_counts_parts_not_faces(station):
    station.since_empty = 0
    station.inspect_frame(sim.render("square", "top", seed=720)["frame"], "square", "top", "L2")
    station.inspect_frame(sim.render("square", "bottom", seed=721)["frame"], "square", "bottom", "L2")
    assert station.since_empty == 1


def test_l3_json_that_is_not_an_object_gets_a_reply(station):
    r = handoff.handle(station, [1, 2])
    assert r["ok"] is False and "JSON object" in r["error"]
    srv = handoff.make_http_server(station, "127.0.0.1", 0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        url = "http://127.0.0.1:%d/paused" % srv.server_address[1]
        rep = json.loads(urllib.request.urlopen(urllib.request.Request(url, b"[1]"), timeout=30).read())
        assert rep["ok"] is False
    finally:
        srv.shutdown()
        srv.server_close()


def test_l5_threshold_reloads_and_log_pictures_are_not_overwritten(station):
    a = station.inspect_frame(sim.render("square", "top", seed=730)["frame"], "square", "top", "same")
    tp = station.path("models", "square", "top", "threshold.json")
    th = json.load(open(tp))
    th["threshold"] = 0.01
    import time
    time.sleep(0.02)
    json.dump(th, open(tp, "w"))
    os.utime(tp, (time.time() + 5, time.time() + 5))
    b = station.inspect_frame(sim.render("square", "top", seed=731)["frame"], "square", "top", "same")
    assert b["threshold"] == 0.01 and b["decision"] == "reject"
    assert a["heatmap_path"] != b["heatmap_path"] and os.path.exists(a["heatmap_path"])
