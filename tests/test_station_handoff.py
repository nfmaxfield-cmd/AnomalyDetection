"""The station end to end on simulated pictures: JSON hand-off over stdio, HTTP and files; the decision log;
part decisions over two faces; unsure cases; periodic empty pictures; drift warnings on parts.

Changelog
  0.1.0  2026-11-12  first version.
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
