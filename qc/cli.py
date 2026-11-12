"""Command line: python -m qc <command> ...

Changelog
  0.1.0  2026-11-12  first version.

  collect-empty        pictures of the empty spot -> models/empty.npz (the background reference)
  collect-good         pictures of good parts -> data/good/<shape>/<face>/
  collect-damaged      pictures of deliberately damaged parts -> data/damaged/<shape>/<face>/
  train                one normal model per shape and face (+ threshold from held-out data)
  learning-curve       how many good pictures are enough: separation vs number of training pictures
  calibrate-threshold  threshold again from held-out good (+ damaged) scores
  run                  live: wait for the arm's signals, judge each part, answer the arm
  inspect IMAGE        judge one picture file and print the decision message
  camera-setup         live picture with sliders: lock exposure, focus and white balance (S saves)
  calibrate-mm         (switch px_to_mm) mm per pixel from a card of known size at the spot

--simulate on any command uses computer-generated pictures and a fake arm instead of the webcam.
"""
import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.request

from . import __version__
from . import config as cfgmod


def _sim_seed_base(shape, face, kind):
    import zlib
    return (zlib.crc32(("%s/%s/%s" % (shape, face, kind)).encode()) % 9973) * 1000


def _shapes(a):
    """(shape, face) pairs: the ones named, or for 'all' / not given: the folders found in data/good/
    (any shape name works), or the simulator's shapes with --simulate."""
    from . import sim
    import glob as _g
    if getattr(a, "simulate", False):
        found = [(s, f) for s in sim.SHAPES for f in sim.FACES]
    else:
        found = sorted(tuple(p.replace("\\", "/").split("/")[-2:]) for p in
                       _g.glob(os.path.join(a.root, "data", "good", "*", "*")) if os.path.isdir(p))
    out = [(s, f) for s, f in found if (a.shape in (None, "all") or s == a.shape)
           and (a.face in (None, "all") or f == a.face)]
    if not out and a.shape not in (None, "all"):
        faces = [a.face] if a.face not in (None, "all") else list(sim.FACES)
        out = [(a.shape, f) for f in faces]
    return out


def cmd_collect_empty(a, cfg):
    from .station import Station
    st = Station(cfg, a.root, simulate=a.simulate)
    if a.simulate:
        from . import sim
        frames = [sim.empty_frame(seed=1000 + i, drift=a.drift) for i in range(int(a.n or cfg["empty_frames"]))]
        r = st.take_empty(frames)
    else:
        cfg["empty_frames"] = int(a.n or cfg["empty_frames"])
        input("Make sure the spot is EMPTY (no part, no hands), then press Enter here ...")
        r = st.take_empty()
    print(json.dumps(r, indent=2, default=float))
    return 0


def _collect(a, cfg, damaged):
    from .station import Station
    from . import capture, sim
    import cv2
    st = Station(cfg, a.root, simulate=a.simulate)
    if st.empty is None:
        print("No empty reference yet: run  python -m qc collect-empty  first.")
        return 2
    n = int(a.n)
    sub = "damaged" if damaged else "good"
    if a.simulate:
        cam = st.cam()
        k = 0
        for shape, face in _shapes(a):
            base = int(a.seed) * 100000 + _sim_seed_base(shape, face, sub)
            for i in range(n):
                defect = sim.DEFECTS[i % len(sim.DEFECTS)] if damaged else None
                cam.next_part(shape, face, defect, seed=base + i)
                frame, loc, info = st.grab(shape, face)
                if damaged:
                    d = os.path.join(a.root, "data", "damaged", shape, face)
                    os.makedirs(d, exist_ok=True)
                    cv2.imwrite(os.path.join(d, "sim_%s_%04d.png" % (defect, i)), frame)
                else:
                    st.collect("sim%04d" % i, shape, face, frame=frame)
                k += 1
            print("%s/%s: %d simulated %s pictures in data/%s/%s/%s/" % (shape, face, n, sub, sub, shape, face))
        return 0
    if (a.manual or damaged) and (a.shape in (None, "all") or a.face in (None, "all")):
        print("Give --shape and --face when you press Enter for each picture.")
        return 2
    if a.manual or damaged:
        cam = st.cam()
        d = os.path.join(a.root, "data", sub, a.shape, a.face)
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            input("[%d/%d] Put a %s %s part %s-up at the spot, hands away, then press Enter ..." % (
                i + 1, n, sub.upper(), a.shape, a.face))
            frame, loc, info = st.grab(a.shape, a.face)
            if loc is None:
                print("   no part found - try again")
                continue
            if damaged:
                p = os.path.join(d, "%s_%s_%03d.png" % (time.strftime("%Y%m%d_%H%M%S"), a.defect or "damage", i))
                cv2.imwrite(p, frame)
                print("   saved", p)
            else:
                print("   saved", st.collect("manual%03d" % i, a.shape, a.face, frame=frame)["file"])
        return 0
    # live with the arm: the arm sends "paused" as usual (with its shape and face), every picture is kept
    st.mode = "collect"
    print("Collecting GOOD pictures from the arm's \"paused\" signals; each is saved under data/good/<shape>/<face>/. "
          "Press Ctrl+C here to stop.")
    return _serve(st, cfg)


def cmd_collect_good(a, cfg):
    return _collect(a, cfg, False)


def cmd_collect_damaged(a, cfg):
    return _collect(a, cfg, True)


def cmd_train(a, cfg):
    from . import train
    for shape, face in _shapes(a):
        d = os.path.join(a.root, "data", "good", shape, face)
        if not os.path.isdir(d):
            if a.shape not in (None, "all") and a.face not in (None, "all"):
                print("No pictures in %s" % d)
                return 2
            continue
        train.train(cfg, a.root, shape, face)
    return 0


def cmd_calibrate(a, cfg):
    from . import train
    for shape, face in _shapes(a):
        if os.path.exists(os.path.join(a.root, "models", shape, face, "model.npz")):
            train.calibrate(cfg, a.root, shape, face)
    return 0


def cmd_learning_curve(a, cfg):
    from . import charts, train
    import csv
    sizes = [int(x) for x in a.sizes.split(",")]
    for shape, face in _shapes(a):
        if not os.path.isdir(os.path.join(a.root, "data", "good", shape, face)):
            continue
        rows, info = train.learning_curve(cfg, a.root, shape, face, sizes, int(a.repeats))
        info["source"] = a.source or "pictures in data/good/%s/%s" % (shape, face)
        out = os.path.join(a.root, "models", shape, face)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "learning_curve.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        png = charts.learning_curve(rows, info, os.path.join(out, "learning_curve.png"))
        print(info["suggestion"]["text"])
        print("Chart: %s" % png)
    return 0


def _serve(st, cfg, ready=None):
    from . import handoff
    t = cfg.get("transport", "http")
    try:
        if t == "http":
            def show(srv):
                print("Listening on http://%s:%d  (the arm POSTs to /paused, /empty; GET /health)" % srv.server_address)
                if ready:
                    ready(srv)
            handoff.serve_http(st, cfg.get("http_host", "127.0.0.1"), int(cfg.get("http_port", 8765)), show)
        elif t == "stdio":
            handoff.serve_stdio(st)
        elif t == "file":
            print("Watching %s for requests; replies go to %s" % (cfg["inbox"], cfg["outbox"]))
            handoff.serve_files(st, os.path.join(st.root, cfg["inbox"]), os.path.join(st.root, cfg["outbox"]))
        else:
            print("Unknown transport %r (http, stdio or file)" % t)
            return 2
    except KeyboardInterrupt:
        print("\nStopped. Log: %s" % st.log.dir)
    return 0


def fake_arm(url, cam, parts, defect_rate=0.3, seed=0, out=print, shapes=None):
    """Simulated arm: for each part, put it down top-up, ask, flip, ask again, report what it would do."""
    from . import sim
    rng = random.Random(seed)

    def post(cmd, body):
        req = urllib.request.Request(url + "/" + cmd, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())

    cam.clear()
    post("empty", {})
    tally = {"good->accept": 0, "good->reject": 0, "damaged->accept": 0, "damaged->reject": 0}
    for i in range(parts):
        shape = rng.choice(shapes or sim.SHAPES)
        bad_face = rng.choice(sim.FACES) if rng.random() < defect_rate else None
        defect = rng.choice(sim.DEFECTS) if bad_face else None
        seed_i = 900000 + seed * 1000 + i
        last = None
        for face in sim.FACES:
            cam.next_part(shape, face, defect if face == bad_face else None, seed=seed_i + (0 if face == "top" else 500))
            last = post("paused", dict(part_id="p%03d" % (i + 1), shape=shape, face=face))
            out("p%03d %-7s %-6s truth=%-15s decision=%-6s score=%s thr=%s %s" % (
                i + 1, shape, face, (defect if face == bad_face else "good"), last.get("decision"),
                "%.2f" % last["score"] if last.get("score") is not None else "-",
                "%.2f" % last["threshold"] if last.get("threshold") is not None else "-",
                ("WARN " + "; ".join(last.get("warnings", []))) if last.get("warnings") else last.get("reason", "")))
            if last.get("decision") == "reject":
                break                    # rejected on the first face: no need to flip
        cam.clear()
        truth = "damaged" if bad_face else "good"
        tally["%s->%s" % (truth, last.get("part_decision") if last.get("part_decision") != "pending" else last["decision"])] += 1
        if last.get("empty_due"):
            post("empty", {})
    return tally


def cmd_run(a, cfg):
    from .station import Station
    st = Station(cfg, a.root, simulate=a.simulate)
    if not a.simulate:
        if st.empty is None:
            print("No empty reference yet: the arm's code must send \"empty\" first (or run collect-empty).")
        return _serve(st, cfg)
    cfg["transport"] = "http"
    cfg["http_port"] = 0                 # any free port
    done = {}

    def ready(srv):
        def go():
            url = "http://%s:%d" % srv.server_address
            try:
                trained = sorted({m.split("/")[0] for m in st.health()["models"]})
                done["tally"] = fake_arm(url, st.cam(), int(a.parts), float(a.defect_rate), int(a.seed),
                                         shapes=trained or None)
            finally:
                srv.shutdown()
        threading.Thread(target=go, daemon=True).start()
    _serve(st, cfg, ready)
    print("Simulated run finished: %s" % json.dumps(done.get("tally")))
    print("Log: %s" % st.log.dir)
    return 0


def cmd_inspect(a, cfg):
    import cv2
    from .station import Station
    st = Station(cfg, a.root, simulate=True)
    frame = cv2.imread(a.image)
    if frame is None:
        print("Cannot read %s" % a.image)
        return 2
    msg = st.inspect_frame(frame, a.shape, a.face, a.part_id or os.path.splitext(os.path.basename(a.image))[0])
    print(json.dumps(msg, indent=2, default=float))
    return 0


def cmd_camera_setup(a, cfg):
    """Live picture with sliders for exposure, focus and white balance; S saves them to qc_config.json.
    NOT tested with a real webcam in the first build (no camera in the build sandbox)."""
    import cv2
    from .capture import OpenCVCamera
    cam = OpenCVCamera(cfg)
    c = cam.cap
    win = "camera setup - S: save, Q: quit"
    cv2.namedWindow(win)
    cv2.createTrackbar("exposure +13", win, int(cfg["exposure"]) + 13, 13, lambda v: c.set(cv2.CAP_PROP_EXPOSURE, v - 13))
    cv2.createTrackbar("focus", win, int(cfg["focus"]), 255, lambda v: c.set(cv2.CAP_PROP_FOCUS, v))
    cv2.createTrackbar("white balance /100", win, int(cfg["white_balance_k"]) // 100, 80,
                       lambda v: c.set(cv2.CAP_PROP_WB_TEMPERATURE, max(20, v) * 100))
    from .capture import sharpness
    while True:
        f = cam.frame()
        g = f.copy()
        over = (f.max(axis=2) >= 250)
        g[over] = (255, 0, 255)                      # clipped (too bright) pixels shown in magenta
        cv2.putText(g, "sharpness %.0f   clipped %.2f %%" % (sharpness(f), 100 * over.mean()), (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(win, g)
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("s"), ord("S")):
            cfg["exposure"] = cv2.getTrackbarPos("exposure +13", win) - 13
            cfg["focus"] = cv2.getTrackbarPos("focus", win)
            cfg["white_balance_k"] = max(20, cv2.getTrackbarPos("white balance /100", win)) * 100
            print("Saved to %s: exposure %s, focus %s, white balance %s K" % (
                cfgmod.save(cfg), cfg["exposure"], cfg["focus"], cfg["white_balance_k"]))
        if k in (ord("q"), ord("Q"), 27):
            break
    cam.close()
    cv2.destroyAllWindows()
    return 0


def cmd_calibrate_mm(a, cfg):
    from . import calib
    from .station import Station
    import cv2
    st = Station(cfg, a.root, simulate=a.simulate)
    w, h = [float(x) for x in a.card.lower().split("x")]
    if a.image:
        frame = cv2.imread(a.image)
    elif a.simulate:
        print("--simulate needs --image (a picture of the card at the spot).")
        return 2
    else:
        input("Lay the %s x %s mm card flat at the spot, then press Enter ..." % (w, h))
        frame = st.cam().frame()
    r = calib.from_card(st.empty, frame, w, h)
    print(r["text"])
    cfg["mm_per_px"] = r["mm_per_px"]
    cfg["px_to_mm"] = True
    print("Saved to %s (px_to_mm switched on)." % cfgmod.save(cfg))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m qc", description="QC anomaly detection station %s" % __version__)
    ap.add_argument("--config", help="settings file (default qc_config.json)")
    ap.add_argument("--root", default=".", help="working folder holding data/, models/, runs/ (default: here)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, shape=True, simulate=True):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        if shape:
            p.add_argument("--shape", default=None, help="disc, square, hexagon ... (simulate/train: 'all' allowed)")
            p.add_argument("--face", default=None, help="top or bottom")
        if simulate:
            p.add_argument("--simulate", action="store_true", help="computer-generated pictures, no webcam")
            p.add_argument("--seed", default=0, type=int)
        return p

    p = add("collect-empty", cmd_collect_empty, shape=False)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--drift", default=None, help="simulate only: light_shift, stray_object, camera_bump, glare")
    p = add("collect-good", cmd_collect_good)
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--manual", action="store_true", help="press Enter for each picture instead of the arm signal")
    p = add("collect-damaged", cmd_collect_damaged)
    p.add_argument("--n", type=int, default=15)
    p.add_argument("--defect", default=None, help="what kind of damage (only used in the file name)")
    p.add_argument("--manual", action="store_true")
    add("train", cmd_train, simulate=False)
    p = add("learning-curve", cmd_learning_curve, simulate=False)
    p.add_argument("--sizes", default="5,10,20,40,80")
    p.add_argument("--repeats", default=3, type=int)
    p.add_argument("--source", default=None, help="text for the chart's data line")
    add("calibrate-threshold", cmd_calibrate, simulate=False)
    p = add("run", cmd_run, shape=False)
    p.add_argument("--parts", default=20, type=int, help="simulate: how many parts the fake arm brings")
    p.add_argument("--defect-rate", default=0.3, type=float, help="simulate: share of damaged parts")
    p.add_argument("--transport", default=None, help="http, stdio or file (overrides the settings file)")
    p = add("inspect", cmd_inspect, simulate=False)
    p.add_argument("image")
    p.add_argument("--part-id", default=None)
    add("camera-setup", cmd_camera_setup, shape=False, simulate=False)
    p = add("calibrate-mm", cmd_calibrate_mm, shape=False)
    p.add_argument("--card", default="85.6x54", help="card size in mm, WIDTHxHEIGHT (default: a bank card)")
    p.add_argument("--image", default=None)
    a = ap.parse_args(argv)
    cfg = cfgmod.load(a.config)
    if getattr(a, "transport", None):
        cfg["transport"] = a.transport
    if a.cmd == "inspect" and (not a.shape or not a.face):
        ap.error("inspect needs --shape and --face")
    return a.fn(a, cfg) or 0


if __name__ == "__main__":
    sys.exit(main())
