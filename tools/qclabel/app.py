#!/usr/bin/env python3
"""QC labelling page: mark pictures good or defect, and draw boxes around defects. The computer guesses
first (from the trained model's heat map), you confirm or correct.

Changelog
  0.1.0  2026-11-12  first version. Adapted from the labelling studio of alvik-fleet-capstone
                     (tools/label/app.py and tools/recognition/studio.py, copied whole into tools/label and
                     tools/recognition): same idea (local page, guesses first, Enter accepts, every change
                     saved at once), rebuilt small for whole-picture good/defect labels and defect boxes. The
                     full studio (point labels, tracking, detector training) is still there for the
                     optional detector (qc setting locate_with_detector).

    python tools/qclabel/app.py <folder of pictures>
    python tools/qclabel/app.py runs/20261112_101500/images --shape square --face top     # with guesses
    python tools/qclabel/app.py <folder> --export . --shape square --face top             # copy labelled
                                   pictures: defect -> data/damaged/<shape>/<face>/, good -> data/good/...

Keys on the page: G good, D defect, drag = draw a box, 1-5 = defect type of the selected box,
Delete = remove the selected box, Enter = confirm and next, Left/Right = previous/next.
Saved after every change in <folder>/qc_labels.json:
  {"<file>": {"label": "good"|"defect", "regions": [{"x","y","w","h","type"}], "by": "person"|"guess",
              "time": "..."}}
Served only on this computer (127.0.0.1). Needs numpy and opencv-python.
"""
import argparse
import datetime as dt
import glob
import json
import os
import shutil
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFECTS = ["scratch", "chip", "dent", "missing_corner", "stain"]
EXTS = (".png", ".jpg", ".jpeg", ".bmp")


class Store:
    def __init__(self, folder):
        self.folder = folder
        self.path = os.path.join(folder, "qc_labels.json")
        self.lock = threading.Lock()
        self.labels = json.load(open(self.path)) if os.path.exists(self.path) else {}
        self.files = sorted(f for f in os.listdir(folder) if f.lower().endswith(EXTS)
                            and not f.endswith(("_heat.png",)))

    def put(self, name, label, regions, by="person"):
        if name not in self.files:
            raise KeyError(name)
        if label not in ("good", "defect"):
            raise ValueError("label must be good or defect")
        with self.lock:
            self.labels[name] = dict(label=label, regions=regions, by=by,
                                     time=dt.datetime.now().isoformat(timespec="seconds"))
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.labels, f, indent=1)
            os.replace(tmp, self.path)
        return self.labels[name]


class Guesser:
    """Guesses from a trained model: score above threshold -> defect, boxes around the hot areas."""

    def __init__(self, root, shape, face):
        sys.path.insert(0, ROOT)
        from qc import config, station
        self.st = station.Station(config.load(os.path.join(root, "qc_config.json")), root, simulate=True,
                                  log_name="qclabel_guesses")
        self.shape, self.face = shape, face

    def guess(self, path):
        import cv2
        import numpy as np
        frame = cv2.imread(path)
        msg = self.st.inspect_frame(frame, self.shape, self.face, os.path.basename(path))
        if msg.get("score") is None or msg.get("threshold") is None:
            return None
        out = dict(label=msg["decision"] == "reject" and "defect" or "good", regions=[], score=msg["score"],
                   threshold=msg["threshold"])
        if out["label"] == "defect":
            x, y = msg["hot_spot_px"]
            out["regions"].append(dict(x=x - 20, y=y - 20, w=40, h=40, type="unknown"))
        return out


def make_server(store, guesser=None, port=8770):
    page = open(os.path.join(HERE, "page.html"), encoding="utf-8").read()
    cache = {}

    class H(BaseHTTPRequestHandler):
        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                b = page.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            elif u.path == "/api/list":
                self._json(dict(files=store.files, labels=store.labels, defects=DEFECTS,
                                guesses=bool(guesser)))
            elif u.path == "/api/guess":
                name = parse_qs(u.query).get("f", [""])[0]
                if not guesser or name not in store.files:
                    return self._json(None)
                if name not in cache:
                    cache[name] = guesser.guess(os.path.join(store.folder, name))
                self._json(cache[name])
            elif u.path.startswith("/img/"):
                name = unquote(u.path[5:])
                if name not in store.files:
                    return self._json(dict(error="no such picture"), 404)
                b = open(os.path.join(store.folder, name), "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", "image/png" if name.lower().endswith("png") else "image/jpeg")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            else:
                self._json(dict(error="not found"), 404)

        def do_POST(self):
            if urlparse(self.path).path != "/api/label":
                return self._json(dict(error="not found"), 404)
            n = int(self.headers.get("Content-Length") or 0)
            try:
                req = json.loads(self.rfile.read(n))
                rec = store.put(req["file"], req["label"], req.get("regions", []), req.get("by", "person"))
            except (ValueError, KeyError) as e:
                return self._json(dict(ok=False, error=str(e)), 400)
            self._json(dict(ok=True, saved=rec))

        def log_message(self, *a):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), H)


def export(store, root, shape, face):
    n = dict(good=0, defect=0)
    for name, rec in store.labels.items():
        sub = "damaged" if rec["label"] == "defect" else "good"
        d = os.path.join(root, "data", sub, shape, face)
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, name)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(store.folder, name), dst)
            n[rec["label"]] += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--shape")
    ap.add_argument("--face")
    ap.add_argument("--root", default=".", help="QC working folder (models/, data/)")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--export", default=None, help="copy labelled pictures into <EXPORT>/data/... and stop")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    store = Store(a.folder)
    if a.export:
        if not (a.shape and a.face):
            ap.error("--export needs --shape and --face")
        print("Copied: %s" % export(store, a.export, a.shape, a.face))
        return 0
    guesser = Guesser(a.root, a.shape, a.face) if (a.shape and a.face) else None
    srv = make_server(store, guesser, a.port)
    url = "http://127.0.0.1:%d/" % srv.server_address[1]
    print("QC labelling page: %s  (%d pictures; Ctrl+C here to stop)" % (url, len(store.files)))
    if not a.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
