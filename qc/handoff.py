"""Talking to the arm: one small JSON message in, one JSON message back.

Changelog
  0.1.0  2026-11-12  first version: HTTP (default), stdin/stdout lines, and a file drop folder.
  0.2.0  2026-11-12  audit fixes: "recheck" in the decision message, "force" on "empty" (H1 / H6); a request
                     that is JSON but not an object gets an error reply instead of a dropped connection (L3).

The arm's code sends a request when something happens; the reply comes back on the same channel.

  request                                   when                          reply
  {"cmd": "paused", "part_id": "p17",       the part is resting at the    the decision (below)
   "shape": "square", "face": "top"}        spot, this face up
  {"cmd": "empty"}                          the spot is empty (start of   {"ok": true, ...}, or {"ok": false,
                                            a session, or when a reply     "refused": true, "reason": ...} when
                                            said "empty_due": true)        something is in view (old one kept)
  {"cmd": "empty", "force": true}           after a deliberate change of backdrop / light / camera
  {"cmd": "health"}                         any time                      models, thresholds, empty age

The decision message:
  {"part_id": "p17", "shape": "square", "face": "top",
   "decision": "accept" | "reject",          # this face
   "part_decision": "accept" | "reject" | "pending",   # the whole part: reject if any face was rejected;
                                             # "pending" until both faces are seen
   "score": 3.91, "threshold": 6.80,         # reject when score > threshold
   "reason": "", "warnings": [],             # why, in words, when not a plain accept/reject
   "needs_attention": false,                 # true: could not judge properly (no part, moving, drift, blur,
                                             # mix-up). The decision is then "reject" (setting unsure_goes_to),
                                             # never "accept"; "reason" says why
   "recheck": false,                         # true: waiting a moment and sending "paused" again for the same
                                             # part_id and face may help (part was moving, blurred, at the edge)
   "empty_due": false,                       # true: please send "empty" before the next part
   "heatmap_path": "runs/.../p17_top_heat.png", "image_path": "runs/.../p17_top_frame.jpg",
   "time": "2026-11-12T10:15:02.120", "model": "square/top 2026-11-12T09:58:11", "version": "0.2.0"}

part_id: the same id for both faces of one part. A face sent again for an unfinished part replaces the
earlier answer (a recheck); once a part is finished (both faces answered, or rejected by its score) the
same id starts a new part. Without part_id the faces pair up in order and the reply carries the id used.

Transports (setting "transport"):
  http   POST the request JSON to http://127.0.0.1:8765/<cmd> (or to / with "cmd" in the body).
         One call: the reply is the decision. Python:  requests.post(url, json=req).json()
  stdio  one JSON request per line on stdin, one JSON reply per line on stdout (for arm programs that
         start this program themselves).
  file   write the request as inbox/<anything>.json; the reply appears as outbox/<same name>.json.
"""
import glob
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

COMMANDS = ("paused", "inspect", "empty", "health", "collect")


def handle(station, req):
    """Run one request against the station; always returns a dict (errors come back as messages)."""
    if not isinstance(req, dict):
        return dict(ok=False, error="the request must be a JSON object such as {\"cmd\": \"paused\", ...}",
                    decision=station.cfg.get("unsure_goes_to", "reject"), needs_attention=True,
                    reason="request is not a JSON object")
    cmd = str(req.get("cmd", "paused")).lower()
    try:
        if cmd in ("paused", "inspect", "collect") and not (req.get("shape") and req.get("face")):
            return dict(ok=False, error="\"%s\" needs \"shape\" and \"face\" (and a \"part_id\")" % cmd,
                        decision=station.cfg.get("unsure_goes_to", "reject"), part_id=req.get("part_id"),
                        needs_attention=True, reason="request without shape or face")
        if cmd in ("paused", "inspect"):
            if station.mode == "collect":
                return station.collect(req.get("part_id"), req.get("shape"), req.get("face"))
            return station.inspect(req.get("part_id"), req.get("shape"), req.get("face"))
        if cmd == "collect":
            return station.collect(req.get("part_id"), req.get("shape"), req.get("face"))
        if cmd == "empty":
            return station.take_empty(force=bool(req.get("force", False)))
        if cmd == "health":
            return station.health()
        return dict(ok=False, error="unknown cmd %r; use one of %s" % (cmd, ", ".join(COMMANDS)))
    except Exception as e:          # noqa: BLE001 - the arm must always get an answer
        station.log.event("error", request=req, error=repr(e))
        return dict(ok=False, error=str(e), decision=station.cfg.get("unsure_goes_to", "reject"),
                    part_id=req.get("part_id"), needs_attention=True, reason="error: %s" % e)


# ------------------------------------------------------------------------------------------- http
def make_http_server(station, host="127.0.0.1", port=8765):
    lock = threading.Lock()         # one inspection at a time: there is one camera

    class H(BaseHTTPRequestHandler):
        def _send(self, obj, code=200):
            b = json.dumps(obj, default=float).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.strip("/") in ("", "health"):
                with lock:
                    return self._send(station.health())
            return self._send(dict(ok=False, error="use POST for %s" % self.path), 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._send(dict(ok=False, error="body is not JSON"), 400)
            p = self.path.strip("/")
            if p and isinstance(req, dict):
                req["cmd"] = p
            with lock:
                return self._send(handle(station, req))

        def log_message(self, *a):
            pass

    return ThreadingHTTPServer((host, port), H)


def serve_http(station, host, port, ready=None):
    srv = make_http_server(station, host, port)
    station.log.event("start", transport="http", host=host, port=srv.server_address[1])
    if ready:
        ready(srv)
    try:
        srv.serve_forever(poll_interval=0.2)
    finally:
        srv.server_close()
    return srv


# ------------------------------------------------------------------------------------------- stdio
def serve_stdio(station, inp=None, out=None):
    inp = inp or sys.stdin
    out = out or sys.stdout
    station.log.event("start", transport="stdio")
    for line in inp:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            rep = dict(ok=False, error="line is not JSON")
        else:
            if isinstance(req, dict) and str(req.get("cmd", "")).lower() in ("quit", "stop", "exit"):
                break
            rep = handle(station, req)
        out.write(json.dumps(rep, default=float) + "\n")
        out.flush()


# ------------------------------------------------------------------------------------------- file
def serve_files(station, inbox, outbox, poll_s=0.1, stop=None):
    os.makedirs(inbox, exist_ok=True)
    os.makedirs(outbox, exist_ok=True)
    station.log.event("start", transport="file", inbox=inbox, outbox=outbox)
    while not (stop and stop.is_set()):
        for p in sorted(glob.glob(os.path.join(inbox, "*.json"))):
            try:
                with open(p) as f:
                    req = json.load(f)
            except ValueError:
                time.sleep(poll_s)          # still being written: try again next round
                continue
            rep = handle(station, req)
            name = os.path.basename(p)
            tmp = os.path.join(outbox, "." + name + ".tmp")
            with open(tmp, "w") as f:
                json.dump(rep, f, default=float)
            os.replace(tmp, os.path.join(outbox, name))
            os.remove(p)
        time.sleep(poll_s)
