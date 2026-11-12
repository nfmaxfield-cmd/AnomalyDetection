"""Decision log: every decision as one JSON line and one CSV row, with its pictures, so a run can be audited.

Changelog
  0.1.0  2026-11-12  first version.
  0.2.0  2026-11-12  CSV also holds needs_attention and recheck (columns appended at the end).

    runs/<YYYYmmdd_HHMMSS>/decisions.jsonl     one JSON object per decision (the message sent to the arm)
    runs/<...>/decisions.csv                   the same, flat, for a spreadsheet
    runs/<...>/images/<part>_<face>_frame.jpg  the picture judged (setting save_raw_frames)
    runs/<...>/images/<part>_<face>_heat.png   straightened part with the heat map over it
    runs/<...>/images/<part>_<face>_where.jpg  the full picture with the suspicious spot marked
    runs/<...>/events.jsonl                    empty-spot pictures, drift warnings, start and stop
"""
import csv
import datetime as dt
import json
import os
import re
import threading

CSV_COLUMNS = ["time", "part_id", "shape", "face", "decision", "score", "threshold", "part_decision",
               "reason", "warnings", "heatmap_path", "image_path", "sharpness", "angle_deg", "model",
               "needs_attention", "recheck"]


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(s)).strip("-.") or "part"


class DecisionLog:
    def __init__(self, root="runs", name=None):
        self.dir = os.path.join(root, name or dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.images = os.path.join(self.dir, "images")
        self.jsonl = os.path.join(self.dir, "decisions.jsonl")
        self.csv = os.path.join(self.dir, "decisions.csv")
        self.events = os.path.join(self.dir, "events.jsonl")
        self.lock = threading.Lock()

    def _ready(self):
        """Folders are made on the first write, so commands that log nothing leave no empty run folder."""
        os.makedirs(self.images, exist_ok=True)
        if not os.path.exists(self.csv):
            with open(self.csv, "w", newline="") as f:
                csv.writer(f).writerow(CSV_COLUMNS)

    def image_path(self, part_id, face, kind, ext):
        self._ready()
        return os.path.join(self.images, "%s_%s_%s.%s" % (safe_name(part_id), safe_name(face), kind, ext))

    def decision(self, msg):
        with self.lock:
            self._ready()
            with open(self.jsonl, "a") as f:
                f.write(json.dumps(msg, default=float) + "\n")
            with open(self.csv, "a", newline="") as f:
                row = []
                for k in CSV_COLUMNS:
                    v = msg.get(k, "")
                    if isinstance(v, (list, tuple)):
                        v = " | ".join(str(x) for x in v)
                    row.append(v)
                csv.writer(f).writerow(row)

    def event(self, kind, **data):
        data = dict(time=dt.datetime.now().isoformat(timespec="milliseconds"), event=kind, **data)
        with self.lock:
            self._ready()
            with open(self.events, "a") as f:
                f.write(json.dumps(data, default=float) + "\n")
        return data


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
