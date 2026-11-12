"""Optional part finder using the CentreNet detector in tools/detector (SWITCH locate_with_detector, off).

Changelog
  0.1.0  2026-11-12  first version: wraps tools/detector/infer.py (copied from alvik-fleet-capstone).

The default way to find the part is the empty reference (background.py): in a controlled scene nothing
else differs from the empty spot. The detector is kept for setups where that is not enough (a busy
background, several parts in view): it finds the part's centre, and the empty-reference mask is then
taken only around that centre. Train it with the recognition studio (tools/recognition, object type
"part"), export to ONNX (tools/detector/export.py), and set "detector_weights" in qc_config.json.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DET = os.path.join(os.path.dirname(HERE), "tools", "detector")


class DetectorLocator:
    def __init__(self, weights, cls="part", search_radius=200):
        if DET not in sys.path:
            sys.path.insert(0, DET)
        import infer                       # tools/detector/infer.py (ONNX Runtime only)
        self.infer = infer
        self.model = infer.load(weights)
        self.cls = cls
        self.search_radius = search_radius

    def centre(self, frame):
        dets = self.infer.predict(self.model, frame)
        dets = [d for d in dets if d.get("cls", self.cls) == self.cls] or dets
        if not dets:
            return None
        d = max(dets, key=lambda d: d.get("score", 0))
        return (float(d["x"]), float(d["y"]))
