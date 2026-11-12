"""Shared test fixtures: computer-generated pictures only (qc/sim.py), never a camera or the arm.

Changelog
  0.1.0  2026-11-12  first version.

Models in these tests use the hand-made features (fast, nothing to download). The pretrained ResNet-18
is tested in test_model.py only when models/backbone/resnet18_l2l3.onnx exists (tools/get_backbone.py).
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qc import background as bgmod  # noqa: E402
from qc import config as cfgmod  # noqa: E402
from qc import features, sim  # noqa: E402
from qc.model import NormalModel  # noqa: E402

BACKBONE_ONNX = os.path.join(ROOT, "models", "backbone", "resnet18_l2l3.onnx")


@pytest.fixture(scope="session")
def cfg():
    c = cfgmod.load(os.path.join(ROOT, "tests", "no-such-config.json"))
    c["backbone"] = "handcrafted"
    return c


@pytest.fixture(scope="session")
def empty_ref():
    return bgmod.EmptyReference.from_frames([sim.empty_frame(seed=100 + i) for i in range(10)])


@pytest.fixture(scope="session")
def hand():
    return features.HandcraftedBackbone()


_MODELS = {}


def trained(shape, face, cfg, ref, backbone, n=16, bank="local"):
    key = (shape, face, backbone.name, bank, n)
    if key not in _MODELS:
        c = dict(cfg, bank=bank)
        m = NormalModel(shape, face, c)
        frames = [sim.render(shape, face, seed=20000 + i)["frame"] for i in range(n)]
        m.fit(m.prepare(frames, ref), backbone)
        _MODELS[key] = m
    return _MODELS[key]


def scores(m, ref, frames, backbone):
    ims, ms = [], []
    for f in frames:
        loc = ref.part_mask(f)
        s = m.straighten(f, loc)
        ims.append(s["image"])
        ms.append(s["mask"])
    return m.score_canonical(ims, ms, backbone)[0]


@pytest.fixture(scope="session")
def square_model(cfg, empty_ref, hand):
    return trained("square", "top", cfg, empty_ref, hand)


def iou(a, b):
    a, b = a > 0, b > 0
    return (a & b).sum() / max(1, (a | b).sum())


np.set_printoptions(precision=3, suppress=True)
