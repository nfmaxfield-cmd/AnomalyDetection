"""Patch features: what each small square of the straightened picture "looks like", as a list of numbers.

Changelog
  0.1.0  2026-11-12  first version: ONNX ResNet-18 (default), torchvision ResNet-18 / Wide-ResNet-50
                     (optional), hand-made features (fallback that needs nothing to download).

Three ways to get them, all giving a (N, C, 28, 28) array for N canonical 224 x 224 pictures, one
feature vector per 8 x 8 pixel cell, each already averaged with its 3 x 3 neighbours as in PatchCore:

  onnx         ResNet-18 trained on ImageNet, cut after layer2 and layer3 (tools/get_backbone.py makes
               the file). Runs with onnxruntime only: no PyTorch, so it fits a Raspberry Pi.
  torch        the same from torchvision (resnet18 or wide_resnet50_2). Needs PyTorch; NOT tested in
               the first build (PyTorch could not be installed where it was built).
  handcrafted  colour, gradients and fine detail at three scales. Needs nothing; weaker on subtle
               texture defects. Used when no backbone file is present, with a warning.

Why ResNet-18 by default and not Wide-ResNet-50 (the PatchCore paper's choice): about 11.7 M against
68.9 M weights and several times less computation per picture (it matters on a Raspberry Pi), and
layer2+layer3 features of 384 numbers against 1536 (model.py projects either to 128 before storing). In a controlled scene (fixed light, black backdrop, straightened parts) the
remaining job is "is this spot different from the same spot on good parts", which the smaller network's
mid-level features already describe; the bigger network's advantage on the varied MVTec benchmark comes
largely from harder, uncontrolled images. The learning-curve tool can compare the two on real parts.
"""
import os

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
GRID = 28


def _neigh_avg(f):
    """3 x 3 neighbourhood average of every cell (PatchCore's local aggregation). f: (N, C, h, w).
    Plain numpy (edges mirrored), because OpenCV's filters take only a limited number of channels."""
    p = np.pad(f, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="reflect")
    h, w = f.shape[2:]
    out = np.zeros_like(f)
    for dy in range(3):
        for dx in range(3):
            out += p[:, :, dy:dy + h, dx:dx + w]
    return out / 9.0


def _upsample(x, size):
    """Bilinear resize of (N, C, h, w) to (N, C, size, size), half-pixel centres (like PyTorch's
    align_corners=False)."""
    def axis(n_in):
        src = (np.arange(size) + 0.5) * n_in / size - 0.5
        src = np.clip(src, 0, n_in - 1)
        i0 = np.floor(src).astype(int)
        i1 = np.minimum(i0 + 1, n_in - 1)
        return i0, i1, (src - i0).astype(np.float32)
    y0, y1, wy = axis(x.shape[2])
    x0, x1, wx = axis(x.shape[3])
    top = x[:, :, y0] * (1 - wy)[None, None, :, None] + x[:, :, y1] * wy[None, None, :, None]
    return top[:, :, :, x0] * (1 - wx) + top[:, :, :, x1] * wx


def _to_input(imgs):
    x = np.stack([cv2.cvtColor(cv2.resize(i, (224, 224)), cv2.COLOR_BGR2RGB) for i in imgs]).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(0, 3, 1, 2).copy()


def _merge(l2, l3):
    """layer3 (14 x 14) brought up to layer2's 28 x 28 and stacked on it."""
    up = _upsample(l3.astype(np.float32), GRID)
    return np.concatenate([l2, up], axis=1).astype(np.float32)


class OnnxBackbone:
    name = "onnx-resnet18"

    def __init__(self, path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, (os.cpu_count() or 2))
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.path = path

    def __call__(self, imgs, batch=8):
        out = []
        for k in range(0, len(imgs), batch):
            x = _to_input(imgs[k:k + batch])
            l2, l3 = self.sess.run(None, {self.inp: x})
            out.append(_neigh_avg(_merge(l2, l3)))
        return np.concatenate(out)


class TorchBackbone:
    """torchvision ResNet; NOT tested in the first build (no PyTorch in the build sandbox)."""

    def __init__(self, arch="resnet18"):
        import torch
        import torchvision
        self.torch = torch
        weights = {"resnet18": "IMAGENET1K_V1", "wide_resnet50_2": "IMAGENET1K_V1"}[arch]
        m = getattr(torchvision.models, arch)(weights=weights).eval()
        self.body = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2)
        self.l3 = m.layer3
        self.name = "torch-" + arch

    def __call__(self, imgs, batch=8):
        out = []
        with self.torch.no_grad():
            for k in range(0, len(imgs), batch):
                x = self.torch.from_numpy(_to_input(imgs[k:k + batch]))
                l2 = self.body(x)
                l3 = self.l3(l2)
                out.append(_neigh_avg(_merge(l2.numpy(), l3.numpy())))
        return np.concatenate(out)


class HandcraftedBackbone:
    """Fallback: per 8 x 8 cell, the mean and spread of colour (Lab) and of gradient / Laplacian
    responses at three blur scales. 36 numbers per cell; no download needed."""
    name = "handcrafted"

    def __call__(self, imgs, batch=None):
        out = []
        for im in imgs:
            im = cv2.resize(im, (224, 224))
            lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
            g = lab[..., 0]
            chans = [lab[..., 0], lab[..., 1] * 2, lab[..., 2] * 2]
            for s in (1.0, 2.0, 4.0):
                gs = cv2.GaussianBlur(g, (0, 0), s)
                gx = cv2.Sobel(gs, cv2.CV_32F, 1, 0, ksize=3) * s
                gy = cv2.Sobel(gs, cv2.CV_32F, 0, 1, ksize=3) * s
                lap = cv2.Laplacian(gs, cv2.CV_32F, ksize=3) * s * s
                chans += [gx, gy, np.hypot(gx, gy), lap]
            # 3 colour + 12 filters = 15 maps; per cell mean, spread and max-abs of the filters
            feats = []
            for k, c in enumerate(chans):
                cell = c.reshape(GRID, 8, GRID, 8)
                feats.append(cell.mean(axis=(1, 3)))
                feats.append(cell.std(axis=(1, 3)))
                if k >= 3 and (k - 3) % 4 in (2, 3):
                    feats.append(np.abs(cell).max(axis=(1, 3)))
            out.append(np.stack(feats).astype(np.float32))
        return _neigh_avg(np.stack(out))


def make(cfg, root="."):
    """The backbone the settings ask for. 'auto': ONNX file if present, else PyTorch if installed,
    else hand-made features (with a warning in .warning)."""
    want = cfg.get("backbone", "auto")
    path = cfg.get("backbone_onnx", "")
    if path and not os.path.isabs(path):
        path = os.path.join(root, path)
    if want in ("auto", "onnx") and path and os.path.exists(path):
        return OnnxBackbone(path)
    if want == "onnx":
        raise FileNotFoundError("Backbone file %s is missing: run  python tools/get_backbone.py" % path)
    if want in ("auto", "torch"):
        try:
            return TorchBackbone(cfg.get("torch_arch", "resnet18"))
        except Exception as e:      # noqa: BLE001 - any failure means: not available here
            if want == "torch":
                raise
            hb = HandcraftedBackbone()
            hb.warning = ("No pretrained backbone found (%s; %s). Using hand-made features: fine for "
                          "testing, weaker on subtle defects. Run  python tools/get_backbone.py" % (path, e))
            return hb
    return HandcraftedBackbone()
