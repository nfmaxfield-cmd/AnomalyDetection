#!/usr/bin/env python3
"""Download the pretrained ResNet-18 once and cut it down to the part the anomaly detector uses.

Changelog
  0.1.0  2026-11-12  first version.

    python tools/get_backbone.py                       # -> models/backbone/resnet18_l2l3.onnx
    python tools/get_backbone.py --from my_resnet18.onnx   # use a file you already have (no download)
    python tools/get_backbone.py --int8                # also an 8-bit copy (smaller, faster on a Pi)

Source: the ONNX Model Zoo's ResNet-18 v1 (ImageNet), resnet18-v1-7.onnx (Apache-2.0). The file is
checked against its SHA-256 so a broken or different download is refused. The network is cut after
its second and third stages (the "layer2" and "layer3" outputs PatchCore uses), which drops about two
thirds of the weights: the result is about 11 MB.

Needs:  pip install onnx onnxruntime   (both in requirements.txt)
"""
import argparse
import hashlib
import os
import sys
import urllib.request

URL = ("https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/"
       "resnet/model/resnet18-v1-7.onnx")
SHA256 = "4e8f8653e7a2222b3904cc3fe8e304cd8b339ce1d05fd24688162f86fb6df52c"
OUTPUTS = ["resnetv15_stage2_activation1", "resnetv15_stage3_activation1"]
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_OUT = os.path.join(ROOT, "models", "backbone", "resnet18_l2l3.onnx")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cut(src, dst):
    import onnx
    from onnx import utils
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    # the zoo file is an old (IR 3) model that lists every weight as an input; the cut keeps only "data"
    # as input, which needs IR 4 or newer to be valid
    utils.extract_model(src, dst, ["data"], OUTPUTS, check_model=False)
    m = onnx.load(dst)
    m.ir_version = max(m.ir_version, 4)
    # let any batch size through
    m.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "N"
    for o in m.graph.output:
        o.type.tensor_type.shape.dim[0].dim_param = "N"
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    return dst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", help="an already downloaded resnet18-v1-7.onnx")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--int8", action="store_true", help="also write an 8-bit quantised copy (*.int8.onnx)")
    a = ap.parse_args(argv)
    src = a.src
    if not src:
        src = os.path.join(os.path.dirname(a.out), "resnet18-v1-7.onnx")
        if not os.path.exists(src):
            os.makedirs(os.path.dirname(src), exist_ok=True)
            print("Downloading ResNet-18 (45 MB) ...")
            urllib.request.urlretrieve(URL, src)
    got = sha256(src)
    if got != SHA256:
        print("The file %s is not the expected ResNet-18 (checksum %s). Delete it and try again." % (src, got))
        return 2
    cut(src, a.out)
    print("Saved %s (%.1f MB)" % (a.out, os.path.getsize(a.out) / 1e6))
    if a.int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        q = a.out.replace(".onnx", ".int8.onnx")
        quantize_dynamic(a.out, q, weight_type=QuantType.QUInt8)
        print("Saved %s (%.1f MB)" % (q, os.path.getsize(q) / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
