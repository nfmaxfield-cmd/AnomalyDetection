# AnomalyDetection

Accept / reject decisions for a robot-arm quality-control station, learned from **good parts only**.

A robot arm with a suction cup puts each part on a fixed spot under a webcam, top face up; the arm then
flips it so the camera sees the bottom. This program looks at each face, answers **accept** or
**reject** to the arm, and shows *where* it found something suspicious (a heat map). Built by Nick
Maxfield (URI systems engineering) for Dr. Sodhi's Machine Vision team (Josh S and Chris). Free and
public under the MIT licence.

**Start here:** [docs/SETUP-GUIDE.md](docs/SETUP-GUIDE.md) (step by step, for the station).
First build's results and limits: [docs/build-reports/2026-11-12-first-build.md](docs/build-reports/2026-11-12-first-build.md).

## Premise

- Damaged parts are rare and every kind of damage looks different; good parts are plentiful and all look
  alike. So the program learns what **good** looks like and flags anything that differs, instead of
  learning each kind of damage from examples.
- That only works if "differs" means the part, not the room. So the scene is controlled: locked camera
  settings, a ring light, a matte black backdrop, a fixed spot. The one thing allowed to vary is the
  part's **angle**.
- The program must also know what the scene looks like **without** a part, to separate part from
  background from the very first picture and to notice when the backdrop or light changes.

## Approach

1. **Empty reference.** Pictures of the empty spot are averaged; how much they differ from each other is
   the camera's noise. A part is whatever differs from the empty spot by more than that noise
   (`qc/background.py`). The same comparison, away from the part, warns when the backdrop, light or
   camera has drifted.
2. **Capture.** When the arm says "paused", a short burst is taken and the sharpest still frame kept
   (`qc/capture.py`).
3. **Straighten.** The part's outline gives its symmetry (disc: round, square: 4-fold, hexagon: 6-fold;
   detected, or set per shape) and its angle; markings on the face (e.g. a stamp) decide between the
   symmetric angles; a fine search within +/-5 degrees lines it up exactly (`qc/straighten.py`).
4. **Normal model** (one per shape and face). A PatchCore-style memory: every 8 x 8 pixel patch of every
   good training picture is described by a pretrained ResNet-18 (ImageNet weights, run with ONNX
   Runtime: no PyTorch needed, fits a Raspberry Pi) and remembered. A new part's patches are compared
   with the same spot on the good parts; the distance to the nearest good patch is that patch's anomaly,
   the largest is the part's score, and the map of them is the heat map (`qc/features.py`, `qc/model.py`).
5. **Threshold from data**, never a picked number: fitted to the scores of good parts the model did not
   train on, and checked against 10-20 deliberately damaged parts per shape when available
   (`qc/threshold.py`). How many good pictures are enough is measured too: `learning-curve`.
6. **Hand-off and log.** A small JSON message back to the arm (HTTP on this computer by default; also
   stdin/stdout or a folder of files), and every decision logged with its pictures (`qc/handoff.py`,
   `qc/log.py`).

## Why this way

- **Good-only training** needs no collection of every defect type, and still catches defects nobody
  has seen yet. PatchCore (Roth et al., 2022) is among the strongest methods of this kind and needs no
  training of the network itself, only a memory of good patches, so it trains in seconds on a laptop.
- **ResNet-18 rather than Wide-ResNet-50** (the paper's choice): about 6x fewer weights (11.7 M vs 68.9 M)
  and several times less computation per picture, which matters on a Raspberry Pi; its patch features
  have 384 numbers instead of 1536 (both are cut to 128 here before they are stored). In a controlled
  scene the job is "is this spot
  different from the same spot on good parts", which the smaller network's mid-level features describe
  well; the bigger network can be switched on (`torch_arch`) and compared with the learning curve.
- **Straightening first** lets each patch be compared with the same spot on good parts. That catches a
  normal-looking feature in the wrong place, which a position-free memory would miss, and keeps the
  memory small.
- **The empty reference** makes the part's outline exact without any training, and turns background
  changes into a warning instead of silent false rejects.

## Quick start (no hardware: computer-generated pictures)

In a command window, in this folder (Python 3.10+):

```
python -m pip install -r requirements.txt
```

```
python tools/get_backbone.py
```

```
python -m qc collect-empty --simulate
```

```
python -m qc collect-good --simulate --shape square --n 60
```

```
python -m qc collect-damaged --simulate --shape square --n 10
```

```
python -m qc train --shape square
```

```
python -m qc run --simulate --parts 20
```

The last command starts the station and a pretend arm that brings 20 square parts (some damaged), prints
each decision, and saves the log in `runs/<date_time>/` (decisions.csv, pictures, heat maps).
Without `tools/get_backbone.py` the program still runs, with simpler hand-made features (it says so).

## Commands

`python -m qc <command>`; add `--simulate` to use computer-generated pictures and a pretend arm.

| Command | What it does |
|---|---|
| `collect-empty` | pictures of the empty spot -> the background reference |
| `collect-good` | good parts -> `data/good/<shape>/<face>/` (from the arm's "paused" signals, or `--manual`) |
| `collect-damaged` | deliberately damaged parts -> `data/damaged/<shape>/<face>/` |
| `train` | one model per shape and face, threshold from held-out data |
| `learning-curve` | how well damaged parts stand out vs. number of good training pictures |
| `calibrate-threshold` | threshold again (e.g. after adding damaged parts) |
| `run` | the live station: waits for the arm, answers each part |
| `inspect <image>` | judge one picture file, print the decision message |
| `camera-setup` | lock exposure, focus and white balance with sliders |
| `calibrate-mm` | (switch, off) millimetres per pixel from a card of known size |

## Results on computer-generated pictures

See the build report for the full table, method and limits. Summary (pretrained ResNet-18, 60 good
training pictures per shape and face; numbers filled from `docs/results/synthetic/`):

| defect type | scratch | chip | dent | missing corner | stain | good parts rejected |
|---|---|---|---|---|---|---|
| normal size | 60/60 | 60/60 | 60/60 | 60/60 | 60/60 | 1/240 |
| subtle (half size and contrast) | 56/60 | 60/60 | 59/60 | 60/60 | 60/60 | 1/240 |

Caught / tested over 3 shapes x 2 faces. The background-drift check flagged all 80 drifted pictures and
raised no false alarm on 45 normal ones.

These are **computer-generated** parts and defects: they show the software works end to end and what it
can tell apart in principle; real parts will be harder. Nothing here was tested with a real webcam, a
real arm or a Raspberry Pi yet.

## Layout

```
qc/            the station: sim, background, capture, straighten, features, model, threshold,
               handoff, log, station, train, cli (+ calib, clock, locate_detector: switched off)
tools/         get_backbone.py, evaluate_synthetic.py, arm_client_example.py, qclabel/ (labelling page),
               detector/, label/, recognition/, calibrate_floor.py  (copied from Nick's robot project,
               robot features switched off)
tests/         pytest, computer-generated pictures only
docs/          SETUP-GUIDE.md, build-reports/, results/, screenshots/
```

Kept but switched off (settings in `qc_config.json`, defaults in `qc/config.py`): finding the part with
the CentreNet detector instead of the empty reference (`locate_with_detector`), lining up recorded video
with an arm log on another clock (`clock_lineup`), defect sizes in mm (`px_to_mm`), and the robot
features of the copied labelling studio (`QC_ROBOT_FEATURES=1`).

## Tests

```
python -m pytest
```

The copied tools have their own suites (`python -m pytest tools/detector/tests tools/label/tests
tools/recognition/tests`, and `python -m pytest tools/tests`); tests that need parts of the robot project
not copied here are skipped with the reason "robot feature (switched off)".

## Credits and licences

MIT, copyright Nick Maxfield and contributors. The pretrained ResNet-18 comes from the ONNX Model Zoo
(Apache-2.0) and is downloaded by `tools/get_backbone.py`, not stored here. Method: PatchCore,
K. Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022.
`tools/detector`, `tools/label`, `tools/recognition` and `tools/calibrate_floor.py` come from
nfmaxfield-cmd/alvik-fleet-capstone (MIT); no AGPL code (Ultralytics YOLO) is included.
