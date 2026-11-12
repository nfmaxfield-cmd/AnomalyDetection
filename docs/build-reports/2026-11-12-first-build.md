# First build: good-parts-only anomaly detection for the QC arm (2026-11-12)

> Superseded in part by the audit fixes the same day: `docs/build-reports/2026-11-12-audit-fixes.md` (harder
> simulated scene, threshold rule, stillness, camera path, part ids). The numbers below are the first build's.

Built by Claude for Nick Maxfield; for Josh S and Chris (Dr. Sodhi's Machine Vision team, URI).
Everything below was run on **computer-generated pictures** only (no webcam, arm or Raspberry Pi).
Conclusions marked *(Claude's)* are mine, not yet the team's.

## Premise

The arm puts each part on a fixed spot under a fixed webcam (top face, then flipped to the bottom). The
software must answer accept/reject per face from **good parts only**, show where it saw a problem, set its
accept/reject line from data, and tell part from background from the first picture using a picture of
the empty spot. Agreed plan: PatchCore-style memory of good patches from a small pretrained CNN, one model
per shape and face; straighten by the outline and its symmetry, fine-align within +/-5 degrees; threshold
from held-out good scores plus 10-20 damaged parts; sharpest frame of a burst; JSON hand-off; full log.

## Data

**What was built** (`qc/` package, 19 Python files; command line `python -m qc ...`; details in Appendix B):
empty reference and drift check, burst capture with sharpest-and-still frame, outline symmetry and
straightening, ResNet-18 patch features through ONNX Runtime (no PyTorch), local and global memory banks,
Gumbel-fit threshold with damaged-part cost balance, learning-curve tool, HTTP / stdin-stdout / file
hand-off, decision log, simulator with a pretend arm, labelling page, and the robot project's tools copied
with their robot features switched off.

**Test set** (`tools/evaluate_synthetic.py`, fixed seeds): 3 shapes x 2 faces. Per shape/face: 60 good
pictures to train; 20 good + 10 damaged (2 per type) to set the threshold; then 40 good test parts and 10
parts per defect type at normal severity and 10 at "subtle" severity (half the size and contrast).

**Result, default settings** (pretrained ResNet-18, local bank). Cell = caught / tested, all 6 shape-faces
together; full per-shape tables in Appendix A.

| defect | scratch | chip | dent | missing corner | stain | good parts wrongly rejected |
|---|---|---|---|---|---|---|
| normal | 60/60 | 60/60 | 60/60 | 60/60 | 60/60 | 1/240 |
| subtle | 56/60 | 60/60 | 59/60 | 60/60 | 60/60 | (same 1/240) |

The heat map's hottest spot lay on the defect (within 12 px) for 275 of 300 normal and 250 of 300 subtle
defects; the misses are almost all chips and missing corners, where the heat covers a wider area than the
cut (picture: `docs/screenshots/2026-11-12/heatmap_square_top_missing_corner.jpg`).

**Choice of memory bank and features, decided by this data** (same test set, subtle defects caught, of 300):

| features / bank | subtle caught | normal caught | good rejected | train per shape-face | per picture |
|---|---|---|---|---|---|
| ResNet-18 / local (default) | 295 | 300 | 1/240 | 13-40 s | 0.4-0.7 s |
| ResNet-18 / global (classic PatchCore) | 266 | 299 | 1/240 | 84-174 s | 0.13-0.35 s |
| hand-made / local (fallback, no download) | 166 | 278 | 34/240 | 8-35 s | 0.13-0.38 s |

<sub>Times on the 2-core build machine while other jobs shared it, so they are upper values.</sub>

**Background drift check:** no false alarms on 30 normal empty frames and 15 part pictures; every one of
40 drifted empty frames and 40 drifted part pictures (light moved, stray screw, camera bumped 6-10 px,
glare) was flagged. **Straightening:** symmetry found correctly for every tested disc (round), square
(4-fold) and hexagon (6-fold); straightened angles agree within 1 degree up to the symmetry; a stamp on the
face settles the full turn within 3 degrees (tests in `tests/test_straighten.py`). **Burst:** the frame kept
was a settled one in 9 of 9 simulated bursts; a part sliding 3 px between frames is flagged as moving.

**How many good pictures:** learning curve on square/top (110 good, 15 subtle damaged parts), Appendix C:
the curve is **flat**: the median share of damaged parts caught at the good-only
threshold is 93-100 % at every size from 5 to 80 good pictures, and the swings between random draws
(up to 2 of 15 parts) are as large as any trend. AUROC is 0.99-1.00 throughout. So on these simulated
parts even 5 good pictures suffice; *(Claude's)* this says the simulated parts vary too little to answer
Nick's question, not that 5 real parts are enough. The estimate of 50-100 per shape and face stands until
the tool is run on real pictures.

**Tests:** QC suite 66 passed (including the pretrained-backbone test); copied tools 392 passed, 39 skipped (robot parts not copied, PyTorch), 2 expected failures (Appendix D). The README quick start ran unchanged in a fresh clone (picture: `docs/screenshots/2026-11-12/quickstart_terminal_output.png`).

## Conclusion

1. The pipeline works end to end on the simulated station: the pretend arm gets a decision per face and a
   part decision, every decision is logged with its pictures, and simulated damage of all five types is
   caught with about 1 in 240 good parts rejected.
2. *(Claude's)* Keep **ResNet-18 with the local (same-spot) bank** as the default: it caught 29 more of the
   300 subtle defects than the classic global bank at the same false-reject count, and needs no PyTorch.
   The hand-made fallback is for testing only (34 of 240 good parts rejected).
3. *(Claude's)* The synthetic defects are easy for the pretrained features; the numbers say the software
   is sound, **not** how it will do on the team's real parts. The real test is the learning curve and the
   10-20 damaged parts per shape on the actual station.

## What's next

1. Josh and Chris set up the station (docs/SETUP-GUIDE.md), collect empty + 50-100 good parts per shape and
   face, run `learning-curve`, then 10-20 damaged parts per shape and `calibrate-threshold`.
2. Real-camera checks the simulator can't make: `camera-setup` on their webcam, the stillness limit with a
   real arm, part shadows near the drift check, shot noise on bright parts.
3. Mirror views (edges) are not inspected yet: add fixed mirror regions with their own models once the
   mirror positions are known.
4. Try the Raspberry Pi: `tools/get_backbone.py --int8` makes an 8-bit backbone (2.8 MB); time it there.
5. Joint evaluation with Dr. Sodhi on a held-back set of real parts that nobody trained or tuned on.

## Sweep 1: new tests that may now be needed

- Real webcam: locked settings stay locked after a restart (some drivers reset them); `camera-setup` sliders
  move the values on their camera.
- Real arm: the "paused" timing (is the part still when the burst starts?); a full session with
  `empty_due` round trips; the arm code's handling of `needs_attention`.
- Shadows or reflections of the part on the backdrop near its edge triggering the drift warning on part
  pictures (simulated parts cast no shadow).
- Parts that are not quite flat, or land tilted: straightening assumes a top-down view of a flat face.
- Very small parts (outline symmetry needs a long enough outline; the code refuses tiny outlines).
- A real damaged-part set per shape: does the cost-balance rule move the line sensibly, and how many
  damaged parts make the threshold stable (a learning curve on the damaged set).
- Long-run drift: scores of accepted parts creeping up over a day (a score-drift warning is not built yet).
- PyTorch backbone (`backbone: torch`, Wide-ResNet-50) and the copied detector training: untested here.

## Sweep 2: possible inaccuracies in what this relied on

- **Simulated parts and defects** (proxy for everything): flat grey parts, ideal ring light, no shadows,
  defects drawn by formula. Real scratches on real surfaces may look more like normal texture.
- **"Subtle" severity** is our definition (half size and contrast), not a measured real-defect size.
- **Gumbel fit** for the good-score tail with only 20 held-out parts: extrapolates to the 1 % target
  false-reject rate; the real rate may differ. A DOE is not the tool; more held-out parts are.
- **Target false-reject rate 1 % and cost ratio 10:1** are business choices typed in as defaults; the team
  should set them (questions for Nick).
- Chosen numbers in the code, marked `[chosen]` (full list in Appendix E). Those where a **designed
  experiment should decide** on the real station: burst length and settle time, the stillness factor (3x
  empty-frame change), the drift margin (1.5x), crop margin, patch smoothing, the projection to 128 numbers,
  the local search window, the extra training rotations (+/-3 degrees), and the number of empty frames.
  Those that need no DOE (method constants): Otsu/noise mask rule, harmonic detection limits (checked on
  all simulated shapes), grain blur and high-pass widths for alignment.
- **Timings** were taken while other jobs used the same 2 CPU cores; they are upper values, not a benchmark.
- **The localisation count** ("hottest spot within 12 px of the defect") is a strict proxy; it undercounts
  edge defects whose heat spreads.
- **Learning curve** with 5 random draws and 15 damaged parts has coarse steps (1 part = 6.7 %).
- **Pretrained weights** are the ONNX Model Zoo's ResNet-18 v1 (converted from MXNet), not torchvision's;
  features differ slightly from the PatchCore paper's setup.

---

## Appendix A. Full synthetic tables

Default (ResNet-18, local bank): `docs/results/synthetic/onnx-resnet18-local/table.md`; the others in
`docs/results/synthetic/onnx-resnet18-global/` and `.../handcrafted-local/`; raw scores in each
`results.json`. Score charts: `.../onnx-resnet18-local/scores_square_top.png` (and disc/bottom, hexagon/top).

Computer-generated pictures (qc/sim.py, fixed seeds). Per shape/face: 60 good pictures to train, 20 good + 10 damaged (2 per type) to set the threshold, then 40 good test parts and 10 per defect type at each severity. Cells: caught / tested (heat map's hottest spot on the defect).

**Normal severity**

| shape / face | scratch | chip | dent | missing_corner | stain | false rejects (good) |
|---|---|---|---|---|---|---|
| disc/top | 10/10 (10) | 10/10 (10) | 10/10 (10) | 10/10 (9) | 10/10 (10) | 0/40 |
| disc/bottom | 10/10 (10) | 10/10 (8) | 10/10 (10) | 10/10 (8) | 10/10 (10) | 0/40 |
| square/top | 10/10 (10) | 10/10 (10) | 10/10 (10) | 10/10 (3) | 10/10 (10) | 0/40 |
| square/bottom | 10/10 (10) | 10/10 (7) | 10/10 (10) | 10/10 (6) | 10/10 (10) | 0/40 |
| hexagon/top | 10/10 (10) | 10/10 (8) | 10/10 (10) | 10/10 (10) | 10/10 (10) | 1/40 |
| hexagon/bottom | 10/10 (10) | 10/10 (8) | 10/10 (10) | 10/10 (8) | 10/10 (10) | 0/40 |
| **all shapes** | **60/60** | **60/60** | **60/60** | **60/60** | **60/60** | **1/240** |

**Subtle severity** (half the normal size and contrast)

| shape / face | scratch | chip | dent | missing_corner | stain | false rejects (good) |
|---|---|---|---|---|---|---|
| disc/top | 8/10 (8) | 10/10 (8) | 10/10 (10) | 10/10 (8) | 10/10 (10) | 0/40 |
| disc/bottom | 9/10 (9) | 10/10 (9) | 10/10 (10) | 10/10 (4) | 10/10 (10) | 0/40 |
| square/top | 10/10 (10) | 10/10 (3) | 10/10 (10) | 10/10 (2) | 10/10 (10) | 0/40 |
| square/bottom | 10/10 (10) | 10/10 (5) | 9/10 (9) | 10/10 (3) | 10/10 (10) | 0/40 |
| hexagon/top | 10/10 (10) | 10/10 (8) | 10/10 (10) | 10/10 (7) | 10/10 (10) | 1/40 |
| hexagon/bottom | 9/10 (9) | 10/10 (8) | 10/10 (10) | 10/10 (10) | 10/10 (10) | 0/40 |
| **all shapes** | **56/60** | **60/60** | **59/60** | **60/60** | **60/60** | **1/240** |

**Background drift check** (warnings / pictures)

| case | empty spot | with a part |
|---|---|---|
| no drift (false alarms) | 0/30 | 0/15 |
| light_shift | 10/10 | 10/10 |
| stray_object | 10/10 | 10/10 |
| camera_bump | 10/10 | 10/10 |
| glare | 10/10 | 10/10 |

Timing on the build machine (2 CPU cores): training 13-40 s per shape/face, 408-664 ms per picture (straighten + score).

## Appendix B. How it works (lightly technical)

- **Empty reference** (`qc/background.py`): mean of N empty frames; noise limit = largest 99.9 % pixel
  difference between each empty frame and the mean of the others (leave-one-out). Part mask = difference
  above both that limit and Otsu's split of the current frame; largest blob, holes filled.
  <sub>Otsu's method picks the brightness split that best separates a picture's two groups of pixels.</sub>
- **Drift**: smoothed difference outside the part (+15 px); limit 1.5x the largest change seen between the
  empty frames themselves; also the overall brightness change.
- **Straightening** (`qc/straighten.py`): radius of the outline at 360 angles; its Fourier harmonics give
  the symmetry order (smallest strong harmonic) and the angle (phase). Candidates (each symmetric angle, or
  the whole turn for a round part in 4-degree steps) are compared on the inside of the part only, with
  ring-shaped content removed so only angle-telling marks count; then +/-5 degrees in 0.5-degree steps with
  phase correlation for the shift. The reference is refined twice (first picture, then the mean).
- **Features** (`qc/features.py`): ResNet-18 layer2 (28 x 28) + layer3 (upsampled), 384 numbers per 8 x 8 px
  cell, 3 x 3 neighbourhood average; standardised and projected to 128 numbers (`qc/model.py`).
- **Local bank**: each cell compared with the same cell +/-1 on every training picture (and its +/-3 degree
  rotations); score = smoothed map's maximum inside the part (+2 cells). **Global bank**: classic PatchCore
  coreset (10 %), nearest neighbour anywhere.
- **Threshold** (`qc/threshold.py`): Gumbel fit to held-out good scores, line at the 99th percentile (target
  1 %), never below the highest held-out good score; if damaged parts score below it, minimise
  10 x (damaged accepted) + 1 x (good rejected).
  <sub>Gumbel: the usual distribution for the largest of many values, as a part's score is the largest
  of its cells' scores.</sub> <sub>AUROC (learning curve): the chance that a randomly picked damaged part
  scores higher than a randomly picked good one; 1.0 = always.</sub>
- **Hand-off** (`qc/handoff.py`): `POST /paused`, `/empty`, `GET /health` on 127.0.0.1:8765; or JSON lines
  on stdin/stdout; or `inbox/*.json` -> `outbox/*.json`. Message fields in the module's docstring.

## Appendix C. Learning curve

Square, top face, simulated: 110 good pictures (22 held out, 88 to draw from), 15 subtle damaged parts
(3 of each type at half size and contrast), pretrained ResNet-18, local bank, 5 random draws per size.

| good pictures | caught at the good-only threshold: median (lowest-highest draw) | AUROC median (lowest) |
|---|---|---|
| 5 | 0.933 (0.867-1.000) | 0.994 (0.970) |
| 10 | 0.933 (0.867-1.000) | 0.988 (0.982) |
| 20 | 1.000 (0.933-1.000) | 1.000 (1.000) |
| 40 | 0.933 (0.867-1.000) | 0.997 (0.961) |
| 80 | 1.000 (0.933-1.000) | 1.000 (0.997) |

The tool's rule: the smallest size from which every larger size's median stays within the typical
draw-to-draw spread of the best median. Here that spread is 0.133 (2 parts of 15), so it answers 5 and adds
that the curve is flat within the spread. Chart: `docs/screenshots/2026-11-12/learning_curve_square_top_synthetic.png`.
The dip at 40 is two draws that each miss 2 of the 15 subtle parts; with 15 damaged parts one miss moves
the share by 6.7 %, which is why more damaged parts make this measure finer.

## Appendix D. Copied components and their tests

From nfmaxfield-cmd/alvik-fleet-capstone @3f37317 (list in `tools/COPIED.md`): the CentreNet detector
(`tools/detector`, without the AGPL YOLO baseline; optional part finder, switch `locate_with_detector`),
the labelling studio (`tools/label`, `tools/recognition`; a QC preset added, robot preset behind
`QC_ROBOT_FEATURES`), floor calibration (`tools/calibrate_floor.py`). Adapted into `qc/`: clock line-up
(`qc/clock.py`, from `outside.estimate_time_offset`), px-to-mm (`qc/calib.py`, from `floorcal.py`
rectangle mode). New small QC labelling page (`tools/qclabel`).

Test runs in a fresh clone (2026-11-12, Python 3.13, OpenCV 5.0, no PyTorch):

| suite | result |
|---|---|
| `python -m pytest` (QC) | 66 passed |
| `python -m pytest tools/detector/tests tools/label/tests tools/recognition/tests` | 378 passed, 33 skipped, 2 expected failures |
| `python -m pytest tools/tests` | 14 passed, 6 skipped |

Skipped: tests needing PyTorch (3 detector test files not collected, 2 tests), parts of the robot project not
copied (experiment scripts, dock files, fleet server, recorded clips), 4 tests written for the robot
preset list (they pass with `QC_ROBOT_FEATURES=1`), and slow or video-dependent tests the source also skips.
Expected failures: `test_augmented_picture_and_heading_agree` fails the same way in the source repository
in this environment (checked); the other is marked so in the source. The copy itself changed no behaviour
except the studio's preset list.

## Appendix E. Chosen numbers

| setting | value | where | DOE? |
|---|---|---|---|
| burst_frames / burst_settle_s | 8 / 0 s | config | yes: sharpness and stillness vs. cycle time |
| STILL_FACTOR | 3 x empty-frame change | capture.py | yes: false "moving" vs. missed motion on the real arm |
| DRIFT_FACTOR | 1.5 x | background.py | yes: false drift alarms over a day vs. caught drift |
| empty_frames / empty_spacing_s / empty_every_parts | 10 / 0.3 s / 50 | config | yes (empty_every_parts from how fast drift appears) |
| crop_margin | 1.15 | config | yes, small effect expected |
| smooth_sigma | 2 cells | config | yes: localisation vs. small-defect sensitivity |
| PROJ_DIM | 128 | model.py | yes: 64 / 128 / 256 vs. speed |
| local_window | +/-1 cell | config | yes, tied to alignment accuracy |
| train_rotations_deg | +/-3 deg | config | yes |
| coreset_fraction (global bank) | 10 % | config | only if the global bank is used |
| fine_align_deg / step | 5 / 0.5 deg | config | no (from the agreed plan; step is a resolution) |
| target_false_reject | 1 % | config | no: a business choice |
| cost_bad_accepted : cost_good_rejected | 10 : 1 | config | no: a business choice |
| holdout_fraction | 20 % (at least 5) | config | no: more held-out parts is always better |
| SIG_FACTOR / MIN_STRENGTH (symmetry) | 8x noise / 0.4 % | straighten.py | no: checked on every simulated shape |
| HIGHPASS / GRAIN / INTERIOR_ERODE | 6 / 1 / 9 px | straighten.py | no unless real parts misalign |
| MIN_PART_AREA, EXCLUDE_GROW | 400 px, 15 px | background.py | EXCLUDE_GROW yes (shadow width on the real backdrop) |
