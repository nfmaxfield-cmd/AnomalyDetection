# Synthetic results: onnx-resnet18 features, local memory bank, easy scene (qc 0.1.0)

Computer-generated pictures (qc/sim.py, easy scene, fixed seeds). Per shape/face: 60 good pictures to train, 20 good + 10 damaged (2 per type) to set the threshold, then 40 good test parts and 10 per defect type at each severity. Cells: caught / tested (heat map's hottest spot on the defect). "unsure": good parts the station's own checks would answer unsure (reject) although their score is below the line.

**Normal severity**

| shape / face | scratch | chip | dent | missing_corner | stain | false rejects (good) | unsure (good) |
|---|---|---|---|---|---|---|---|
| disc/top | 10/10 (10) | 10/10 (10) | 10/10 (10) | 10/10 (9) | 10/10 (10) | 0/40 | 0/40 |
| disc/bottom | 10/10 (10) | 10/10 (8) | 10/10 (10) | 10/10 (8) | 10/10 (10) | 0/40 | 0/40 |
| square/top | 10/10 (10) | 10/10 (10) | 10/10 (10) | 10/10 (3) | 10/10 (10) | 0/40 | 0/40 |
| square/bottom | 10/10 (10) | 10/10 (7) | 10/10 (10) | 10/10 (6) | 10/10 (10) | 0/40 | 0/40 |
| hexagon/top | 10/10 (10) | 10/10 (8) | 10/10 (10) | 10/10 (10) | 10/10 (10) | 1/40 | 0/40 |
| hexagon/bottom | 10/10 (10) | 10/10 (8) | 10/10 (10) | 10/10 (8) | 10/10 (10) | 0/40 | 0/40 |
| **all shapes** | **60/60** | **60/60** | **60/60** | **60/60** | **60/60** | **1/240** | **0/240** |

**Subtle severity** (half the normal size and contrast)

| shape / face | scratch | chip | dent | missing_corner | stain | false rejects (good) | unsure (good) |
|---|---|---|---|---|---|---|---|
| disc/top | 8/10 (8) | 10/10 (8) | 10/10 (10) | 10/10 (8) | 10/10 (10) | 0/40 | 0/40 |
| disc/bottom | 9/10 (9) | 10/10 (9) | 10/10 (10) | 10/10 (4) | 10/10 (10) | 0/40 | 0/40 |
| square/top | 10/10 (10) | 10/10 (3) | 10/10 (10) | 10/10 (2) | 10/10 (10) | 0/40 | 0/40 |
| square/bottom | 10/10 (10) | 10/10 (5) | 9/10 (9) | 10/10 (3) | 10/10 (10) | 0/40 | 0/40 |
| hexagon/top | 10/10 (10) | 10/10 (8) | 10/10 (10) | 10/10 (7) | 10/10 (10) | 1/40 | 0/40 |
| hexagon/bottom | 9/10 (9) | 10/10 (8) | 10/10 (10) | 10/10 (10) | 10/10 (10) | 0/40 | 0/40 |
| **all shapes** | **56/60** | **60/60** | **59/60** | **60/60** | **60/60** | **1/240** | **0/240** |

**Faint severity** (a quarter of the normal size and contrast)

| shape / face | scratch | chip | dent | missing_corner | stain | false rejects (good) | unsure (good) |
|---|---|---|---|---|---|---|---|
| disc/top | 1/10 (1) | 10/10 (5) | 0/10 (0) | 9/10 (6) | 0/10 (0) | 0/40 | 0/40 |
| disc/bottom | 8/10 (8) | 6/10 (2) | 4/10 (4) | 8/10 (5) | 6/10 (6) | 0/40 | 0/40 |
| square/top | 2/10 (2) | 4/10 (0) | 2/10 (2) | 9/10 (3) | 5/10 (5) | 0/40 | 0/40 |
| square/bottom | 6/10 (5) | 7/10 (2) | 4/10 (3) | 9/10 (5) | 3/10 (3) | 0/40 | 0/40 |
| hexagon/top | 6/10 (6) | 6/10 (6) | 3/10 (3) | 10/10 (7) | 6/10 (5) | 1/40 | 0/40 |
| hexagon/bottom | 5/10 (5) | 7/10 (7) | 0/10 (0) | 10/10 (5) | 1/10 (1) | 0/40 | 0/40 |
| **all shapes** | **28/60** | **40/60** | **13/60** | **55/60** | **21/60** | **1/240** | **0/240** |

**Background drift check** (warnings / pictures)

| case | empty spot | with a part |
|---|---|---|
| no drift (false alarms) | 0/30 | 0/15 |
| light_shift | 10/10 | 10/10 |
| stray_object | 10/10 | 10/10 |
| camera_bump | 10/10 | 10/10 |
| glare | 10/10 | 10/10 |

Timing on the build machine (2 CPU cores): training 47-142 s per shape/face, 1343-2416 ms per picture (straighten + score).
