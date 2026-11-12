# Synthetic results: onnx-resnet18 features, local memory bank, hard scene (qc 0.1.0)

Computer-generated pictures (qc/sim.py, hard scene, fixed seeds). Per shape/face: 60 good pictures to train, 20 good + 10 damaged (2 per type) to set the threshold, then 40 good test parts and 10 per defect type at each severity. Cells: caught / tested (heat map's hottest spot on the defect). "unsure": good parts the station's own checks would answer unsure (reject) although their score is below the line.

**Normal severity**

| shape / face | scratch | chip | dent | missing_corner | stain | false rejects (good) | unsure (good) |
|---|---|---|---|---|---|---|---|
| disc/top | 10/10 (10) | 10/10 (7) | 10/10 (10) | 10/10 (10) | 10/10 (10) | 6/40 | 0/40 |
| disc/bottom | 9/10 (9) | 10/10 (8) | 10/10 (10) | 10/10 (7) | 10/10 (10) | 0/40 | 0/40 |
| square/top | 7/10 (7) | 10/10 (8) | 5/10 (5) | 10/10 (5) | 10/10 (10) | 1/40 | 0/40 |
| square/bottom | 10/10 (10) | 10/10 (8) | 9/10 (8) | 10/10 (5) | 10/10 (10) | 0/40 | 0/40 |
| hexagon/top | 8/10 (8) | 10/10 (10) | 9/10 (9) | 8/10 (8) | 10/10 (10) | 3/40 | 0/40 |
| hexagon/bottom | 10/10 (10) | 10/10 (9) | 10/10 (10) | 10/10 (8) | 10/10 (10) | 1/40 | 0/40 |
| **all shapes** | **54/60** | **60/60** | **53/60** | **58/60** | **60/60** | **11/240** | **0/240** |

**Subtle severity** (half the normal size and contrast)

| shape / face | scratch | chip | dent | missing_corner | stain | false rejects (good) | unsure (good) |
|---|---|---|---|---|---|---|---|
| disc/top | 6/10 (1) | 10/10 (8) | 0/10 (0) | 10/10 (6) | 8/10 (8) | 6/40 | 0/40 |
| disc/bottom | 2/10 (2) | 8/10 (6) | 0/10 (0) | 4/10 (3) | 10/10 (10) | 0/40 | 0/40 |
| square/top | 0/10 (0) | 2/10 (1) | 0/10 (0) | 8/10 (4) | 10/10 (10) | 1/40 | 0/40 |
| square/bottom | 5/10 (4) | 8/10 (4) | 3/10 (3) | 10/10 (3) | 10/10 (10) | 0/40 | 0/40 |
| hexagon/top | 3/10 (2) | 10/10 (9) | 1/10 (1) | 5/10 (2) | 9/10 (9) | 3/40 | 0/40 |
| hexagon/bottom | 5/10 (4) | 10/10 (8) | 3/10 (1) | 6/10 (4) | 9/10 (8) | 1/40 | 0/40 |
| **all shapes** | **21/60** | **48/60** | **7/60** | **43/60** | **56/60** | **11/240** | **0/240** |

**Faint severity** (a quarter of the normal size and contrast)

| shape / face | scratch | chip | dent | missing_corner | stain | false rejects (good) | unsure (good) |
|---|---|---|---|---|---|---|---|
| disc/top | 2/10 (0) | 8/10 (5) | 1/10 (0) | 8/10 (4) | 0/10 (0) | 6/40 | 0/40 |
| disc/bottom | 0/10 (0) | 2/10 (0) | 0/10 (0) | 2/10 (2) | 0/10 (0) | 0/40 | 0/40 |
| square/top | 1/10 (1) | 0/10 (0) | 0/10 (0) | 0/10 (0) | 0/10 (0) | 1/40 | 0/40 |
| square/bottom | 0/10 (0) | 5/10 (1) | 1/10 (0) | 5/10 (4) | 1/10 (0) | 0/40 | 0/40 |
| hexagon/top | 1/10 (0) | 1/10 (1) | 1/10 (0) | 0/10 (0) | 2/10 (0) | 3/40 | 0/40 |
| hexagon/bottom | 2/10 (0) | 6/10 (6) | 1/10 (0) | 2/10 (0) | 0/10 (0) | 1/40 | 0/40 |
| **all shapes** | **6/60** | **22/60** | **4/60** | **17/60** | **3/60** | **11/240** | **0/240** |

**Background drift check** (warnings / pictures)

| case | empty spot | with a part |
|---|---|---|
| no drift (false alarms) | 0/30 | 0/15 |
| light_shift | 10/10 | 10/10 |
| stray_object | 10/10 | 10/10 |
| camera_bump | 10/10 | 10/10 |
| glare | 10/10 | 10/10 |

Timing on the build machine (2 CPU cores): training 42-109 s per shape/face, 1396-1991 ms per picture (straighten + score).
