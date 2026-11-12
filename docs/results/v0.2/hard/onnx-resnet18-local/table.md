# Synthetic results: onnx-resnet18 features, local memory bank, hard scene (qc 0.2.0)

Computer-generated pictures (qc/sim.py, hard scene, fixed seeds). Per shape/face: 60 good pictures to train, 20 good + 10 damaged (2 per type) to set the threshold, then 40 good test parts and 10 per defect type at each severity. Cells: caught / tested (heat map's hottest spot on the defect). "unsure": good parts the station's own checks would answer unsure (reject) although their score is below the line.

**Normal severity**

| shape / face | scratch | chip | dent | missing_corner | stain | false rejects (good) | unsure (good) |
|---|---|---|---|---|---|---|---|
| disc/top | 8/10 (8) | 10/10 (7) | 9/10 (9) | 10/10 (10) | 10/10 (10) | 0/40 | 0/40 |
| disc/bottom | 9/10 (8) | 10/10 (7) | 10/10 (10) | 10/10 (7) | 10/10 (10) | 0/40 | 0/40 |
| square/top | 7/10 (7) | 10/10 (10) | 6/10 (6) | 10/10 (5) | 10/10 (10) | 1/40 | 0/40 |
| square/bottom | 10/10 (10) | 10/10 (8) | 9/10 (8) | 10/10 (6) | 10/10 (10) | 0/40 | 0/40 |
| hexagon/top | 8/10 (8) | 10/10 (10) | 9/10 (9) | 7/10 (7) | 10/10 (10) | 1/40 | 0/40 |
| hexagon/bottom | 10/10 (10) | 10/10 (9) | 10/10 (10) | 10/10 (8) | 10/10 (10) | 4/40 | 0/40 |
| **all shapes** | **52/60** | **60/60** | **53/60** | **57/60** | **60/60** | **6/240** | **0/240** |

**Subtle severity** (half the normal size and contrast)

| shape / face | scratch | chip | dent | missing_corner | stain | false rejects (good) | unsure (good) |
|---|---|---|---|---|---|---|---|
| disc/top | 2/10 (0) | 10/10 (8) | 0/10 (0) | 10/10 (6) | 7/10 (7) | 0/40 | 0/40 |
| disc/bottom | 3/10 (3) | 9/10 (7) | 2/10 (2) | 4/10 (3) | 10/10 (10) | 0/40 | 0/40 |
| square/top | 0/10 (0) | 2/10 (0) | 0/10 (0) | 6/10 (4) | 10/10 (10) | 1/40 | 0/40 |
| square/bottom | 5/10 (4) | 9/10 (5) | 4/10 (4) | 9/10 (3) | 10/10 (10) | 0/40 | 0/40 |
| hexagon/top | 3/10 (2) | 10/10 (9) | 1/10 (1) | 3/10 (0) | 9/10 (9) | 1/40 | 0/40 |
| hexagon/bottom | 5/10 (4) | 10/10 (8) | 3/10 (1) | 6/10 (4) | 9/10 (9) | 4/40 | 0/40 |
| **all shapes** | **18/60** | **50/60** | **10/60** | **38/60** | **55/60** | **6/240** | **0/240** |

**Faint severity** (a quarter of the normal size and contrast)

| shape / face | scratch | chip | dent | missing_corner | stain | false rejects (good) | unsure (good) |
|---|---|---|---|---|---|---|---|
| disc/top | 0/10 (0) | 7/10 (5) | 0/10 (0) | 7/10 (4) | 0/10 (0) | 0/40 | 0/40 |
| disc/bottom | 0/10 (0) | 2/10 (0) | 0/10 (0) | 2/10 (2) | 0/10 (0) | 0/40 | 0/40 |
| square/top | 0/10 (0) | 0/10 (0) | 0/10 (0) | 0/10 (0) | 0/10 (0) | 1/40 | 0/40 |
| square/bottom | 0/10 (0) | 6/10 (1) | 1/10 (0) | 3/10 (2) | 2/10 (0) | 0/40 | 0/40 |
| hexagon/top | 1/10 (0) | 1/10 (1) | 1/10 (0) | 0/10 (0) | 1/10 (0) | 1/40 | 0/40 |
| hexagon/bottom | 3/10 (0) | 6/10 (6) | 4/10 (0) | 4/10 (0) | 1/10 (0) | 4/40 | 0/40 |
| **all shapes** | **4/60** | **22/60** | **6/60** | **16/60** | **4/60** | **6/240** | **0/240** |

**Background drift check** (warnings / pictures)

| case | empty spot | with a part |
|---|---|---|
| no drift (false alarms) | 0/30 | 0/15 |
| light_shift | 10/10 | 10/10 |
| stray_object | 10/10 | 10/10 |
| camera_bump | 10/10 | 10/10 |
| glare | 10/10 | 10/10 |

Timing on the build machine (2 CPU cores): training 96-187 s per shape/face, 1102-1875 ms per picture (straighten + score).
