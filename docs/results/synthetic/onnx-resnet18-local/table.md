# Synthetic results: onnx-resnet18 features, local memory bank

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
