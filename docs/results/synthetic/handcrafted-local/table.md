# Synthetic results: handcrafted features, local memory bank

Computer-generated pictures (qc/sim.py, fixed seeds). Per shape/face: 60 good pictures to train, 20 good + 10 damaged (2 per type) to set the threshold, then 40 good test parts and 10 per defect type at each severity. Cells: caught / tested (heat map's hottest spot on the defect).

**Normal severity**

| shape / face | scratch | chip | dent | missing_corner | stain | false rejects (good) |
|---|---|---|---|---|---|---|
| disc/top | 7/10 (7) | 10/10 (10) | 8/10 (7) | 10/10 (10) | 10/10 (10) | 8/40 |
| disc/bottom | 9/10 (8) | 10/10 (8) | 10/10 (10) | 10/10 (5) | 10/10 (10) | 9/40 |
| square/top | 5/10 (5) | 10/10 (9) | 10/10 (10) | 10/10 (10) | 10/10 (10) | 3/40 |
| square/bottom | 7/10 (7) | 10/10 (10) | 10/10 (10) | 10/10 (10) | 10/10 (10) | 5/40 |
| hexagon/top | 5/10 (5) | 10/10 (9) | 9/10 (9) | 10/10 (10) | 10/10 (10) | 2/40 |
| hexagon/bottom | 8/10 (8) | 10/10 (8) | 10/10 (10) | 10/10 (10) | 10/10 (10) | 7/40 |
| **all shapes** | **41/60** | **60/60** | **57/60** | **60/60** | **60/60** | **34/240** |

**Subtle severity** (half the normal size and contrast)

| shape / face | scratch | chip | dent | missing_corner | stain | false rejects (good) |
|---|---|---|---|---|---|---|
| disc/top | 1/10 (0) | 6/10 (5) | 0/10 (0) | 10/10 (8) | 9/10 (7) | 8/40 |
| disc/bottom | 4/10 (3) | 5/10 (4) | 2/10 (0) | 9/10 (6) | 9/10 (9) | 9/40 |
| square/top | 1/10 (1) | 9/10 (6) | 1/10 (0) | 10/10 (3) | 9/10 (9) | 3/40 |
| square/bottom | 4/10 (3) | 7/10 (4) | 2/10 (0) | 10/10 (3) | 9/10 (9) | 5/40 |
| hexagon/top | 1/10 (1) | 0/10 (0) | 1/10 (0) | 8/10 (8) | 5/10 (5) | 2/40 |
| hexagon/bottom | 4/10 (4) | 9/10 (6) | 2/10 (1) | 10/10 (10) | 9/10 (9) | 7/40 |
| **all shapes** | **15/60** | **36/60** | **8/60** | **57/60** | **50/60** | **34/240** |

**Background drift check** (warnings / pictures)

| case | empty spot | with a part |
|---|---|---|
| no drift (false alarms) | 0/30 | 0/15 |
| light_shift | 10/10 | 10/10 |
| stray_object | 10/10 | 10/10 |
| camera_bump | 10/10 | 10/10 |
| glare | 10/10 | 10/10 |

Timing on the build machine (2 CPU cores): training 8-35 s per shape/face, 129-380 ms per picture (straighten + score).
