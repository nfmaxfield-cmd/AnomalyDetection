# Files copied from Nick's robot project

Source: [nfmaxfield-cmd/alvik-fleet-capstone](https://github.com/nfmaxfield-cmd/alvik-fleet-capstone)
(MIT), commit `3f37317`, copied 2026-11-12 for the QC station.

| Here | There | Changed here |
|---|---|---|
| `tools/detector/` (all but `baseline_yolo.py` and its test) | same | nothing; YOLO baseline left out (AGPL-3.0) |
| `tools/label/` | same | nothing |
| `tools/recognition/` | same | `project.py`: QC preset (part + five defect types) added; Alvik preset only with the robot switch on |
| `tools/calibrate_floor.py`, `tools/tests/conftest.py`, `tools/tests/test_calibrate_floor.py` | same | nothing |
| `start-recognition.bat`, `track-with-model.bat` | repo root | nothing |
| `docs/data-formats.md` | same | nothing |

Every copied `.py` file says this in its first lines. HTML pages, Markdown files and the Windows launchers
are listed here instead, because their own tests check their first lines and the text people see.

**Robot features are kept and switched off.** Set `QC_ROBOT_FEATURES=1` (or `"robot_features": true` in
`qc_config.json`) to switch them back on. Tests that need parts of the robot project that were not copied
(experiment scripts, dock hardware files, recorded robot clips) are skipped with the reason
"robot feature (switched off)" (see `tools/conftest.py`).

Adapted (not copied whole) into the QC package: `qc/calib.py` (from `tools/recognition/floorcal.py`
rectangle mode), `qc/clock.py` (from `tools/recognition/outside.py` `estimate_time_offset`),
`tools/qclabel/` (the labelling studio's idea, rebuilt small for good/defect labels and defect boxes),
`qc/locate_detector.py` (wraps `tools/detector/infer.py`).
