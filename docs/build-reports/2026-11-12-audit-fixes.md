# Audit fixes, a harder test scene, and the checkerboard calibration card (2026-11-12)

Built by Claude for Nick Maxfield; for Josh S and Chris (Dr. Sodhi's Machine Vision team, URI). Version
0.2.0. Everything was run on **computer-generated pictures** and pretend hardware only (no webcam, arm or
Raspberry Pi). Conclusions marked *(Claude's)* are mine, not yet the team's.

## Premise

An independent audit of the first build (commit `7878be2`) found 7 high, 9 medium and 8 low findings. In
short: the station could answer "accept" for a picture it should not trust (moving part, changed backdrop);
the threshold rule with damaged parts could scrap half the good parts because of one odd picture; the
simulator was easy, so the published results were flattering; the real-camera path could feed old
buffered frames and never checked its settings; part ids could carry an old reject; "empty" accepted any
scene; and the install failed on Python 3.10/3.11. This build fixes every high and medium finding and most
low ones, and adds Nick's extra request: a printed checkerboard card for a one-time lens and scale
calibration.

Decisions taken from Nick: 1 % good-part scrap is the starting target (to be confirmed with the team); the
10:1 cost ratio stays a placeholder `[chosen]`; the white corner dots and the "largest part a third to half
of the picture width" advice stay in the guide; mirror views wait for a later build.

## Data

**1. Results, both simulated scenes, before and after** (pretrained ResNet-18, local bank, 60 good training
pictures, 20 good + 10 damaged to set the threshold, 40 good test parts and 10 per defect type and severity,
per shape and face; 3 shapes x 2 faces; "before" = the first build's code on the same pictures):

| scene | code | normal caught | subtle caught | faint caught | good parts scrapped (by score + answered unsure) |
|---|---|---|---|---|---|
| easy | first build | 300/300 (100 %) | 295/300 (98 %) | 157/300 (52 %) | 1/240 (1 + 0) |
| easy | **now** | **300/300 (100 %)** | **295/300 (98 %)** | **155/300 (52 %)** | **1/240 (1 + 0)** |
| hard | first build | 285/300 (95 %) | 175/300 (58 %) | 52/300 (17 %) | 11/240 (11 + 0) |
| hard | **now** | **282/300 (94 %)** | **171/300 (57 %)** | **52/300 (17 %)** | **6/240 (6 + 0)** |

<sub>95 % ranges (Wilson interval: the span of true rates these counts cannot rule out), hard scene now:
normal 91-96 %, subtle 51-62 %, faint 13-22 %, good scrapped 1.2-5.3 % (first build 2.6-8.0 %); easy scene,
both: good scrapped 0.1-2.3 %. "Answered unsure": good parts below the line that the station's own checks
(drift, focus, outline, alignment, frame edge) would still reject; none in either scene. Subtle = half the
normal size and contrast; faint = a quarter. The background drift check warned on 40/40 drifted pictures
per kind and raised no false alarm (0/30 empty, 0/15 part pictures) in every run. Per-picture times are not
comparable between the runs (2-3 jobs shared 2 cores); the memory comparison alone is in point 2 (M4).
Peak memory of a whole evaluation run: about 1.2 GB before and after. Full tables:
`docs/results/v0.2/<scene>/onnx-resnet18-local/` (now) and `docs/results/v0.1/<scene>/...` (first build's code on
the same pictures).</sub>

Hard scene by defect type (caught / tested, all six shape/faces; first build -> now):

| severity | scratch | chip | dent | missing corner | stain |
|---|---|---|---|---|---|
| normal | 54 -> 52 /60 | 60 -> 60 /60 | 53 -> 53 /60 | 58 -> 57 /60 | 60 -> 60 /60 |
| subtle | 21 -> 18 /60 | 48 -> 50 /60 | 7 -> 10 /60 | 43 -> 38 /60 | 56 -> 55 /60 |
| faint | 6 -> 4 /60 | 22 -> 22 /60 | 4 -> 6 /60 | 17 -> 16 /60 | 3 -> 4 /60 |

Where the first build caught more on the hard scene (disc/top: subtle scratches 6 vs 2), its line had been
pulled into the good parts by the old cost rule (6 of 40 good disc/top parts rejected; now 0 of 40).
Pictures: `scores_square_top_hard_scene.png` (scores), `heatmap_square_top_dent_hard_scene_missed.jpg` (a
missed dent), `heatmap_square_top_scratch_hard_scene.jpg`.

Picture of the two scenes: `docs/screenshots/2026-11-12/sim_easy_vs_hard_scene.jpg`.

**2. What each fix does, measured on the simulator** (finding by finding in Appendix B):

| finding | before | now | evidence |
|---|---|---|---|
| H1 moving part / changed backdrop | slide 1 px/frame: accept; slide 3 px/frame, light shift, stray object, camera bump: accept with a flag | all `reject`, `needs_attention`, reason; moving: `recheck: true`; the arm example acts on both | `station_probes_before_after.png`, `stillness_check_slow_slide.png`, tests |
| H2 threshold with one odd damaged picture | 17-46 % of good parts scrapped (median of 200 draws) | about 1 %; the picture is named for a person to check | `threshold_rule_audit_cases.png`, `tests/test_threshold.py` |
| H5 reused / missing part id | new part sent as p1 after a reject: part reject; no id: faces never paired | new part; faces pair, the reply carries the id | probes picture, tests |
| H6 "empty" with a part in view | taken; every later part rejected | refused, old one kept; `force` after deliberate changes; warning while the empty spot differs from training | probes picture, tests |
| M5 empty spot saved as "good" | kept | not kept, with the reason; leave-one-out scores (each training picture scored against all the others) name odd training pictures | probes picture, tests |
| M9 mix-ups (wrong face / shape, edge, suction cup, arm in view) | plain reject, no reason | reject with `needs_attention` and the reason | probes picture |
| M1 leftover shift after straightening | 2 px: 6/20 good parts rejected; 4 px: 20/20 | 2 px: 1/20; 4 px: 14/20; subtle still 20/20 | Appendix F, `tests/test_model.py` |
| M4 speed | memory comparison 717 ms (48 x 3 pictures) | 120 ms, same scores; model tied to its network file | `memory_compare_speed.png`, `tests/test_model.py` |

**3. Stillness** (H1): of 96 settled simulated bursts (easy and hard, 4 and 8 frames) none is called moving;
a steady slide of 1 px per frame or more is always caught (both scenes); 0.5 px per frame: 4/4 easy, 2/4 hard.

**4. Lens calibration with the printed card** (new): on 12 simulated tilted views through a barrel-distorting
lens, the fit recovers the focal length within 0.2 % and the scale at the spot as 0.7145 mm per pixel (true
0.7143); a straight row of corners that bowed 0.27 px bows 0.03 px after correction. The accept limit for the
reprojection error is 3 x the corner jitter measured on the still card. <sub>Reprojection error: how far, in
pixels, the fitted lens puts each card corner from where the camera saw it.</sub> Pictures:
`calibration_card_print.png`, `lens_calibration_before_after.jpg`.

**5. Camera path** (H4), tested with a pretend OpenCV camera that keeps 4 old frames: the first build's
reads returned 5 old frames of 8 after a pause; the reader thread and the buffer drain return none. A setting
the camera does not keep is named (`camera_check_terminal.png`).

**6. Install** (H7): `requirements.txt` now gives ranges; a fresh Python 3.11 environment installed and passed
the whole QC suite (versions in Appendix D). Python 3.10 was not tried (the build machine ran out of disk
space while fetching a second interpreter).

**7. Found in the fresh-clone review** (and fixed): on the hard scene, `collect-good` in the README quick
start discarded 7 of 120 good pictures for "overall brightness changed" by about 1 grey level: the light's own
frame-to-frame flicker, judged against a limit from only 10 empty pictures. With H1's rule the same false
alarm in `run` would scrap good parts. Now a background change counts only when it also exceeds what the
training pictures themselves showed, and collection keeps a picture whose whole-picture brightness moved
slightly (noted). After the fix: 1 of 120 discarded (a change in one area just over its limit), and the quick
start's 20 parts went 15 good accepted, 5 damaged rejected. The result tables above were made before this
fix with the stricter rule; it only loosens a check that flagged no good test part there, so they stand.
Pictures: `quickstart_hard_scene_collect_refusals_before_review_fix.png`, `quickstart_hard_scene_after_review_fix.png`.

**8. Tests**: QC suite 123 passed in a fresh clone, on Python 3.13 and on Python 3.11 (pretrained backbone present); copied tools 378 passed, 33 skipped, 2 expected failures and 14 passed,
6 skipped, as in the first build (Appendix D).

## Conclusion

1. The station no longer answers "accept" for a picture it cannot trust (moving part, changed backdrop or
   light, blur, mix-ups): it rejects, says why, and asks for a recheck when that may help; and one odd damaged
   picture can no longer drag the line into the good parts. Part ids, "empty", the good folder and the camera
   reads are guarded. Every high and medium finding is fixed in code, guide and tests (Appendix B).
2. On the easy scene the results are unchanged (300/300, 295/300, 1/240): the fixes cost nothing there. On the
   hard scene the catch rates are the same within chance and good parts scrapped fell from 11/240 to 6/240.
3. *(Claude's)* The honest simulated picture is the hard scene: about 94 % of normal-size damage, 57 % of
   subtle and 17 % of faint damage caught. Dents and scratches suffer most because good parts' own handling
   marks look alike; better features or light corrections did not change that (Appendix F). Only real labelled
   parts can say how the real station does.
4. *(Claude's)* 6/240 (2.5 %) is still above the 1 % target (95 % range 1.2-5.3 %): a line set from 20
   held-out parts underestimates the tail when good parts vary as much as in the hard scene. On the station,
   watch the false-reject rate in the first runs; more held-out pictures (or a later re-calibration with the
   accepted parts' scores) is the fix, not a typed-in margin.
5. *(Claude's)* The camera path and the lens are now checked by the program itself (`camera-check`,
   `calibrate-camera`); those two commands are the first real test of this build.

## What's next

1. Josh and Chris: steps 6-8 of the setup guide on the real station first (`camera-setup`, `camera-check`,
   `calibrate-camera`); send `runs/camera_check_*.json` to Nick if any check fails.
2. Then collect 50-100 different good parts per shape and face and 10-20 damaged ones (damaged face only),
   train, and look at every `CHECK:` line.
3. Nick and the team: confirm the 1 % target, set the expected defect rate (placeholder 5 %), the
   false-reject cap (placeholder 2 %) and the cost ratio (placeholder 10:1).
4. A real held-back set with graded damage (clear, mild, barely visible) to find the smallest defect the
   station sees; the simulator cannot answer that.
5. Mirror views, once the mirrors' positions are fixed.

## Sweep 1: new tests that may now be needed

- On the real station: `camera-check` on the team's webcam and computer (old frames, settings after a
  restart, automatic exposure), and the lens calibration's reprojection error against its limit.
- The stillness limits with the real arm: how often a settled part is called moving (each costs a recheck)
  and whether a slow slide is caught.
- How often a real "empty" sent every 50 parts is refused after a slow light drift (each refusal stops the
  arm until a person clears it).
- The station's checks on real good parts: how many good parts are answered "unsure" (each is scrapped).
  Damaged parts that also trip a check get `needs_attention`, which asks a person to look: count how often.
- The threshold rule with real damaged parts added one at a time, including one photographed on its
  undamaged face.
- Leave-one-out outlier listing on real training sets: how many false alarms per 60 pictures.
- Timing and memory on the actual Raspberry Pi, with 7 copies per training picture.
- Python 3.10 install (not tried here).
- Shift copies, rotation copies and `local_window` together (DOE), on real parts.

## Sweep 2: possible inaccuracies in what this relied on

- **The hard scene is still a simulator.** Its factors are chosen, not measured (10 % light slope, a 40-grey
  reflection, 6 % brightness spread, 1.2 px print offset, marks of 6-14 grey, placement sd 15 px, JPEG 75,
  noise sd 4). No DOE fixes this; real pictures do.
- **Subtle and faint** are our definitions (half and a quarter of the normal size and contrast). In the hard
  scene the good parts' own handling marks (6-14 grey levels) are about as strong as a subtle scratch, so
  "subtle" misses there are partly by construction *(Claude's)*.
- **Small samples**: 10 parts per defect type and severity per shape/face; differences of a few parts
  between cells are within chance (95 % ranges under the results table in Data 1).
- **Business numbers** typed in as placeholders: expected defect rate 5 %, false-reject cap 2 %, cost 10:1
  (`[chosen]`, no DOE: the team decides). Target 1 % from Nick, to be confirmed.
- **Chosen numbers new in this build** (Appendix E), with whether a DOE should set them.
- **Drift limits from the training pictures**: a picture with a stray object that slipped into the good folder
  would widen them; collection refuses pictures with a change in one area, which limits that risk.
- **Pretend camera**: old-frame and settings behaviour was tested with a fake OpenCV camera written from how
  UVC drivers usually behave, not with a real webcam.
- **Lens calibration** was tested on simulated cards only; the simulated corner jitter (0.006 px) is far
  below a real webcam's, so the limit's floor (0.02 px) decided it in the simulator.
- **mm per pixel** is measured at the card's height; a part's top face sits higher by its thickness (a 5 mm
  part at 40 cm looks about 1.3 % bigger).
- **Timings** were taken while other jobs shared the 2 CPU cores; Raspberry Pi times are not measured.
- **Shift tolerance (M1)** was re-measured on one shape/face (square/bottom, 20 good and 20 subtle parts);
  4 px of leftover shift still rejects most good parts: straightening must stay within about 2 px.

---

## Appendix A. Full result tables

Per shape and face, hard scene, this build (the other three runs: `docs/results/v0.2/easy/`, `docs/results/v0.1/easy/`, `docs/results/v0.1/hard/`, each `onnx-resnet18-local/table.md` with `results.json` and score charts). Cells: caught / tested (heat map's hottest spot on the defect).

Computer-generated pictures (qc/sim.py, hard scene, fixed seeds). Per shape/face: 60 good pictures to train, 20 good + 10 damaged (2 per type) to set the threshold, then 40 good test parts and 10 per defect type at each severity. Cells: caught / tested (heat map's hottest spot on the defect). "unsure": good parts the station's own checks would answer unsure (reject) although their score is below the line.

**Hard scene, now: normal severity**

| shape / face | scratch | chip | dent | missing_corner | stain | false rejects (good) | unsure (good) |
|---|---|---|---|---|---|---|---|
| disc/top | 8/10 (8) | 10/10 (7) | 9/10 (9) | 10/10 (10) | 10/10 (10) | 0/40 | 0/40 |
| disc/bottom | 9/10 (8) | 10/10 (7) | 10/10 (10) | 10/10 (7) | 10/10 (10) | 0/40 | 0/40 |
| square/top | 7/10 (7) | 10/10 (10) | 6/10 (6) | 10/10 (5) | 10/10 (10) | 1/40 | 0/40 |
| square/bottom | 10/10 (10) | 10/10 (8) | 9/10 (8) | 10/10 (6) | 10/10 (10) | 0/40 | 0/40 |
| hexagon/top | 8/10 (8) | 10/10 (10) | 9/10 (9) | 7/10 (7) | 10/10 (10) | 1/40 | 0/40 |
| hexagon/bottom | 10/10 (10) | 10/10 (9) | 10/10 (10) | 10/10 (8) | 10/10 (10) | 4/40 | 0/40 |
| **all shapes** | **52/60** | **60/60** | **53/60** | **57/60** | **60/60** | **6/240** | **0/240** |

**Hard scene, now: subtle severity** (half the normal size and contrast)

| shape / face | scratch | chip | dent | missing_corner | stain | false rejects (good) | unsure (good) |
|---|---|---|---|---|---|---|---|
| disc/top | 2/10 (0) | 10/10 (8) | 0/10 (0) | 10/10 (6) | 7/10 (7) | 0/40 | 0/40 |
| disc/bottom | 3/10 (3) | 9/10 (7) | 2/10 (2) | 4/10 (3) | 10/10 (10) | 0/40 | 0/40 |
| square/top | 0/10 (0) | 2/10 (0) | 0/10 (0) | 6/10 (4) | 10/10 (10) | 1/40 | 0/40 |
| square/bottom | 5/10 (4) | 9/10 (5) | 4/10 (4) | 9/10 (3) | 10/10 (10) | 0/40 | 0/40 |
| hexagon/top | 3/10 (2) | 10/10 (9) | 1/10 (1) | 3/10 (0) | 9/10 (9) | 1/40 | 0/40 |
| hexagon/bottom | 5/10 (4) | 10/10 (8) | 3/10 (1) | 6/10 (4) | 9/10 (9) | 4/40 | 0/40 |
| **all shapes** | **18/60** | **50/60** | **10/60** | **38/60** | **55/60** | **6/240** | **0/240** |

**Hard scene, now: faint severity** (a quarter of the normal size and contrast)

| shape / face | scratch | chip | dent | missing_corner | stain | false rejects (good) | unsure (good) |
|---|---|---|---|---|---|---|---|
| disc/top | 0/10 (0) | 7/10 (5) | 0/10 (0) | 7/10 (4) | 0/10 (0) | 0/40 | 0/40 |
| disc/bottom | 0/10 (0) | 2/10 (0) | 0/10 (0) | 2/10 (2) | 0/10 (0) | 0/40 | 0/40 |
| square/top | 0/10 (0) | 0/10 (0) | 0/10 (0) | 0/10 (0) | 0/10 (0) | 1/40 | 0/40 |
| square/bottom | 0/10 (0) | 6/10 (1) | 1/10 (0) | 3/10 (2) | 2/10 (0) | 0/40 | 0/40 |
| hexagon/top | 1/10 (0) | 1/10 (1) | 1/10 (0) | 0/10 (0) | 1/10 (0) | 1/40 | 0/40 |
| hexagon/bottom | 3/10 (0) | 6/10 (6) | 4/10 (0) | 4/10 (0) | 1/10 (0) | 4/40 | 0/40 |
| **all shapes** | **4/60** | **22/60** | **6/60** | **16/60** | **4/60** | **6/240** | **0/240** |

**Hard scene, now: background drift check** (warnings / pictures)

| case | empty spot | with a part |
|---|---|---|
| no drift (false alarms) | 0/30 | 0/15 |
| light_shift | 10/10 | 10/10 |
| stray_object | 10/10 | 10/10 |
| camera_bump | 10/10 | 10/10 |
| glare | 10/10 | 10/10 |

Timing on the build machine (2 CPU cores): training 96-187 s per shape/face, 1102-1875 ms per picture (straighten + score).

## Appendix B. Finding by finding

| finding | status | what changed | tests |
|---|---|---|---|
| H1 accept on a moving part / changed backdrop | fixed | Any warning (moving, drift, blurrier than training, frame edge, outline or face unlike training) makes the answer `unsure_goes_to` (reject), never accept; `reason` says why; `recheck: true` when asking again may help. Stillness also compares frames 2+ apart and uses the burst's own noise floor (a bright part is noisier than the dark backdrop). The arm example reads `needs_attention` and `recheck`. Switch `warnings_make_unsure` (off = first build). Guide and code agree. | test_station_handoff (h1 x 6), test_hard_scene, test_capture |
| H2 threshold rule with damaged parts | fixed | Expected defect rate and counts per part made; the line never rises above the good-only line nor falls below the false-reject cap; damaged pictures that score like good parts are named, not used. Business numbers: target 1 % (Nick), defect rate 5 % and cap 2 % `[chosen]`, cost 10:1 `[chosen]`. | test_threshold (6 new) |
| H3 flattering simulator | fixed | Hard scene in `qc/sim.py` (default; easy kept behind `sim_scene`), faint severity, both scenes reported before and after. Improvement tried: shift copies (on), light flattening and a learned light field (off: no gain; Appendix F). | test_hard_scene |
| H4 real-camera path | fixed (untested on hardware) | Backend per system (DirectShow / V4L2 / AVFoundation with a warning); reader thread holds the newest frame, burst takes only frames made after the request (first build's reads kept behind `camera_reader_thread`, now with a buffer drain); every setting read back, a setting that did not hold is named in the window and the log; gain locked; exposure / gain / focus / white balance default to the camera's own value instead of Windows-only numbers; `camera-check` day-one command; camera-setup's exposure slider uses the backend's scale. Resolution left at 640 x 480 (a DOE item). | test_camera (9, pretend webcam) |
| H5 part ids | fixed | A finished part's id starts a new part; a repeated face of an unfinished part replaces the earlier answer; faces without an id pair up, the reply carries the id. | test_station_handoff (3) |
| H6 empty with a part in view | fixed | Refused when something part-sized differs from the previous empty (or a model's training-time empty); `force` for deliberate changes; models keep their training-time empty and parts carry a warning while the current one differs. | test_station_handoff (2), test_cli_flow |
| H7 install on 3.10 / 3.11 | fixed (3.11 tested; 3.10 not) | Version ranges; guide says 3.10-3.13, Windows or Linux. | fresh 3.11 environment, whole QC suite |
| M1 1-px alignment dependence | fixed | +/-3 px shifted training copies (`train_shifts_px`, `[chosen]`, DOE with rotations and `local_window`). | test_model |
| M2 blur | fixed | Focus check: a picture blurrier than every training picture by more than their own spread answers unsure. Blurred training copies (audit suggestion) not added: they would hide real blur. | test_station_handoff |
| M3 mirrors | fixed in the guide | Guide: remove or cover mirrors; mirror views in a later build. | (guide) |
| M4 speed and network guard | fixed | Cell-first memory (same scores, 717 -> 120 ms at 48 x 3 pictures); features embedded in chunks (peak memory); the backbone file's SHA-256 is stored and a different file makes the answer unsure; the 8-bit file has its own name. | test_model, test_station_handoff |
| M5 what is saved as good | fixed | collect refuses no part, frame edge, moving, drift; leave-one-out scores list odd training pictures; guide: different parts, each once. | test_station_handoff, test_model |
| M6 small and dark parts | fixed | Warnings after training (crop below the network's 224 px, low contrast, no part found); guide: every shape at least a third of the width, light backdrop for dark parts. | test_cli_flow |
| M7 learning curve | fixed | "Enough" judged by the good-only line settling (needs no damaged parts); painted damage labelled as a stand-in; two-panel chart. | test_cli_flow |
| M8 thresholds from few pictures | fixed | At least 20 held back; count checked before a model is saved; a held-out outlier is set aside and named. | test_threshold, test_cli_flow |
| M9 mix-ups without a reason | fixed | Frame edge, area and alignment checks with limits from the training pictures; reject with `needs_attention` and the reason. | test_station_handoff |
| L1 guide clarity | fixed | Order (light and backdrop before camera settings), same computer, how to send a request, real messages quoted, "what you should see" for each step, troubleshooting, Raspberry Pi note, shiny parts under a ring light. | (guide) |
| L2 empty_every_parts | fixed | Counts parts, not faces. | test_station_handoff |
| L3 non-object JSON | fixed | Error reply instead of a dropped connection. | test_station_handoff |
| L4 camera read failure | fixed | Retried, then the camera is reopened; a plain error if it never recovers. | test_camera |
| L5 threshold reload, log overwrite | fixed | threshold.json read again when it changes; repeated ids get `-2`, `-3` picture names. | test_station_handoff |
| L6 fallback warning | fixed | Shown in the run window. | (manual) |
| L7 rule gaps | fixed | Remaining numbers marked `[chosen]`; statistics explained in fine print at first use; the learning-curve sentence no longer shows a variable name. | |
| L8 licences | nothing to fix | Guide: keep a copy of the backbone file. | |
| Review (fresh clone): drift false alarms on the hard scene | fixed | Drift limits also from the training pictures' own flicker; collect keeps slight whole-picture brightness flicker (noted). | test_station_handoff (2) |
| Added: lens calibration card (Nick, 12 Nov) | built | `make-card`, `calibrate-camera`, undistortion and mm scale behind `lens_correction` (inert without a calibration); models and the empty reference remember the calibration they were made with. | test_lens (7) |

## Appendix C. The threshold rule (H2, M8)

1. Held-out good scores: the highest is checked against a Gumbel fit to the others; if a score that high
   would come up by chance less than 1 time in 100 among that many parts, the picture is set aside and named.
   <sub>Gumbel: the usual distribution for the largest of many values, as a part's score is the largest of
   its cells' scores.</sub>
2. Good-only line: where the fit expects `target_false_reject` (1 %) of good parts above it, and never below
   the highest held-out good score.
3. Damaged pictures scoring no higher than a good part seen are listed (`CHECK:`) and not used.
4. The others may lower the line, never raise it, never below where the fit expects `max_false_reject`
   (2 %), and only when that lowers the expected cost per part made:
   `defect_rate x cost_bad x (share of damaged accepted) + (1 - defect_rate) x cost_good x (share of good rejected)`.
5. At least 20 good pictures are held back when there are 40 or more (`min_holdout`); with fewer than 3
   held back, training stops before a model is saved.

Audit cases (20 held-out good scores, 9 clearly damaged, the 10th as named; 200 draws, median):

| the 10th damaged picture | first build: line | first build: good parts rejected (long run) | now: line | now: good parts rejected | good-only line |
|---|---|---|---|---|---|
| clearly damaged | 7.26 | 1.1 % (95th percentile 3.9 %) | 7.21 | 1.2 % (95th percentile 4.7 %) | 7.21 |
| damage not visible on this face (scores like a good part, 5.4) | 5.38 | 37.5 % (95th percentile 41.9 %) | 7.21 | 1.2 % (95th percentile 4.7 %) | 7.21 |
| faint (5.9) | 5.85 | 16.8 % (95th percentile 20.4 %) | 7.21 | 1.2 % (95th percentile 4.7 %) | 7.21 |
| scores below the good median (5.25) | 5.25 | 45.8 % (95th percentile 49.4 %) | 7.21 | 1.2 % (95th percentile 4.7 %) | 7.21 |

<sub>Scores drawn from a Gumbel distribution (location 5, scale 0.5), 200 draws, median shown; "long run" from the true distribution. The 95th-percentile column shows how far a line set from 20 held-out parts can be off (audit M8). Expected defect rate 5 %, cost 10:1, cap 2 %.</sub>

## Appendix D. Install and tests

`requirements.txt` now gives version ranges (numpy >= 1.24 < 3, scipy >= 1.10, opencv-python >= 4.8,
onnxruntime >= 1.16, onnx >= 1.14, matplotlib >= 3.7, pytest >= 7.4) instead of exact versions that needed
Python 3.12 or newer.

| environment | what pip chose | QC suite |
|---|---|---|
| Python 3.11.18, fresh environment (Raspberry Pi OS Bookworm's version) | numpy 2.4.6, scipy 1.17.1, opencv 5.0.0, onnxruntime 1.31.1, onnx 1.23.2, matplotlib 3.11.2, pytest 9.2.0 | 123 passed (fresh clone) |
| Python 3.13.18 (build machine) | numpy 2.5.3, scipy 1.18.1, opencv 5.0.0, onnxruntime 1.29.0, onnx 1.23.2, matplotlib 3.11.2, pytest 9.1.1 | 123 passed (fresh clone) |

Copied tools (Python 3.13): `python -m pytest tools/detector/tests tools/label/tests tools/recognition/tests`
378 passed, 33 skipped, 2 expected failures; `python -m pytest tools/tests` 14 passed, 6 skipped (both as in
the first build). New QC test files: `test_camera.py` (pretend webcam), `test_lens.py` (card and
calibration), `test_hard_scene.py`; new tests in `test_station_handoff.py`, `test_threshold.py`,
`test_model.py`, `test_cli_flow.py`. Unit tests of single mechanisms pin the easy scene
(`tests/conftest.py`); the hard scene has its own tests.

## Appendix E. Chosen numbers new in this build

| setting | value | where | DOE? |
|---|---|---|---|
| train_shifts_px | +/-3 px | config | yes, with train_rotations_deg and local_window |
| STILL_FACTOR / SLIDE_FACTOR | 3x / 1.5x the noise unit | capture.py | yes: false "moving" vs. missed slides on the real arm |
| SLIDE_MIN_BURST | 6 frames | capture.py | no (the slide check needs frames 2+ apart on a settled half) |
| RECHECKS (arm example) | 2 | arm_client_example.py | no: a cycle-time choice |
| expected_defect_rate / max_false_reject | 5 % / 2 % | config | no: business numbers for the team |
| cost_bad_accepted : cost_good_rejected | 10 : 1 | config | no: business placeholder (Nick: not confirmed) |
| min_holdout | 20 | config | no: more held-out parts is always better |
| OUTLIER_CHANCE | 1 % | threshold.py | no |
| MIN_CONTRAST | 3x camera noise | train.py | no (a warning only) |
| JITTER_FACTOR / JITTER_FLOOR | 3x / 0.02 px | lens.py | no: a quality gate |
| MIN_VIEWS / MIN_TILT_SPREAD_DEG / MIN_COVER | 8 / 10 deg / 30 % | lens.py | no (OpenCV practice) |
| READ_TOL | exposure 0.5, gain 1, wb 50 K, focus 1 | capture.py | no: read-back rounding |
| auto-exposure check | 3x frame-to-frame change | camcheck.py | no |
| LIGHT_SMOOTH (light_field, off) | 6 px | model.py | only if the switch is used |

Limits that now come from the training pictures (no number typed in): the area, sharpness and alignment
ranges (lowest or highest seen, widened by the training pictures' own spread on that side), the stillness
unit (the larger of the empty scene's change and the burst's own floor), the calibration's accept limit
(measured corner jitter), the held-out outlier test (a fit to the other scores).

## Appendix F. Options tried on the hard scene and left off

Square/bottom and hexagon/top, hard scene, 60 good to train, 20 held out (good-only line), 30 good and 30
damaged per severity to test (6 per type). Cells: good rejected, then caught / tested at normal, subtle, faint.

| option | square/bottom | hexagon/top |
|---|---|---|
| first build (no shift copies) | 0/30; 30/30, 19/30, 3/30 | 0/30; 24/30, 15/30, 1/30 |
| shift copies +/-3 px (now the default) | 0/30; 30/30, 20/30, 2/30 | 0/30; 25/30, 14/30, 1/30 |
| shift copies + `flatten_light` | 0/30; 30/30, 21/30, 4/30 | 0/30; 23/30, 13/30, 1/30 |
| `flatten_light` alone | 0/30; 30/30, 19/30, 4/30 | 0/30; 24/30, 13/30, 1/30 |
| shift copies + `light_field` | 2/30; 30/30, 21/30, 4/30 | (not run) |
| `light_field` alone | 2/30; 30/30, 20/30, 4/30 | (not run) |

None of the light corrections helps by more than chance, and the learned light field raised false rejects,
so both stay switched off (kept, not deleted). Shift copies are neutral on catch rate here; they are on for
robustness (re-measured on this build, square/bottom, easy scene, 40 good to train: a leftover shift after
straightening of 2 px rejected 6/20 good parts without the copies and 1/20 with; 4 px: 20/20 and 14/20;
subtle defects caught 20/20 in every case; the audit measured 1/20 and 11/20 for the same change). *(Claude's)* In the hard scene the good
parts carry handling marks of 6-14 grey levels, about as strong as a subtle scratch: no feature or light
trick separates those without examples of which marks are damage; real labelled parts can.
