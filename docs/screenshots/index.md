# Screenshots

One line per picture: what it shows, **feature** or **problem**, and the build or report it belongs to.
Append new lines; never overwrite another build's lines.

## 2026-11-12 (first build, report: docs/build-reports/2026-11-12-first-build.md)

- `2026-11-12/straighten_before_after_square_bottom.jpg`: six simulated square parts at random angles and the same parts straightened; the "A7" stamp picks which of the four symmetric angles. **feature**, first build.
- `2026-11-12/burst_sharpest_frame_pick.jpg`: an 8-frame burst after "paused"; sharpness and still/moving per frame, the kept frame boxed. **feature**, first build.
- `2026-11-12/drift_check_empty_frames.jpg`: background drift check on a normal empty frame and four kinds of drift, with the biggest change ringed. **feature**, first build.
- `2026-11-12/heatmap_square_top_scratch.jpg`: square top face, simulated scratch, heat map on the straightened part. **feature**, first build.
- `2026-11-12/heatmap_square_top_chip.jpg`: square top face, simulated chip on the edge. **feature**, first build.
- `2026-11-12/heatmap_square_top_dent.jpg`: square top face, simulated dent. **feature**, first build.
- `2026-11-12/heatmap_square_top_missing_corner.jpg`: square top face, simulated missing corner; the heat covers a wider area than the cut, which is why the "points at the defect" count is lower for edge damage. **feature** and **problem**, first build.
- `2026-11-12/heatmap_square_top_stain.jpg`: square top face, simulated stain. **feature**, first build.
- `2026-11-12/heatmap_disc_top_scratch.jpg`: disc top face, simulated scratch. **feature**, first build.
- `2026-11-12/heatmap_disc_top_stain.jpg`: disc top face, simulated stain. **feature**, first build.
- `2026-11-12/heatmap_hexagon_top_scratch.jpg`: hexagon top face, simulated scratch. **feature**, first build.
- `2026-11-12/heatmap_hexagon_top_stain.jpg`: hexagon top face, simulated stain. **feature**, first build.
- `2026-11-12/learning_curve_square_top_synthetic.png`: learning curve on simulated square/top (5-80 good pictures, 5 draws each, 15 subtle damaged parts); flat within draw-to-draw noise. **feature**, first build (report Appendix C).
- `2026-11-12/scores_square_top_local_bank.png`: anomaly scores of 40 good test parts and 10 of each defect type, square/top, default settings; all damaged parts far right of the threshold. **feature**, first build.
- `2026-11-12/scores_disc_bottom_handcrafted_fallback.png`: the same chart with the hand-made fallback features on disc/bottom: some of the damaged parts used to set the threshold score among the good ones, so the cost balance pulls the line into the good parts (9 of 40 good rejected). Why the pretrained backbone is the default. **problem**, first build.
- `2026-11-12/quickstart_terminal_output.png`: terminal output of the README quick start run in a fresh clone (simulated: 60 good + 10 damaged square parts per face, then 20 parts through the pretend arm: 15 good accepted, 5 damaged rejected). **feature**, first build (review).

## 2026-11-12 (audit fixes, report: docs/build-reports/2026-11-12-audit-fixes.md)

- `2026-11-12/sim_easy_vs_hard_scene.jpg`: the same simulated parts in the first build's easy scene and the new hard scene (light fixed in the room, a reflection, more variation, handling marks, noisy JPEG camera), and one scratch at normal, subtle and faint severity. **feature**, audit fixes (H3).
- `2026-11-12/station_probes_before_after.png`: the audit's station probes, first build vs now: moving parts, changed backdrops, reused and missing part ids, mix-ups, an empty spot in the good folder, "empty" with a part in view. **problem** (first build) and **feature** (fix), audit fixes (H1, H5, H6, M5, M9).
- `2026-11-12/threshold_rule_audit_cases.png`: chart, good parts rejected in the long run when one of ten damaged pictures scores like a good part: first build's rule 17-46 %, now about 1 %. **problem** and **feature**, audit fixes (H2).
- `2026-11-12/stillness_check_slow_slide.png`: chart, the stillness measure against slide speed, first build vs now, easy and hard scene: a 1 px per frame slide used to pass as still. **problem** and **feature**, audit fixes (H1).
- `2026-11-12/memory_compare_speed.png`: chart, time to compare a picture with the memory against memory size, first build's layout vs the cell-first layout. **feature**, audit fixes (M4).
- `2026-11-12/camera_check_terminal.png`: terminal output of `python -m qc camera-check` against a pretend webcam that keeps old frames and ignores white balance: the station's reader thread gets no old frames, the first build's plain reads get five. **feature** (and the H4 **problem** it shows), audit fixes (H4).
- `2026-11-12/calibration_card_print.png`: the printable checkerboard calibration card (`python -m qc make-card`). **feature**, audit fixes (lens calibration, Nick's added scope).
- `2026-11-12/lens_calibration_before_after.jpg`: a simulated webcam's barrel distortion on the card, before and after the lens calibration; corners bow off a straight line before, lie on it after. **feature**, audit fixes (lens calibration).
- `2026-11-12/scores_square_top_hard_scene.png`: chart, anomaly scores of 40 good test parts and 10 of each defect type, square/top, hard scene, default settings: good parts and some scratches and dents now overlap near the line. **problem** (what the hard scene shows), audit fixes (H3).
- `2026-11-12/heatmap_square_top_dent_hard_scene_missed.jpg`: square top face, hard scene, a normal-size dent whose heat map points at it but whose score stays just under the line (11.3 vs 12.1): missed. **problem**, audit fixes (H3).
- `2026-11-12/heatmap_square_top_scratch_hard_scene.jpg`: square top face, hard scene, a scratch caught, heat map on the straightened part. **feature**, audit fixes (H3).
- `2026-11-12/learning_curve_two_panels_hard_scene.png`: chart, the new two-panel learning curve on the hard scene (hand-made features as a fast stand-in): the good-only line is still falling at 40 pictures, so the tool says "not settled". **feature**, audit fixes (M7).
