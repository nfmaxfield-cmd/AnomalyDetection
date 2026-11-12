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
