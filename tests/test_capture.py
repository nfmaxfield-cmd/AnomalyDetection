"""Sharpest-frame pick and the stillness check on a burst.

Changelog
  0.1.0  2026-11-12  first version.
"""
from qc import capture, sim


def test_sharpest_frame_is_a_settled_one(empty_ref):
    for shape in sim.SHAPES:
        for seed in range(3):
            frames, settled = sim.burst(shape, "top", seed=seed)
            loc = empty_ref.part_mask(frames[-1])
            i, s = capture.pick_sharpest(frames, loc.filled)
            assert i in settled, (shape, seed, i, settled, s)


def test_motion_blurred_frames_score_lower():
    frames, settled = sim.burst("hexagon", "bottom", seed=3)
    s = [capture.sharpness(f) for f in frames]
    assert max(s[:2]) < min(s[i] for i in settled)


def test_still_check(empty_ref):
    frames, settled = sim.burst("disc", "top", seed=5)
    crisp = settled[0]
    loc = empty_ref.part_mask(frames[-1])
    ok, mv, lim = capture.is_still(frames, crisp, empty_ref.still_noise, loc.filled)
    assert ok, (mv, lim)
    # a part that slides 3 px between two frames is not still
    moving = [sim.render("square", "top", seed=6, angle=20.0, centre=(320 + 3 * k, 240), noise_seed=k)["frame"]
              for k in range(3)]
    ok0, mv0, lim0 = capture.is_still(moving, 1, empty_ref.still_noise, loc.filled)
    assert not ok0 and mv0 > lim0 > mv


def test_sim_camera_burst_and_empty():
    cam = capture.SimCamera(seed=1)
    e = cam.burst(3)
    assert len(e) == 3 and e[0].shape == (sim.H, sim.W, 3)
    cam.next_part("square", "top")
    b = cam.burst(6)
    assert len(b) == 6
