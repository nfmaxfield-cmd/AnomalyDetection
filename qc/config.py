"""Settings for the QC station, with every switch in one place.

Changelog
  0.1.0  2026-11-12  first version.
  0.2.0  2026-11-12  audit fixes: camera backend, reader thread, gain; exposure / focus / white balance
                     default to "keep the camera's value" (the old -6 fitted Windows only); shifted
                     training copies; expected defect rate and false-reject cap for the threshold; what
                     happens to a picture with warnings; simulator scene; minimum held-out pictures;
                     lens_correction (checkerboard calibration card); light_field / flatten_light options.

Settings live in a JSON file (default qc_config.json in the working folder). Anything missing takes the
default below, so an old file keeps working when new settings are added. Switches that turn a kept-but-
unneeded feature on or off default to OFF (Nick's rule: never delete a feature, put it behind a switch).

Numbers here are of three kinds, and each is marked:
  [data]    worked out from the station's own pictures at run time; the value here is only a fallback
  [chosen]  picked by us; a designed experiment (DOE) could set it better - see the build report
  [fixed]   a property of the method (e.g. the backbone's input size), not a tuning knob
"""
import copy
import json
import os

DEFAULTS = {
    # --- scene / camera -------------------------------------------------------------------------
    "camera_index": 0,                 # which webcam (0 = first)
    "frame_width": 640,
    "frame_height": 480,
    "camera_backend": "auto",          # auto (Windows: dshow, Linux / Pi: v4l2, Mac: avfoundation) | dshow | msmf | v4l2 | any
    "camera_reader_thread": True,      # a thread always holds the newest frame (no old buffered frames);
                                       # False = the first build's direct reads, now with a buffer drain
    "lock_camera_settings": True,      # turn auto exposure / gain / white balance / focus off, read back
    "exposure": None,                  # driver-specific; None = keep the camera's own value. Set once with
    "gain": None,                      #   camera-setup (SETUP-GUIDE step 4); camera-check shows what held
    "white_balance_k": None,           #   (the first build typed in -6 / 4500 / 30: right for Windows only)
    "focus": None,
    # --- capture ----------------------------------------------------------------------------------
    "burst_frames": 8,                 # [chosen] frames grabbed after "paused"; DOE: sharpness vs time
    "burst_settle_s": 0.0,             # [chosen] wait after "paused" before grabbing
    "still_check": True,               # refuse a burst whose part is still moving
    "focus_check": True,               # warn (and answer "unsure") when a picture is blurrier than any
                                       # training picture by more than their own spread (audit M2)
    # --- background / locate ----------------------------------------------------------------------
    "empty_frames": 10,                # [chosen] empty pictures averaged into the reference
    "empty_spacing_s": 0.3,            # [chosen] seconds between empty pictures (catches slow flicker)
    "empty_every_parts": 50,           # [chosen] ask for a fresh empty picture every N parts (0 = never)
    "mask_noise_quantile": 0.999,      # [chosen] noise level taken from empty-vs-empty differences
    "locate_with_detector": False,     # SWITCH (off): use tools/detector CentreNet to find the part
    "detector_weights": "",            # path to a detector .onnx when the switch above is on
    # --- straighten -------------------------------------------------------------------------------
    "symmetry": {},                    # per shape: {"square": 4}; missing = detected from the outline
    "fine_align_deg": 5.0,             # [chosen] fine search range (+/-), from the agreed plan
    "fine_align_step_deg": 0.5,        # [chosen]
    "crop_margin": 1.15,               # [chosen] canonical crop = part size x this
    "canonical_px": 224,               # [fixed] backbone input size
    # --- model ------------------------------------------------------------------------------------
    "backbone": "auto",                # auto | onnx | torch | handcrafted
    "backbone_onnx": "models/backbone/resnet18_l2l3.onnx",
    "torch_arch": "resnet18",          # resnet18 (default, small) | wide_resnet50_2
    "bank": "local",                   # local (patches compared with the same spot) | global (classic PatchCore)
    "local_window": 1,                 # [chosen] +/- patches searched around the same spot (local bank)
    "coreset_fraction": 0.1,           # [chosen] global bank only; PatchCore paper used 1-10 %
    "train_rotations_deg": [-3.0, 3.0],  # [chosen] extra small rotations added to training (augmentation)
    "train_shifts_px": [3],            # [chosen] copies shifted +/-3 px left/right/up/down (audit M1); DOE with
                                       # the rotations and local_window. [] = the first build (no shifts)
    "flatten_light": False,            # SWITCH: divide out a straight light slope across the part before the
                                       # features (tested on the hard scene, see the audit-fixes report)
    "light_field": False,              # SWITCH: learn light fixed in the room (slope, reflection) from the good
                                       # training parts and divide it out before straightening
    "smooth_sigma": 2.0,               # [chosen] heat-map smoothing in feature cells (PatchCore uses 4 px)
    # --- threshold --------------------------------------------------------------------------------
    "holdout_fraction": 0.2,           # [chosen] share of good pictures kept back to set the threshold
    "min_holdout": 20,                 # [chosen] at least this many held back (up to half the pictures); audit M8
    "target_false_reject": 0.01,       # business choice: share of good parts we accept scrapping (Nick: 1 % to start)
    "max_false_reject": 0.02,          # [chosen] business choice: the cost rule may lower the line at most to here
    "expected_defect_rate": 0.05,      # [chosen] business choice: share of parts that are really damaged
    "cost_bad_accepted": 10.0,         # [chosen] placeholder business choice: cost of a bad part reaching a
    "cost_good_rejected": 1.0,         # ... customer, compared with scrapping a good one (10:1 not confirmed)
    # --- decisions / hand-off ---------------------------------------------------------------------
    "transport": "http",               # http | stdio | file
    "http_host": "127.0.0.1",
    "http_port": 8765,
    "outbox": "outbox",                # file transport: results written here
    "inbox": "inbox",                  # file transport: requests read from here
    "unsure_goes_to": "reject",        # where a part goes when it can't be judged (no part found ...)
    "warnings_make_unsure": True,      # a picture with a warning (part moving, backdrop / light changed, blurred,
                                       # mix-up) answers unsure_goes_to, never "accept" (audit H1). False = the
                                       # first build: decide from the score and only set needs_attention
    "faces_per_part": 2,               # a part record is complete after this many faces; the next face starts a new part
    "log_dir": "runs",
    "save_raw_frames": True,
    # --- simulator ----------------------------------------------------------------------------------
    "sim_scene": "hard",               # --simulate pictures: hard (default, audit H3) | easy (the first build's)
    # --- kept robot-project features (all off) ----------------------------------------------------
    "robot_features": False,           # SWITCH (off): alvik presets, robot logs, home positions in the studio
    "clock_lineup": False,             # SWITCH (off): match recorded frames to an arm log with its own clock
    "lens_correction": True,           # undistort every picture and give mm with the checkerboard calibration
                                       # (python -m qc calibrate-camera); nothing changes while none is saved
    "px_to_mm": False,                 # SWITCH (off): report defect sizes in mm (needs `qc calibrate-mm`)
    "mm_per_px": None,
    "homography": None,
}


def load(path=None):
    cfg = copy.deepcopy(DEFAULTS)
    path = path or os.environ.get("QC_CONFIG", "qc_config.json")
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg.update(json.load(f))
    cfg["_path"] = path
    return cfg


def save(cfg, path=None):
    path = path or cfg.get("_path") or "qc_config.json"
    out = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return path
