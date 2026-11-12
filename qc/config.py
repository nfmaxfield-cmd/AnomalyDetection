"""Settings for the QC station, with every switch in one place.

Changelog
  0.1.0  2026-11-12  first version.

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
    "lock_camera_settings": True,      # turn auto exposure / white balance / focus off
    "exposure": -6,                    # [chosen] driver-specific value; set once in SETUP-GUIDE step 2
    "white_balance_k": 4500,           # [chosen] matches a typical LED ring light
    "focus": 30,                       # [chosen] driver-specific; set once with the part in place
    # --- capture ----------------------------------------------------------------------------------
    "burst_frames": 8,                 # [chosen] frames grabbed after "paused"; DOE: sharpness vs time
    "burst_settle_s": 0.0,             # [chosen] wait after "paused" before grabbing
    "still_check": True,               # refuse a burst whose part is still moving
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
    "smooth_sigma": 2.0,               # [chosen] heat-map smoothing in feature cells (PatchCore uses 4 px)
    # --- threshold --------------------------------------------------------------------------------
    "holdout_fraction": 0.2,           # [chosen] share of good pictures kept back to set the threshold
    "target_false_reject": 0.01,       # business choice: share of good parts we accept scrapping
    "cost_bad_accepted": 10.0,         # business choice: cost of a bad part reaching a customer ...
    "cost_good_rejected": 1.0,         # ... compared with scrapping a good one
    # --- decisions / hand-off ---------------------------------------------------------------------
    "transport": "http",               # http | stdio | file
    "http_host": "127.0.0.1",
    "http_port": 8765,
    "outbox": "outbox",                # file transport: results written here
    "inbox": "inbox",                  # file transport: requests read from here
    "unsure_goes_to": "reject",        # where a part goes when it can't be judged (no part found ...)
    "log_dir": "runs",
    "save_raw_frames": True,
    # --- kept robot-project features (all off) ----------------------------------------------------
    "robot_features": False,           # SWITCH (off): alvik presets, robot logs, home positions in the studio
    "clock_lineup": False,             # SWITCH (off): match recorded frames to an arm log with its own clock
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
