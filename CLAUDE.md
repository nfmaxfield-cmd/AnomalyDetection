# Standing rules for work in this repo (AnomalyDetection)

Public repo (MIT). Built by Nick Maxfield (URI systems engineering) for Dr. Sodhi's Machine Vision team
(Josh S and Chris): a robot arm sorts parts into accept / reject piles from a webcam's decision.

- **Keep features, switch them off.** When a new method replaces an existing feature, never delete the old
  one. Keep it working behind a setting (`qc/config.py` DEFAULTS, or an environment switch such as
  `QC_ROBOT_FEATURES`), off while it isn't needed. Copied files say where they came from in their first
  lines; keep that note and add a changelog line when you change one.
- **Thresholds and limits come from data.** Accept/reject thresholds, drift limits, stillness limits and
  the like are worked out from the station's own pictures (held-out good parts, damaged parts, empty
  frames), never typed in as a fixed number. Where a number had to be chosen, mark it `[chosen]` in the
  code and say in the build report whether a designed experiment (DOE) should set it instead.
- **Agents test offline only.** Use the computer-generated pictures (`qc/sim.py`, `--simulate`) and the
  pytest suites. Never drive a real camera, arm, robot or Raspberry Pi, and never SSH anywhere.
  Run: `python -m pytest` (QC), `python -m pytest tools/detector/tests tools/label/tests tools/recognition/tests`
  and `python -m pytest tools/tests` (copied tools; robot-only tests skip with a reason).
- **Screenshots.** Every build saves pictures of the features it adds and the problems it finds or fixes
  (heat maps, before/after, charts, terminal output) to `docs/screenshots/<YYYY-MM-DD>/` with descriptive
  names, and adds one line per picture to `docs/screenshots/index.md`: what it shows, **feature** or
  **problem**, and the build or report it belongs to. Append; never overwrite another build's lines.
- **Charts.** Every chart states what is measured and where the data comes from, with one plain sentence
  underneath saying what it shows. No single-bar or two-bar charts.
- **Reports** (`docs/build-reports/<date>-<name>.md`): premise -> data -> conclusion -> what's next, lean, detail
  in appendices. Flag conclusions that are the agent's own rather than the team's. End with two sweeps:
  (1) new tests that may now be needed, (2) possible inaccuracies (assumptions, proxies, chosen-not-derived
  numbers, each with whether a DOE should set it). Explain a statistic in fine print the first time it is used.
- **Instructions for Josh and Chris** are step by step: say exactly which window to type in, one command per
  box, what to press and what they should see.
- **Each script** has a short changelog / version block at the top; bump it when you change the script.
- **Commits** end with the attribution lines given in the session. Don't commit data, models or large
  binaries (`data/`, `models/`, `runs/` are ignored); keep doc pictures small.
- Never print or copy tokens or secrets.
