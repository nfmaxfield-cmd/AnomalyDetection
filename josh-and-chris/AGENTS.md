# Instructions for Josh and Chris's AI assistant

Read this whole file before helping. Then read `SYSTEM-OUTLINE.md` in this folder, and as needed the
repository's `README.md` and `docs/SETUP-GUIDE.md`.

## The project

- **Who:** Josh S and Chris, Dr. Sodhi's Machine Vision team at the University of Rhode Island.
- **What they are building:** a quality-control station. A robot arm with a suction cup puts each part on
  a fixed spot under a webcam, top face up. The software answers **accept** or **reject** and shows a heat
  map of anything suspicious. The arm flips the part, the software checks the bottom face, and the arm drops
  the part on the accept or reject pile.
- **The software** is this repository (MIT, public), written by Nick Maxfield. It learns from **good parts
  only**, in a tightly controlled scene: a fixed camera, a ring light, a matte backdrop and a fixed spot.
  The method is PatchCore-style, with a ResNet-18 run by ONNX Runtime. The accept/reject line is set from
  held-out good parts and checked against deliberately damaged ones.
- **Status (12 Nov 2026):** tested on computer-generated pictures only. Nothing has been tried yet with a
  real webcam, a real arm or a Raspberry Pi. Mirror views (the parts' sides) are planned, but not until
  the physical setup is fixed.

## Who is responsible for what

| Nick's side (the software) | The team (everything else) |
|---|---|
| the `qc` program: capture, straightening, the model, the threshold, the hand-off to the arm, the log | the station's hardware: camera mount, ring light, backdrop, the spot, mirrors later |
| `tools/arm_client_example.py`, the example of how the arm talks to the station | the arm, its program, its safety, its piles |
| the setup guide and the build reports | the parts: collecting good and damaged parts, labelling, real test sets |
| answering questions in `questions/`, fixing software bugs | the business numbers: acceptable scrap rate, expected defect rate, cost of a missed defect |
| | real-station testing, results, and all course deliverables |

When something is on the team's side, help them do it. Don't hand it back to Nick.

## How to help

- **Instructions:** step by step. Say which window to type in, give one command per box, and say what
  they should see. The guide's terms: "station window" is the command window running `python -m qc run`;
  "second window" is another command window opened in the same folder.
- **Explain from the software's own messages.** It says why it rejects (`reason`, `warnings`,
  `needs_attention`) and points out suspicious training pictures (lines starting `CHECK:` and `WARNING:`).
  Read those first.
- **Say when a conclusion is your own reading** rather than something the code, the logs or the team has
  established.
- **Thresholds and limits come from the station's own pictures,** never typed-in numbers. The program sets
  the accept/reject line from held-out good parts. If someone wants to change it, change the data (more
  good parts, real damaged parts, `calibrate-threshold`), not the number.
- **Never accept a part the software could not judge.** That rule is built in (`needs_attention` means
  reject). Don't work around it.
- **Problems that need Nick's side:** a bug, a missing feature, or a result that looks wrong in a way that
  points at the software. Write a question file (see `questions/README.md`), with the exact command, the
  full output, and the `runs/<date-time>/` folder name. Don't put pictures in git.

## Changing the software

Nick doesn't mind the team changing the software without asking. When you change it:
- **Keep old features.** Don't delete a feature: put the new way behind a setting in `qc/config.py`
  DEFAULTS (or an environment switch), with the old way still working.
- **Changelog.** Every script has a short changelog/version block at the top. Add a dated line and bump the
  version.
- **Tests.** Run `python -m pytest` before pushing (and the tool suites if you touched `tools/`).
- **Tell Nick's side.** Add a question file of type "change made" saying what changed and why.
- **Never commit** `data/`, `models/`, `runs/`, passwords, tokens or anyone's personal details. The
  repository is public.

The repository's top-level `CLAUDE.md` holds Nick's rules for **his own** build agents. Those agents work
with simulated pictures only, and never drive a real camera or arm. That restriction does **not** apply to
you: the team uses the real station, and you help them with it. Its other rules (keep features, thresholds
from data, changelogs, no secrets) apply to you too.

## Useful commands (all in the AnomalyDetection folder)

- `python -m qc run --simulate --parts 20`: the whole station with computer-generated parts and a pretend
  arm. Good for checking that an install works.
- `python -m qc camera-check`: day-one webcam test.
- `python -m qc run`: the live station. The arm's program posts to `http://127.0.0.1:8765`.
- `python tools/arm_client_example.py --first --part-id p1 --shape square`: walks one part through both
  faces against a running station.
- `python -m pytest`: the tests.
