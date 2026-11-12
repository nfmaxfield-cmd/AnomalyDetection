# Setup guide for the QC station

For **Josh S and Chris** (Dr. Sodhi's Machine Vision team, URI). Dr. Sodhi is welcome to evaluate the
work together with you. Written by Nick Maxfield; the software is free and public (MIT licence).
Version 0.2 (12 Nov 2026, after the independent audit; what changed: `docs/build-reports/2026-11-12-audit-fixes.md`).

What you will have at the end: the arm puts a part on a fixed spot, the webcam looks at it, the program
answers **accept** or **reject** (plus a heat map of the suspicious spot), the arm flips the part, the
program checks the other face, and the arm drops the part on the right pile. Every decision is logged.

Words used below:
- **Command window**: on Windows, the **Command Prompt** (Start menu, type `cmd`, press Enter). On Linux
  (and the Raspberry Pi), **Terminal**. Type each command in the grey boxes, one box at a time, then press
  **Enter**.
- **The folder**: the AnomalyDetection folder (the one holding `README.md` and the `qc` folder).
- **Station window**: the command window where `python -m qc run` is running. **Second window**: another
  command window, opened the same way (do step 1.5 in it first).

Order of the steps: install (1), build the scene (2-5), lock and check the camera (6-7), calibrate the lens
with the printed card (8), connect the arm (9), collect pictures and train (10-12), run (13).
If something looks wrong, see **Troubleshooting** at the end.

---

## Step 1. Install (once, about 10 minutes)

1. Install **Python 3.10, 3.11, 3.12 or 3.13** from python.org (tested: 3.11 and 3.13). On Windows tick
   **"Add python.exe to PATH"** on the first screen of the installer. Use **Windows or Linux** for the
   station: on a Mac the webcam's exposure and focus cannot be locked (the program says so); a Mac is fine
   for trying the simulator only.
2. Get the folder: on github.com/nfmaxfield-cmd/AnomalyDetection click **Code > Download ZIP**, unzip it
   to your Desktop.
3. Open a command window and go into the folder (Windows example; change the path to where it is):

   ```
   cd %USERPROFILE%\Desktop\AnomalyDetection-main
   ```

   On Linux:

   ```
   cd ~/Desktop/AnomalyDetection-main
   ```

4. Make a private Python environment for it (on Linux type `python3` instead of `python` in this step only):

   ```
   python -m venv venv
   ```

5. Switch it on. Windows:

   ```
   venv\Scripts\activate
   ```

   Linux:

   ```
   source venv/bin/activate
   ```

   You should now see `(venv)` at the start of the line. **Do this step 5 every time you open a new
   command window for the station.**
6. Install what the program needs:

   ```
   python -m pip install -r requirements.txt
   ```

   The last line should start with `Successfully installed`.
7. Download the pretrained picture network (45 MB, once):

   ```
   python tools/get_backbone.py
   ```

   You should see `Saved ...resnet18_l2l3.onnx (11.2 MB)`. Keep a copy of the `models/backbone` folder
   somewhere safe: every trained model is tied to this exact file and refuses a different one.
8. Check that everything works:

   ```
   python -m pytest
   ```

   After a few minutes the last line should say `... passed` in green (a few "skipped" are fine).

## Step 2. Camera: mount it

1. Mount the webcam **rigidly, pointing straight down** at the spot where the arm puts parts. Nothing
   about it may move once you start collecting pictures (tape or clamp the cable too).
2. Height: the **largest part fills about a third to half of the picture's width** (closer = more pixels
   on small scratches; leave room for the part to land a little off-centre). **Every** shape should fill at
   least a third of the width: if your parts differ a lot in size, use one camera height per size class.
   Smaller parts lose small defects; the program warns after training when a part is small in the picture.
3. A webcam that allows **manual focus and manual exposure** is needed (most Logitech C920-class cameras
   do). Phone cameras are not used.
4. The arm's code and the station run **on the same computer** (the station listens on `127.0.0.1`, which
   only that computer can reach).

## Step 3. Ring light

- Fix the ring light **around the lens**, so the light comes from the camera's direction. This suits
  matte parts. **Flat shiny parts** reflect a ring light straight back as a bright patch: for those, use
  diffuse light (a white diffuser around the spot) or light from a low angle, and tell Nick.
- Set it to **one brightness and never change it** during collecting and running. If it has a dimmer,
  use it near full power (some LED dimmers flicker at low settings).
- Keep other light changes away from the spot: close blinds, don't park a desk lamp next to it.

## Step 4. Backdrop and corner dots

- Cover the spot and everything the camera sees with **matte black** material (black felt or flocked
  paper; not glossy card, not black plastic, which reflect the ring light).
- **Dark parts** (black or dark grey) disappear on a black backdrop: use a **light grey** matte backdrop
  for them instead. The program says so when it cannot find a part, or finds it hard to see.
- Put two **small white dots** (a hole-punch circle) in two corners of the picture, away from the part.
  They let the program notice if the camera gets bumped. Leave them there for good.

## Step 5. Mirrors: cover them for now

This version does **not** handle mirror views: a mirror that shows the part's side makes every part look
like a changed backdrop, so every part would be rejected. **Remove the mirrors, or cover them with the
backdrop material**, before taking any pictures. Mirror views are planned for a later build, once the
physical setup and mirror positions are fixed. Edge damage that shows from above (chips, missing corners)
is checked already.

## Step 6. Camera settings: lock exposure, gain, focus and white balance

1. Put a good part on the spot, switch the ring light on. In a command window (step 1.5 first):

   ```
   python -m qc camera-setup
   ```

2. A window with the live picture and four sliders opens. Move **focus** until the green text "sharpness"
   is highest. Move **exposure** down until no part of the part turns magenta (magenta = too bright); then
   **gain** as low as possible while the part still looks clearly lit. Set **white balance** so a white
   sheet looks white.
3. Press **S** to save (the command window prints `Saved to qc_config.json: ...`), then **Q** to close.
   The settings are now used every time.

The exposure slider's numbers depend on the system: on Windows they run from -13 to 0, on Linux from 1 to
2000. Step 7 checks that the camera really keeps what you set.

## Step 7. Camera check (day one, about 3 minutes)

This tests the things the simulator can't: whether your webcam keeps its settings, and whether a burst of
pictures can contain old pictures from before the arm said "paused".

```
python -m qc camera-check
```

It walks you through four checks. Do exactly what each line says, then press **Enter**:

1. It lists each setting as `asked -> what the camera reports` with `held` or `DID NOT HOLD`.
2. It closes and opens the camera again and says whether the settings survived.
3. **Old frames** (twice): hold your hand under the camera and press Enter; then take your hand fully out
   of view and press Enter straight away. You should see `no old frames` for the first run ("reader
   thread"). The second run ("plain reads, no drain") uses the first build's way of reading and just shows whether your
   camera keeps old frames in a buffer (frames listed there are expected on many webcams; it is not a problem).
4. **Automatic exposure**: clear the view and press Enter; then lay a **white sheet** over the **left half**
   of the view and press Enter. You should see `exposure held`.

The last lines say `RESULT: all checks passed`, or list the problems. The result is saved as
`runs/camera_check_<date_time>.json`: **if anything failed, send that file to Nick** with the webcam's
model name.

## Step 8. Lens calibration with the printed checkerboard card (once)

A webcam lens bends straight lines a little, most near the edges, and the program needs to know how big a
pixel is at the spot. A printed checkerboard card fixes both, once.

1. Make the card (it is saved in the folder as `calibration_card.pdf` and `.png`):

   ```
   python -m qc make-card
   ```

   (For US Letter paper add `--paper letter`.)
2. Print `calibration_card.pdf` at **100 % / "actual size"** (not "fit to page"). With a ruler, measure
   the line under the squares: it must be **100 mm**. If it isn't, the printer scaled the page: print again.
   Glue the page **flat** onto stiff card (foam board or a clipboard). The square size (20 mm) is printed on it.
3. Ring light on, mirrors covered, then:

   ```
   python -m qc calibrate-camera
   ```

4. It asks for 12 pictures. For each one, hold or prop the card **over the spot, the whole card in the
   picture, tilted the way the line says** (left, right, away, towards you ...; about 10-30 degrees), keep it
   still, and press **Enter**. After each picture you should see `card found`. If you see `card NOT found`,
   carry on: that one just doesn't count (next time keep the whole card in view and less steep).
5. Last, it asks you to lay the card **flat on the spot where parts rest** and press Enter.
6. You should see the reprojection error (how far, in pixels, the fitted lens puts each corner from
   where it was seen), its accept limit (worked out from how much the corners jitter between pictures of
   the still card), the coverage, the millimetres per pixel at the spot, and `Saved models/camera_calibration.json`.
   If it says `NOT saved`, it lists why (too few pictures with the card found, tilts too alike, card not flat
   or moved): fix that and run step 8.3 again.

From now on every picture is corrected for the lens, and defect sizes are given in mm. **If you calibrate
again later** (camera moved or refocused), take a new empty picture and train again (the program refuses
to judge with a model trained under another calibration and says so).

## Step 9. Two signals from the arm's code: "empty" and "paused"

The program waits for messages from the arm's code. Start it in its own command window (step 1.5 first):

```
python -m qc run
```

It says `Listening on http://127.0.0.1:8765`. Leave this window open (the **station window**). Any
`CAMERA WARNING` lines here mean a setting did not hold: run step 7. The arm's code then sends:

| When | Send | The reply |
|---|---|---|
| Start of a session, spot empty, arm out of view | `empty` | `{"ok": true, ...}` |
| Part resting at the spot, arm out of view, top face up | `paused` with `part_id`, `shape`, `face: "top"` | the decision |
| After the arm flipped it | `paused` with the same `part_id`, `face: "bottom"` | the decision, plus `part_decision` for the whole part |
| Whenever a reply says `"empty_due": true` | `empty` again (spot empty) | |

How the arm's code sends a message: an HTTP POST of a small JSON text to `http://127.0.0.1:8765/paused`
(or `/empty`) on the same computer. `tools/arm_client_example.py` holds the exact Python code to copy
(the functions `ask`, `ask_face`, `inspect_part` and `send_empty`).

What the reply means:
- `"decision": "accept"` or `"reject"` for this face. Put the part on the **accept** pile only when the
  last reply says `"part_decision": "accept"`; everything else goes to **reject**.
- `"needs_attention": true`: the program could not judge this picture properly (part still moving,
  backdrop or light changed, picture blurrier than the training pictures, part at the picture's edge,
  wrong face up or wrong shape). The decision is then **always `reject`**, and `reason` says why: a person
  should read it.
- `"recheck": true`: waiting a moment and sending `paused` again **for the same part and face** may help
  (e.g. the part was still moving). The new answer replaces the old one. The example tries twice.
- A reply to `empty` with `"ok": false` and `"refused": true`: something is still in view (a part, a hand,
  the arm, a tool). Clear the spot and send `empty` again; the old empty picture is kept meanwhile. Only
  after a **deliberate** change of backdrop, light or camera, send `{"force": true}` with it (and train
  again).
- Use a new `part_id` for every part. If an id comes again after its part was finished, the program
  starts a new part with it.

To try it: with `python -m qc run` open in the station window, type in a second window:

```
python tools/arm_client_example.py --first --part-id p1 --shape square
```

Only send `paused` when the part has stopped and the arm is out of the picture.

## Step 10. Collect the empty picture and the good parts

1. Spot empty, arm out of view, ring light on:

   ```
   python -m qc collect-empty
   ```

   Press Enter when asked. You should see `"ok": true`.
2. Run **good parts only**. Start the collector instead of `run`:

   ```
   python -m qc collect-good
   ```

   Your arm code stays the same (it sends `paused` for the top, flips, sends `paused` for the bottom);
   every picture is saved as a good example and the reply is always `accept`. A picture with no part, a
   part at the picture's edge, a part still moving or a changed backdrop is **not kept** (the reply says
   `"ok": false` and why). Press **Ctrl+C** in the window when done.
3. **How many?** **50 to 100 different parts per shape per face, each photographed once** (the same part
   photographed ten times teaches the program that part, not "good parts"). 20 of them are kept back to set
   the accept/reject line, so fewer than 40 makes that line rough. The data decides whether it is enough:
   after training (step 11), run

   ```
   python -m qc learning-curve --shape square --face top
   ```

   It trains with 5, 10, 20, 40, 80 pictures and watches the accept/reject line set from good parts only:
   it drops while more variety is seen and settles when there is enough. The last line says either
   `Enough at N good pictures: ...` or `Not settled by N pictures ...: collect more good pictures`. Before
   you have damaged parts it paints stand-in damage on good pictures and says so. The chart is saved as
   `models/square/top/learning_curve.png`.

## Step 11. Train

```
python -m qc train
```

One model per shape and face, from the pictures in `data/good/`. For each you should see `Training
square/top on N good pictures (20 held back ...)`, `Symmetry of the outline: ...`, then `Threshold ...`
with how many held-back good parts it would reject. Read any line starting with:
- `WARNING:` the part is small in the picture, hard to see on the backdrop, or not found in some pictures
  (what to do is in the line).
- `CHECK:` a training picture looks unlike the other good ones (it is named): open it; if it shows a damaged
  part, an empty spot or another part, delete it from `data/good/...` and train again.

## Step 12. Damaged parts: 10 to 20 per shape and face (for the threshold)

The accept/reject line is never a number we pick: it comes from the scores of good parts the model did
not train on, and is checked with **10 to 20 deliberately damaged parts per shape and face** (scratches,
chips, dents, a cut corner, stains: whatever real damage looks like). **Photograph only the face that
carries the damage**: a part scratched on top goes in the `top` folder only. This command asks for 15
parts, one after another (`--n 20` for 20):

```
python -m qc collect-damaged --shape square --face top --manual --defect scratch
```

For each, put a damaged part on the spot **damaged face up**, hands away, press Enter; you should see
`saved ...`. Then:

```
python -m qc calibrate-threshold
```

It prints how many damaged parts are caught and how many good ones would be scrapped. A line starting with
`CHECK:` names damaged pictures that score like good parts: open each one's heat map; if the damage isn't
visible on that face, move the picture to the right face's folder or delete it, and run the command again.

Another way: after a run, open the rejected pictures in the labelling page, mark each **good** or
**defect** (and draw a box around the damage), then copy them into the right folders:

```
python tools/qclabel/app.py runs/<the run's folder>/images --shape square --face top
```

```
python tools/qclabel/app.py runs/<the run's folder>/images --export . --shape square --face top
```

## Step 13. Run

```
python -m qc run
```

Every decision is saved in `runs/<date_time>/`: `decisions.csv` (opens in Excel), the pictures, and a
heat map for each part (`..._heat.png`: left the straightened part, right the same with colour where it
differs from good parts; red = at or above the reject line).

To judge a single picture file by hand:

```
python -m qc inspect path/to/picture.png --shape square --face top
```

## Troubleshooting

| What you see | What to do |
|---|---|
| Every part is rejected | Read `reason` in the replies (or the `reason` column of `decisions.csv`). "background drift": something changed on the backdrop or in the light: put it back, or send `empty` again. "another camera calibration": train again. No reason: the parts differ from training (another batch?): collect and train again. |
| "no part found at the spot" | Is the part on the spot, inside the picture? Dark part on the black backdrop: use a light grey backdrop (step 4), then new empty picture and training. |
| "part was still moving" again and again | Send `paused` a little later after the arm releases the part (or raise `burst_settle_s` in `qc_config.json`). |
| "picture blurrier than every training picture" | Focus moved, the part is taller than the training parts, or it was still moving: run step 6 again (and train again if you refocus). |
| "this face does not match the trained face" | Wrong face up, wrong shape sent, or a large defect. Check the part. |
| `empty` refused | Something is still in view: the reply says where (x, y in pixels). |
| CAMERA WARNING / camera-check problems | Send `runs/camera_check_...json` to Nick with the webcam's model. |
| After moving the camera, the light or the backdrop | Steps 6-8 again if the camera moved, then a new empty picture (step 10.1) and train again (step 11). |

## Raspberry Pi

Not tested on a Raspberry Pi yet. Raspberry Pi OS "Bookworm" (64-bit) ships Python 3.11, which works
(step 1 on Linux). Use a Pi 5 with 4 GB or more; time it before relying on it.

## Trying it without any hardware

Everything above also works with computer-generated pictures and a pretend arm; add `--simulate`
(`--scene easy` gives the first build's simpler pictures):

```
python -m qc collect-empty --simulate
```

```
python -m qc collect-good --simulate --shape square --n 60
```

```
python -m qc train --shape square
```

```
python -m qc run --simulate --parts 20
```

## Coming later

- **Mirror views** of the part's sides, once the physical setup and the mirror positions are fixed.
