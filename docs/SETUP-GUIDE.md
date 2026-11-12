# Setup guide for the QC station

For **Josh S and Chris** (Dr. Sodhi's Machine Vision team, URI). Dr. Sodhi is welcome to evaluate the
work together with you. Written by Nick Maxfield; the software is free and public (MIT licence).

What you will have at the end: the arm puts a part on a fixed spot, the webcam looks at it, the program
answers **accept** or **reject** (plus a heat map of the suspicious spot), the arm flips the part, the
program checks the other face, and the arm drops the part on the right pile. Every decision is logged.

Words used below:
- **Command window**: on Windows, the **Command Prompt** (Start menu, type `cmd`, press Enter). On a Mac,
  **Terminal** (Spotlight, type `terminal`). Type each command in the grey boxes, one box at a time, then
  press **Enter**.
- **The folder**: the AnomalyDetection folder (the one holding `README.md` and the `qc` folder).

---

## Step 1. Install (once, about 10 minutes)

1. Install **Python 3.10 or newer** from python.org. On Windows tick **"Add python.exe to PATH"** on the
   first screen of the installer.
2. Get the folder: on github.com/nfmaxfield-cmd/AnomalyDetection click **Code > Download ZIP**, unzip it
   to your Desktop.
3. Open a command window and go into the folder (Windows example; change the path to where it is):

   ```
   cd %USERPROFILE%\Desktop\AnomalyDetection-main
   ```

   On a Mac:

   ```
   cd ~/Desktop/AnomalyDetection-main
   ```

4. Make a private Python environment for it:

   ```
   python -m venv venv
   ```

   (On a Mac type `python3` instead of `python` in this step only.)
5. Switch it on. Windows:

   ```
   venv\Scripts\activate
   ```

   Mac:

   ```
   source venv/bin/activate
   ```

   You should now see `(venv)` at the start of the line. **Do this step 5 every time you open a new
   command window for the station.**
6. Install what the program needs:

   ```
   python -m pip install -r requirements.txt
   ```

7. Download the pretrained picture network (45 MB, once):

   ```
   python tools/get_backbone.py
   ```

   You should see `Saved ...resnet18_l2l3.onnx (11.2 MB)`.
8. Check that everything works:

   ```
   python -m pytest
   ```

   After a minute or two the last line should say `... passed` in green (a few "skipped" are fine).

## Step 2. Camera: mount it and lock its settings

1. Mount the webcam **rigidly, pointing straight down** at the spot where the arm puts parts. Nothing
   about it may move once you start collecting pictures (tape or clamp the cable too).
2. Height: our suggestion is that the **largest part fills about a third to half of the picture's width**
   (closer = more pixels on small scratches; leave room for the part to land a little off-centre).
3. A webcam that allows **manual focus and manual exposure** is needed (most Logitech C920-class cameras
   do). Phone cameras are not used.
4. Put a good part on the spot, switch the ring light on (step 3), then in the command window:

   ```
   python -m qc camera-setup
   ```

   A window with the live picture and three sliders opens. Move **focus** until the text "sharpness"
   is highest. Move **exposure** down until no part of the part turns magenta (magenta = too bright).
   Set **white balance** so a white sheet looks white. Press **S** to save, then **Q** to close.
   The settings are now locked in `qc_config.json` and used every time.
   (This screen was written without a camera to test it on: if a slider does nothing on your webcam,
   tell Nick which webcam it is.)

## Step 3. Ring light

- Fix the ring light **around the lens**, so the light comes from the camera's direction.
- Set it to **one brightness and never change it** during collecting and running. If it has a dimmer,
  use it near full power (some LED dimmers flicker at low settings).
- Keep other light changes away from the spot: close blinds, don't park a desk lamp next to it.

## Step 4. Matte black backdrop

- Cover the spot and everything the camera sees with **matte black** material (black felt or flocked
  paper; not glossy card, not black plastic, which reflect the ring light).
- Optional, our suggestion: two **small white dots** (a hole-punch circle) in two corners of the picture,
  away from the part. They let the program notice if the camera gets bumped.

## Step 5. Mirrors (optional, later)

Mirrors at the side can show a part's edges. **This first version does not inspect the mirror views yet**:
mirrors may stay in place (take the empty picture with them in place), but their views are ignored for
now. Edge damage that shows from above (chips, missing corners) is checked already.

## Step 6. Two signals from the arm's code: "empty" and "paused"

The program waits for messages from the arm's code. Start it in its own command window (do step 1.5
first):

```
python -m qc run
```

It says `Listening on http://127.0.0.1:8765`. Leave this window open. The arm's code then sends:

| When | Send | The reply |
|---|---|---|
| Start of a session, spot empty, arm out of view | `empty` | `{"ok": true, ...}` |
| Part resting at the spot, arm out of view, top face up | `paused` with `part_id`, `shape`, `face: "top"` | the decision |
| After the arm flipped it | `paused` with the same `part_id`, `face: "bottom"` | the decision, plus `part_decision` for the whole part |
| Whenever a reply says `"empty_due": true` | `empty` again (spot empty) | |

The decision reply contains `"decision": "accept"` or `"reject"`. Put the part on the **reject** pile if
either face says reject (`part_decision` says this for you after the second face). If the program could
not judge a picture (no part found, part still moving, the backdrop changed) the decision is `reject`,
`needs_attention` is `true` and `reason` says why.

`tools/arm_client_example.py` holds the exact code to copy (about 20 lines of Python). To try it, with
`python -m qc run` open in the first window, type in a second window (step 1.5 first):

```
python tools/arm_client_example.py --first --part-id p1 --shape square
```

Only send `paused` when the part has stopped and the arm is out of the picture: the program keeps the
sharpest picture of a short burst, and says so if the part was still moving.

## Step 7. Collect the empty picture and the good parts

1. Spot empty, arm out of view, ring light on:

   ```
   python -m qc collect-empty
   ```

   Press Enter when asked.
2. Run **good parts only**. Start the collector instead of `run`:

   ```
   python -m qc collect-good
   ```

   Your arm code stays the same (it sends `paused` for the top, flips, sends `paused` for the bottom);
   every picture is saved as a good example and the reply is always `accept`. Press **Ctrl+C** in the
   window when done.
3. **How many?** Start with **50 to 100 per shape per face** (our estimate). The data decides whether
   that is enough: after training (step 8), run

   ```
   python -m qc learning-curve --shape square --face top
   ```

   It trains with 5, 10, 20, 40, 80 pictures and shows how well damaged parts stand out at each. The last
   line says either "N good pictures: ..." (enough) or "Still improving: collect more". The chart is
   saved as `models/square/top/learning_curve.png`.

## Step 8. Train

```
python -m qc train
```

One model per shape and face, from the pictures in `data/good/`. A fifth of the pictures are kept back
to set the accept/reject line from data; the line is printed at the end with how many held-back good
parts it would reject.

## Step 9. Damaged parts: 10 to 20 per shape (for the threshold)

The accept/reject line is never a number we pick: it comes from the scores of good parts the model did
not train on, and gets much better with **10 to 20 deliberately damaged parts per shape and face**
(scratches, chips, dents, a cut corner, stains: whatever real damage looks like). For each one:

```
python -m qc collect-damaged --shape square --face top --manual --defect scratch
```

It asks you to put each damaged part on the spot and press Enter. Then:

```
python -m qc calibrate-threshold
```

It prints how many damaged parts are caught and how many good ones would be scrapped.

Another way: after a run, open the rejected pictures in the labelling page, mark each **good** or
**defect** (and draw a box around the damage), then copy them into the right folders:

```
python tools/qclabel/app.py runs/<the run's folder>/images --shape square --face top
```

```
python tools/qclabel/app.py runs/<the run's folder>/images --export . --shape square --face top
```

## Step 10. Run

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

## Trying it without any hardware

Everything above also works with computer-generated pictures and a pretend arm; add `--simulate`:

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
