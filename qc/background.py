"""The empty-spot reference: tells part from backdrop, and notices when the backdrop or lighting drifts.

Changelog
  0.1.0  2026-11-12  first version.
  0.2.0  2026-11-12  the reference remembers which lens calibration its pictures were corrected with (lens_id);
                     numbers marked [chosen] (audit L7).

Before a session (and every so often during it) the camera takes pictures of the EMPTY spot. Their
average is the reference; how much they differ from each other is the camera's own noise. Both are
used, so nothing here is a fixed brightness number:

  part mask    pixels that differ from the reference by more than the noise ever does on an empty
               frame, and more than the split between "dark" and "bright" differences in this frame
               (Otsu's method). The biggest blob is the part; its holes are filled for the outline.
  drift check  the same difference, smoothed, away from the part. Limits come from the empty frames
               themselves: each empty frame is compared with the average of the others (leave one out),
               and the biggest change seen that way is what "no drift" looks like. A change bigger than
               DRIFT_FACTOR x that is reported as drift.

    ref = EmptyReference.from_frames([empty1, empty2, ...])
    loc = ref.part_mask(frame)                 # Locate(mask, filled, contour, centre, area) or None
    d = ref.drift(frame, exclude=loc.filled)   # dict(drift=bool, local=..., global_=..., ...)
    ref.save("models/empty.npz"); ref = EmptyReference.load("models/empty.npz")
"""
import collections
import json

import cv2
import numpy as np

DRIFT_FACTOR = 1.5      # [chosen] margin over the largest change seen between empty frames; DOE-able
DRIFT_SMOOTH = 3.0      # [chosen] px; smoothing before the drift check, so single noisy pixels never count
MIN_PART_AREA = 400     # [chosen] px; anything smaller is dust, not a part
EXCLUDE_GROW = 15       # [chosen] px; the part's own edge blur and shadow are not drift

Locate = collections.namedtuple("Locate", "mask filled contour centre area threshold")


def _absdiff(frame, mean):
    return np.abs(frame.astype(np.float32) - mean).max(axis=2)


class EmptyReference:
    def __init__(self, mean, noise_level, local_ref, global_ref, n_frames, still_noise=None, lens_id=None):
        self.lens_id = lens_id
        self.mean = mean.astype(np.float32)
        self.noise_level = float(noise_level)     # brightest pixel difference an empty frame shows
        self.local_ref = float(local_ref)         # biggest smoothed local change between empty frames
        self.global_ref = float(global_ref)       # biggest overall brightness change between empty frames
        self.n_frames = int(n_frames)
        # mean change between two pictures of a still scene (capture.motion's measure): the stillness check's unit
        self.still_noise = float(still_noise) if still_noise is not None else self.noise_level / 4.0

    # ------------------------------------------------------------------------------------- building
    @classmethod
    def from_frames(cls, frames, noise_quantile=0.999):
        st = np.stack([f.astype(np.float32) for f in frames])
        n = len(st)
        mean = st.mean(axis=0)
        if n >= 2:
            noise, local, glob = [], [], []
            for i in range(n):
                others = (mean * n - st[i]) / (n - 1)
                d = _absdiff(st[i], others)
                noise.append(np.quantile(d, noise_quantile))
                local.append(cv2.GaussianBlur(d, (0, 0), DRIFT_SMOOTH).max())
                glob.append(abs(float((st[i] - others).mean())))
            # a leave-one-out difference holds the noise of both sides; the live comparison is with
            # the mean of all n, so the same numbers are a slightly cautious (high) estimate
            pair = []
            for i in range(n - 1):
                a = cv2.GaussianBlur(st[i], (0, 0), 1.0)
                b = cv2.GaussianBlur(st[i + 1], (0, 0), 1.0)
                pair.append(float(np.abs(a - b).max(axis=2).mean()))
            return cls(mean, max(noise), max(local), max(glob), n, max(pair))
        # one frame only: estimate the sensor noise from the frame itself (median absolute deviation
        # of a high-pass), and say so in .n_frames == 1
        g = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
        hp = g - cv2.GaussianBlur(g, (0, 0), 2.0)
        sd = 1.4826 * np.median(np.abs(hp - np.median(hp))) * 1.6
        return cls(mean, 4.0 * sd, 2.0 * sd, 0.5 * sd, 1)     # [chosen] factors for the one-frame fallback; not DOE

    def save(self, path):
        np.savez_compressed(path, mean=self.mean.astype(np.float16), meta=json.dumps(dict(
            noise_level=self.noise_level, local_ref=self.local_ref, global_ref=self.global_ref,
            n_frames=self.n_frames, still_noise=self.still_noise, lens_id=self.lens_id)))
        return path

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        return cls(z["mean"].astype(np.float32), m["noise_level"], m["local_ref"], m["global_ref"], m["n_frames"],
                   m.get("still_noise"), m.get("lens_id"))

    # ------------------------------------------------------------------------------------- part mask
    def difference(self, frame):
        return cv2.GaussianBlur(_absdiff(frame, self.mean), (0, 0), 1.0)

    def part_mask(self, frame, near=None, search_radius=None):
        """The part's mask. near=(x, y) and search_radius limit the search (used when the optional
        detector found the part first). Returns Locate or None when nothing part-sized is there."""
        d = self.difference(frame)
        d8 = np.clip(d, 0, 255).astype(np.uint8)
        otsu, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = max(float(otsu), self.noise_level * 1.0 + 1.0)     # [chosen] +1 grey level above the noise; not DOE
        m = (d > t).astype(np.uint8)
        if near is not None and search_radius:
            win = np.zeros_like(m)
            cv2.circle(win, (int(near[0]), int(near[1])), int(search_radius), 1, -1)
            m &= win
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, cents = cv2.connectedComponentsWithStats(m, 8)
        if n <= 1:
            return None
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[k, cv2.CC_STAT_AREA] < MIN_PART_AREA:
            return None
        mask = (lab == k).astype(np.uint8)
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, [c], -1, 1, -1)
        mo = cv2.moments(filled, True)
        centre = (mo["m10"] / mo["m00"], mo["m01"] / mo["m00"])
        return Locate(mask, filled, c, centre, float(mo["m00"]), t)

    # ------------------------------------------------------------------------------------- drift
    def drift(self, frame, exclude=None, factor=DRIFT_FACTOR):
        """Has the backdrop or the lighting changed since the reference was taken? exclude: a mask of
        pixels to ignore (the part). Returns dict(drift, local, global_, local_limit, global_limit,
        where (x, y) of the biggest change, reasons)."""
        f = frame.astype(np.float32)
        d = cv2.GaussianBlur(_absdiff(f, self.mean), (0, 0), DRIFT_SMOOTH)
        signed = (f - self.mean).mean(axis=2)
        keep = np.ones(d.shape, bool)
        if exclude is not None:
            grow = cv2.dilate(exclude.astype(np.uint8), np.ones((2 * EXCLUDE_GROW + 1,) * 2, np.uint8))
            keep = grow == 0
        local = float(d[keep].max()) if keep.any() else 0.0
        glob = abs(float(signed[keep].mean())) if keep.any() else 0.0
        # a floor of one grey level each: identical empty frames (a perfect simulator) would otherwise
        # give a zero limit and call every speck drift
        llim = factor * max(self.local_ref, 1.0)               # [chosen] floors 1.0 / 0.25 grey levels; not DOE
        glim = factor * max(self.global_ref, 0.25)
        reasons = []
        if local > llim:
            reasons.append("something changed in one area (%.1f grey levels, limit %.1f)" % (local, llim))
        if glob > glim:
            reasons.append("overall brightness changed (%.2f grey levels, limit %.2f)" % (glob, glim))
        dd = np.where(keep, d, 0)
        yx = np.unravel_index(int(np.argmax(dd)), dd.shape)
        return dict(drift=bool(reasons), local=local, global_=glob, local_limit=llim, global_limit=glim,
                    where=(int(yx[1]), int(yx[0])), reasons=reasons)
