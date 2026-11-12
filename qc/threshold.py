"""Where to draw the line between accept and reject, worked out from scores of real pictures.

Changelog
  0.1.0  2026-11-12  first version.

Never a fixed number: the threshold comes from
  (a) scores of GOOD parts the model was not trained on (held out), and
  (b) when available, scores of 10-20 deliberately DAMAGED parts per shape and face.

Rule (a) alone: a good part's score is the maximum over many cells, so the scores of good parts are fitted
with a Gumbel (extreme-value) distribution, and the threshold is where that fit says only
target_false_reject of good parts would score higher (default 1 %: a business choice, ask the team).
With few held-out parts this extrapolates past the highest score seen; the fit is what makes that
possible, and the report says how many parts it rests on.

Rule (b) with damaged parts: start from rule (a). If every damaged part scores above that line, keep
it (it is the LOWEST line that keeps false rejects at the target, so it leaves the most room for damage
milder than any seen). If some damaged parts would pass, move the line to where the expected cost
cost_bad_accepted x (share of damaged parts accepted) + cost_good_rejected x (share of good parts rejected,
from the fit) is smallest, and say so in the result.

    t = from_scores(good_scores, bad_scores, target_false_reject=0.01, cost_bad_accepted=10, cost_good_rejected=1)
    t["threshold"], t["method"], t["est_false_reject"], t["damaged_caught"], t["text"]
"""
import math

import numpy as np
from scipy import stats


def wilson(k, n, z=1.96):
    """95 % interval for a share k / n (Wilson score interval); (0, 1) when n == 0."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fit_good(good):
    g = np.asarray(good, float)
    if len(g) < 3 or g.std() == 0:
        return None
    loc, scale = stats.gumbel_r.fit(g)
    return loc, scale


def good_tail(fit, t):
    """Share of good parts the fit expects above t."""
    if fit is None:
        return float("nan")
    return float(stats.gumbel_r.sf(t, *fit))


def from_scores(good, bad=None, target_false_reject=0.01, cost_bad_accepted=10.0, cost_good_rejected=1.0):
    good = np.asarray(good, float)
    bad = np.asarray(bad if bad is not None else [], float)
    if len(good) < 3:
        raise ValueError("need at least 3 held-out good pictures to set a threshold (have %d)" % len(good))
    fit = fit_good(good)
    t_good = float(stats.gumbel_r.isf(target_false_reject, *fit)) if fit else float(good.max())
    t_good = max(t_good, float(good.max()))            # never below a good part we have actually seen
    method = "good parts only: Gumbel fit to %d held-out good scores, %.1f %% expected false rejects" % (
        len(good), 100 * target_false_reject)
    t = t_good
    if len(bad):
        if (bad > t_good).all():
            method += "; all %d damaged parts score above it" % len(bad)
        else:
            cands = np.unique(np.concatenate([good, bad, [t_good]]))
            mids = np.concatenate([[cands[0] - 1e-6], (cands[:-1] + cands[1:]) / 2, [cands[-1] + 1e-6]])
            mids = mids[mids >= float(np.median(good))]
            costs = [cost_bad_accepted * (bad <= m).mean() + cost_good_rejected * good_tail(fit, m) for m in mids]
            t = float(mids[int(np.argmin(costs))])
            method = ("cost balance: %d held-out good and %d damaged parts, a damaged part let through counts "
                      "%.0fx a good part scrapped" % (len(good), len(bad), cost_bad_accepted / cost_good_rejected))
    fr_seen = int((good > t).sum())
    caught = int((bad > t).sum())
    res = dict(threshold=t, method=method, n_good=int(len(good)), n_bad=int(len(bad)),
               est_false_reject=good_tail(fit, t), false_reject_seen=fr_seen,
               false_reject_seen_ci=wilson(fr_seen, len(good)),
               damaged_caught=caught, damaged_caught_ci=wilson(caught, len(bad)) if len(bad) else None,
               good_max=float(good.max()), good_median=float(np.median(good)),
               bad_min=float(bad.min()) if len(bad) else None, gumbel=fit,
               target_false_reject=target_false_reject)
    lines = ["Threshold %.3f (%s)." % (t, method),
             "Held-out good parts: %d of %d would be rejected; the fit expects %.2f %% in the long run." % (
                 fr_seen, len(good), 100 * res["est_false_reject"])]
    if len(bad):
        lo, hi = res["damaged_caught_ci"]
        lines.append("Damaged parts: %d of %d caught (95 %% range %.0f-%.0f %%)." % (caught, len(bad), 100 * lo, 100 * hi))
        if res["bad_min"] is not None and res["bad_min"] > res["good_max"]:
            lines.append("Every damaged part scores above every good part (gap %.3f to %.3f)." % (
                res["good_max"], res["bad_min"]))
    else:
        lines.append("No damaged parts yet: collect 10-20 per shape and face and run calibrate-threshold again.")
    res["text"] = " ".join(lines)
    return res
