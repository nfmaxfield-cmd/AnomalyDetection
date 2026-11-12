"""Where to draw the line between accept and reject, worked out from scores of real pictures.

Changelog
  0.1.0  2026-11-12  first version.
  0.2.0  2026-11-12  audit fixes H2 / M8: the cost rule weighs by the expected defect rate and works in
                     counts per part made, not shares of the samples; the line never rises above the
                     good-only line and never falls below the false-reject cap; damaged pictures that score
                     like a good part are listed by name for a person to look at instead of moving the
                     line; a held-out good picture far above the others is set aside and listed.

Never a fixed number: the threshold comes from
  (a) scores of GOOD parts the model was not trained on (held out), and
  (b) when available, scores of 10-20 deliberately DAMAGED parts per shape and face.

Rule (a), good parts only. A good part's score is the maximum over many cells, so the held-out good
scores are fitted with a Gumbel (extreme-value) distribution, and the line goes where that fit says only
target_false_reject of good parts would score higher (default 1 %: Nick's starting point, to be confirmed
with the team). It is never below the highest held-out good score. Before fitting, the highest held-out
score is checked against a fit to the others: if a score that high would come up by chance less than
1 time in 100 among this many parts, that picture is set aside and named (it may be a damaged part or a
mix-up in the good folder); the line is then set without it, which errs towards rejecting.

Rule (b), with damaged parts. Damaged pictures that score no higher than a good part already seen are
"look like good": no line can catch them without scrapping good parts we have seen, so they are listed by
file name ("look at its heat map: is the damage visible on this face?") and do not move the line. For the
rest, if any would pass the good-only line, the line may move DOWN, but only as far as the false-reject
cap (max_false_reject) allows, and only when that lowers the expected cost per part made:

    cost(t) = defect_rate x cost_bad_accepted x (share of damaged parts accepted at t)
            + (1 - defect_rate) x cost_good_rejected x (share of good parts rejected at t, from the fit)

defect_rate (expected_defect_rate) is the share of parts that really are damaged; without it, ten damaged
samples would count as if every tenth part were damaged. The line never moves UP for damaged parts: the
good-only line is already the lowest line that keeps false rejects at the target, so it leaves the most
room for damage milder than any seen.

    t = from_scores(good_scores, bad_scores, target_false_reject=0.01, cost_bad_accepted=10, cost_good_rejected=1,
                    expected_defect_rate=0.05, max_false_reject=0.02, good_names=[...], bad_names=[...])
    t["threshold"], t["method"], t["est_false_reject"], t["damaged_caught"], t["look_like_good"], t["text"]
"""
import math

import numpy as np
from scipy import stats

OUTLIER_CHANCE = 0.01   # [chosen] a held-out good score this unlikely (given the others) is set aside; not DOE


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


def outlier_check(good):
    """Is the highest held-out good score far above what the others predict? Returns (index or None,
    chance): chance = how often the largest of len(good) parts would reach that score, by a Gumbel fit to
    the other scores."""
    g = np.asarray(good, float)
    if len(g) < 8:                       # too few to judge one of them against the rest
        return None, None
    i = int(np.argmax(g))
    fit = fit_good(np.delete(g, i))
    if fit is None:
        return None, None
    chance = 1.0 - (1.0 - good_tail(fit, g[i])) ** len(g)
    return (i if chance < OUTLIER_CHANCE else None), float(chance)


def from_scores(good, bad=None, target_false_reject=0.01, cost_bad_accepted=10.0, cost_good_rejected=1.0,
                expected_defect_rate=0.05, max_false_reject=None, good_names=None, bad_names=None):
    good = np.asarray(good, float)
    bad = np.asarray(bad if bad is not None else [], float)
    good_names = list(good_names) if good_names is not None else ["good #%d" % (k + 1) for k in range(len(good))]
    bad_names = list(bad_names) if bad_names is not None else ["damaged #%d" % (k + 1) for k in range(len(bad))]
    if max_false_reject is None:
        max_false_reject = target_false_reject
    max_false_reject = max(float(max_false_reject), float(target_false_reject))
    if len(good) < 3:
        raise ValueError("need at least 3 held-out good pictures to set a threshold (have %d)" % len(good))
    # held-out good picture far above the rest: set aside (the line is then set without it) and named
    oi, chance = outlier_check(good)
    set_aside = []
    if oi is not None:
        set_aside = [dict(name=good_names[oi], score=float(good[oi]), chance=chance)]
        good = np.delete(good, oi)
        good_names = good_names[:oi] + good_names[oi + 1:]
    fit = fit_good(good)
    t_good = float(stats.gumbel_r.isf(target_false_reject, *fit)) if fit else float(good.max())
    t_good = max(t_good, float(good.max()))            # never below a good part we have actually seen
    t_floor = max(float(stats.gumbel_r.isf(max_false_reject, *fit)) if fit else t_good, float(good.max()))
    method = "good parts only: Gumbel fit to %d held-out good scores, %.1f %% expected false rejects" % (
        len(good), 100 * target_false_reject)
    t = t_good
    gmax = float(good.max())
    like_good = [dict(name=n, score=float(s)) for n, s in zip(bad_names, bad) if s <= gmax]
    usable = np.array([s for s in bad if s > gmax], float)
    if len(bad):
        if (bad > t_good).all():
            method += "; all %d damaged parts score above it" % len(bad)
        elif len(usable) and (usable <= t_good).any():
            p = float(expected_defect_rate)

            def cost(x):
                return p * cost_bad_accepted * float((usable <= x).mean()) + (1 - p) * cost_good_rejected * good_tail(fit, x)
            # candidate lines: the good-only line, or just below a damaged score it would let through,
            # never lower than the false-reject cap allows
            cands = [t_good] + [float(b) - 1e-6 for b in usable if t_floor <= b - 1e-6 < t_good]
            costs = [cost(x) for x in cands]
            k = int(np.argmin(costs))
            t = cands[k]
            if k == 0:
                method += ("; %d damaged part(s) score below it, but catching them would scrap more good parts than "
                           "it saves (expected defect rate %.1f %%, a damaged part let through counts %.0fx a good "
                           "part scrapped, false-reject cap %.1f %%)" % (int((usable <= t_good).sum()), 100 * p,
                                                                          cost_bad_accepted / cost_good_rejected,
                                                                          100 * max_false_reject))
            else:
                method = ("cost balance: %d held-out good and %d damaged parts, expected defect rate %.1f %%, a damaged "
                          "part let through counts %.0fx a good part scrapped; false-reject cap %.1f %%" % (
                              len(good), len(usable), 100 * p, cost_bad_accepted / cost_good_rejected,
                              100 * max_false_reject))
        if like_good:
            method += "; %d damaged picture(s) score like good parts and were not used (listed)" % len(like_good)
    fr_seen = int((good > t).sum())
    caught = int((bad > t).sum())
    res = dict(threshold=t, method=method, n_good=int(len(good)), n_bad=int(len(bad)),
               est_false_reject=good_tail(fit, t), false_reject_seen=fr_seen,
               false_reject_seen_ci=wilson(fr_seen, len(good)),
               damaged_caught=caught, damaged_caught_ci=wilson(caught, len(bad)) if len(bad) else None,
               good_max=gmax, good_median=float(np.median(good)), threshold_good_only=t_good,
               bad_min=float(bad.min()) if len(bad) else None, gumbel=fit,
               target_false_reject=target_false_reject, max_false_reject=max_false_reject,
               expected_defect_rate=float(expected_defect_rate),
               look_like_good=like_good, good_set_aside=set_aside,
               damaged_missed=[dict(name=n, score=float(s)) for n, s in zip(bad_names, bad) if s <= t])
    lines = ["Threshold %.3f (%s)." % (t, method),
             "Held-out good parts: %d of %d would be rejected; the fit expects %.2f %% in the long run." % (
                 fr_seen, len(good), 100 * res["est_false_reject"])]
    if set_aside:
        lines.append("CHECK: held-out good picture %s scores %.2f, far above the others (chance %.2f %%): it was "
                     "left out of the threshold. Look at it: is it damaged, empty, or another part?" % (
                         set_aside[0]["name"], set_aside[0]["score"], 100 * set_aside[0]["chance"]))
    if len(bad):
        lo, hi = res["damaged_caught_ci"]
        lines.append("Damaged parts: %d of %d caught (95 %% range %.0f-%.0f %%)." % (caught, len(bad), 100 * lo, 100 * hi))
        if res["bad_min"] is not None and res["bad_min"] > res["good_max"]:
            lines.append("Every damaged part scores above every good part (gap %.3f to %.3f)." % (
                res["good_max"], res["bad_min"]))
        if like_good:
            lines.append("CHECK: these damaged pictures score like good parts, so the line ignores them: %s. Open "
                         "each one's heat map: if the damage is not visible on this face (e.g. it is on the other "
                         "face), move the picture to the other face's damaged folder or delete it." % ", ".join(
                             "%s (%.2f)" % (d["name"], d["score"]) for d in like_good))
    else:
        lines.append("No damaged parts yet: collect 10-20 per shape and face and run calibrate-threshold again.")
    res["text"] = " ".join(lines)
    return res
