"""Charts for the learning curve and score spreads (matplotlib, saved as PNG).

Changelog
  0.1.0  2026-11-12  first version.
  0.2.0  2026-11-12  learning curve in two panels: the good-only threshold (decides "enough", audit M7) and
                     the damaged-part measures; the sentence under it no longer shows a variable name (L7).

Every chart carries, in its own picture: what is measured, where the data comes from, and one plain
sentence underneath saying what it shows.
"""
import numpy as np

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8})
    return plt


def _footer(fig, source, sentence):
    import textwrap
    w = int(fig.get_figwidth() * 13)
    src = "\n".join(textwrap.wrap("Data: " + source, int(w * 1.25)))
    fig.text(0.01, 0.012, src, ha="left", va="bottom", fontsize=8, color=MUTED, linespacing=1.3)
    fig.text(0.01, 0.012 + 0.032 * (src.count("\n") + 1) + 0.01, "\n".join(textwrap.wrap(sentence, w)),
             ha="left", va="bottom", fontsize=10, color=INK, linespacing=1.3)


def learning_curve(rows, info, path):
    """Two panels: left the good-only threshold (falls, then settles: that decides "enough", no damaged
    parts needed), right how well damaged parts stand out (painted stand-ins until real ones exist)."""
    plt = _plt()
    ns = sorted({r["n"] for r in rows})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5.4))
    fig.subplots_adjust(bottom=0.33, top=0.84, left=0.08, right=0.98, wspace=0.28)

    def band(ax, key, colour, label):
        med = [np.median([r[key] for r in rows if r["n"] == n]) for n in ns]
        lo = [np.min([r[key] for r in rows if r["n"] == n]) for n in ns]
        hi = [np.max([r[key] for r in rows if r["n"] == n]) for n in ns]
        ax.fill_between(ns, lo, hi, color=colour, alpha=0.15, linewidth=0)
        ax.plot(ns, med, color=colour, linewidth=2, marker="o", markersize=6, label=label)
        ax.annotate("%.2f" % med[-1], (ns[-1], med[-1]), textcoords="offset points", xytext=(6, 0),
                    va="center", fontsize=9, color=INK)
    if all("thr_good" in r for r in rows):
        band(a1, "thr_good", BLUE, "good-only threshold")
    a1.set_ylabel("threshold (anomaly score)")
    a1.set_title("left: the line settles when enough good parts are seen", fontsize=9, color=MUTED, loc="left")
    for key, colour, label in (("caught_at_threshold", ORANGE, "damaged parts caught at that threshold"),
                               ("auroc", AQUA, "AUROC (chance a damaged part outscores a good one)")):
        band(a2, key, colour, label)
    a2.set_ylim(0, 1.05)
    a2.set_ylabel("share (0 to 1)")
    a2.set_title("right: damaged parts (%s)" % ("painted stand-ins" if "stand-in" in info.get("damaged", "")
                                                 else "real pictures"), fontsize=9, color=MUTED, loc="left")
    a2.legend(loc="lower right", frameon=False, fontsize=8)
    sug = info.get("suggestion") or {}
    for ax in (a1, a2):
        if sug.get("n"):
            ax.axvline(sug["n"], color=MUTED, linestyle="--", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("good pictures used for training (log scale)")
    if sug.get("n"):
        a1.text(sug["n"], a1.get_ylim()[1], " enough: %d" % sug["n"], color=INK, fontsize=9, va="top")
    fig.suptitle("Learning curve, %s / %s: lines = median of %d random draws, bands = lowest to highest draw" % (
        info["shape"], info["face"], max(1, len(rows) // max(1, len(ns)))), fontsize=11, color=INK, x=0.01, ha="left")
    _footer(fig, "%s; %d held-out good pictures; damaged: %s; features: %s." % (
        info.get("source", "pictures in data/good"), info["n_holdout"], info["damaged"], info["backbone"]),
        sug.get("text", ""))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def score_strip(groups, threshold, title, source, sentence, path):
    """groups: [(label, scores, colour)] drawn as dots in rows, with the threshold as a line."""
    plt = _plt()
    h = 0.55 * len(groups) + 2.6
    fig, ax = plt.subplots(figsize=(8, h))
    fig.subplots_adjust(bottom=1.55 / h, top=1 - 0.55 / h, left=0.2, right=0.97)
    rng = np.random.default_rng(0)
    for i, (label, sc, colour) in enumerate(groups):
        sc = np.asarray(sc, float)
        ax.scatter(sc, i + rng.uniform(-0.18, 0.18, len(sc)), s=22, color=colour, edgecolor="white", linewidth=0.6)
    if threshold is not None:
        ax.axvline(threshold, color=INK, linewidth=1.5)
        ax.text(threshold, len(groups) - 0.4, " threshold", fontsize=9, color=INK, va="bottom")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g[0] for g in groups])
    ax.set_ylim(-0.6, len(groups) - 0.1)
    ax.invert_yaxis()
    ax.set_xlabel("anomaly score (higher = less like the good parts)")
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    _footer(fig, source, sentence)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
