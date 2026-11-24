import numpy as np

METHOD_COLORS = {"em": "#1f77b4", "gibbs": "#d62728", "cnn": "#2ca02c"}
METHOD_STYLE = {"em": ("s", "--"), "gibbs": ("o", "-"), "cnn": ("^", ":")}
METHOD_LABELS = {"em": "EM", "gibbs": "Gibbs", "cnn": "CNN"}


def _logo_matrix(pwm):
    import pandas as pd
    p = np.clip(np.asarray(pwm, dtype=float), 1e-12, 1)
    p = p / p.sum(axis=0, keepdims=True)
    ic = 2 + (p * np.log2(p)).sum(axis=0)          # bits per column, background 0.25
    return pd.DataFrame((p * ic).T, columns=list("ACGT"))   # letter height = p * IC


def plot_logo(pwm, ax, title=""):
    """Draw a sequence logo from a 4 x W matrix (rows A, C, G, T); height = information content."""
    import logomaker
    logomaker.Logo(_logo_matrix(pwm), ax=ax, shade_below=0.0, show_spines=False)
    ax.set_ylim(0, 2)
    width = np.asarray(pwm).shape[1]
    ax.set_xticks(range(width))
    ax.set_xticklabels(range(1, width + 1), fontsize=7)
    ax.set_ylabel("bits")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=10)
    return ax


def plot_similarity_vs_n(df, ax, methods=("em", "gibbs", "cnn")):
    """Similarity against number of sequences, one line per method: mean with min-max band."""
    for method in methods:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        g = sub.groupby("n_seqs")["similarity"].agg(["mean", "min", "max"]).sort_index()
        color = METHOD_COLORS.get(method)
        marker, ls = METHOD_STYLE.get(method, ("o", "-"))
        ax.plot(g.index, g["mean"], marker=marker, ls=ls, color=color, lw=1.8, ms=6,
                label=METHOD_LABELS.get(method, method))
        ax.fill_between(g.index, g["min"], g["max"], color=color, alpha=0.2, linewidth=0)
    ax.set_xscale("log")
    xs = sorted(df["n_seqs"].unique())
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.minorticks_off()
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("number of sequences")
    ax.set_ylabel("similarity to planted motif")
    ax.spines[["top", "right"]].set_visible(False)
    return ax