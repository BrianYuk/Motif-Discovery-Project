"""Make the report figures from a results CSV."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from motifbench import get_pwm  # noqa: E402
from motifbench.plots import METHOD_LABELS, plot_logo, plot_similarity_vs_n  # noqa: E402


def similarity_figure(df, out):
    fracs = sorted(df["frac_with_site"].unique(), reverse=True)
    fig, axes = plt.subplots(1, len(fracs), figsize=(4.2 * len(fracs), 3.8), sharey=True)
    for ax, frac in zip(np.atleast_1d(axes), fracs):
        plot_similarity_vs_n(df[df["frac_with_site"] == frac], ax)
        ax.set_title(f"{int(round(frac * 100))}% of sequences with a site")
    np.atleast_1d(axes)[0].legend(loc="lower right", frameon=False)
    for ax in np.atleast_1d(axes)[1:]:
        ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def logo_figure(df, condition, out):
    sub = df[df["dataset_id"] == condition]
    if sub.empty:
        raise SystemExit(f"condition {condition!r} not in the CSV")
    planted = get_pwm(sub["profile"].iloc[0])
    panels = [("Planted", planted)]
    for method in ("em", "gibbs", "cnn"):
        row = sub[sub["method"] == method]
        if row.empty:
            continue
        r = row.iloc[0]
        panels.append((f"{METHOD_LABELS[method]} (similarity {r['similarity']:.2f})",
                       np.array(json.loads(r["pwm"]))))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 2.6))
    for ax, (title, pwm) in zip(np.atleast_1d(axes), panels):
        plot_logo(pwm, ax, title)
    fig.suptitle(condition, fontsize=9, y=0.02, va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--condition", default="N500_f050_high_r00",
                    help="dataset_id for the logo panel")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    similarity_figure(df, out / "similarity_vs_n.png")
    logo_figure(df, args.condition, out / f"logos_{args.condition}.png")
    print(f"saved figures to {out}/")


if __name__ == "__main__":
    main()