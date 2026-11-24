"""Live demo: plant a known motif, then let EM, Gibbs sampling and the CNN find it.

    python scripts/demo.py                    # NFE2, 300 sequences, half with a site
    python scripts/demo.py --profile MA0162.5 --n 300 --frac 0.2
    python scripts/demo.py --no-cnn           # skip the CNN (no PyTorch needed)
    python scripts/demo.py --logos            # also show planted and found logos side by side
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from motifbench import (consensus, find_motif, generate_dataset, information_content,  # noqa: E402
                        load_profiles, pwm_similarity, site_metrics)

BARS = " ▁▂▃▄▅▆▇█"


def ic_bars(pwm):
    # One bar per column, height = information content of that column (0-2 bits).
    p = np.clip(pwm, 1e-12, 1)
    col_ic = (p * np.log2(p / 0.25)).sum(axis=0)
    return "".join(BARS[min(8, int(round(v / 2 * 8)))] for v in col_ic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="MA0841.2", help="JASPAR ID from data/profiles.json")
    ap.add_argument("--n", type=int, default=300, help="number of sequences")
    ap.add_argument("--frac", type=float, default=0.5, help="share of sequences with a site")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-cnn", action="store_true")
    ap.add_argument("--logos", action="store_true",
                    help="show the planted and found sequence logos side by side")
    ap.add_argument("--save-logos", metavar="FILE", help="save the logo figure instead of showing it")
    args = ap.parse_args()

    profiles = load_profiles()
    planted = profiles[args.profile]["pwm"]
    width = planted.shape[1]
    ds = generate_dataset(planted, args.n, frac_with_site=args.frac, seed=args.seed)
    n_sites = sum(s is not None for s in ds.sites)
    print(f"Planted {profiles[args.profile]['name']} ({args.profile}), width {width}, "
          f"{information_content(planted):.1f} bits, in {n_sites} of {args.n} sequences of 200 bp\n")
    print(f"{'':10}{'consensus':<{width + 4}}{'IC per column':<{width + 4}}"
          f"{'similarity':>11}{'site F1':>9}{'time':>8}")
    print(f"{'planted':10}{consensus(planted):<{width + 4}}{ic_bars(planted):<{width + 4}}")

    methods = [("EM", "em", {"n_starts": 1, "n_candidates": 20}),
               ("Gibbs", "gibbs", {"n_starts": 3}),
               ("CNN", "cnn", {})]
    if args.no_cnn:
        methods = methods[:2]
    panels = [("Planted", planted)]
    for label, method, kwargs in methods:
        t0 = time.perf_counter()
        res = find_motif(ds.sequences, width, method, seed=args.seed, **kwargs)
        dt = time.perf_counter() - t0
        sim = pwm_similarity(res["pwm"], planted)
        f1 = site_metrics(res["sites"], ds.sites)["f1"]
        panels.append((f"{label} (similarity {sim['score']:.2f})", res["pwm"]))
        print(f"{label:10}{consensus(res['pwm']):<{width + 4}}{ic_bars(res['pwm']):<{width + 4}}"
              f"{sim['score']:>11.3f}{f1:>9.2f}{dt:>7.1f}s")
    print("\nSimilarity: Tomtom-style column correlation with the planted matrix (1 = identical).")
    print("Lower-case consensus letters mark weak columns (top base below 50%).")
    if args.logos or args.save_logos:
        show_logos(panels, args.save_logos)


def show_logos(panels, save=None):
    import matplotlib
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from motifbench.plots import plot_logo
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 2.6))
    for ax, (title, pwm) in zip(axes, panels):
        plot_logo(np.asarray(pwm), ax, title)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=200)
        print(f"saved {save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()