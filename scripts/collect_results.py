import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

DATASET_RE = re.compile(r"^N(?P<n_seqs>\d+)_f(?P<frac>\d+)_(?P<strength>[a-z]+)_r(?P<replicate>\d+)$")
KEEP = ["job_id", "method", "dataset_id", "profile", "similarity", "f1", "f1_aligned",
        "precision", "recall", "planted_ic", "found_ic", "runtime_s", "pwm"]


def parse_dataset_id(dataset_id: str) -> dict:
    # N500_f050_high_r00 -> 500 sequences, 50% with a site, high-information group, replicate 0
    m = DATASET_RE.match(dataset_id)
    if not m:
        raise ValueError(f"cannot parse dataset_id {dataset_id!r}")
    return {"n_seqs": int(m["n_seqs"]), "frac_with_site": int(m["frac"]) / 100,
            "strength": m["strength"], "replicate": int(m["replicate"])}


def collect(results_dir) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(results_dir).glob("*.json")):   # failed/ is a subfolder, so skipped
        rec = json.loads(path.read_text())
        if "dataset_id" not in rec:
            continue
        row = {k: rec.get(k) for k in KEEP}
        row["pwm"] = json.dumps(rec["pwm"])                  # keep the matrix, one cell per job
        row.update(parse_dataset_id(rec["dataset_id"]))
        rows.append(row)
    if not rows:
        raise SystemExit(f"no result files found in {results_dir}")
    df = pd.DataFrame(rows)
    front = ["job_id", "method", "dataset_id", "n_seqs", "frac_with_site", "strength", "replicate"]
    return df[front + [c for c in df.columns if c not in front]].sort_values(
        ["strength", "frac_with_site", "n_seqs", "method"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("out_csv")
    args = ap.parse_args()
    df = collect(args.results_dir)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {len(df)} rows to {args.out_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()