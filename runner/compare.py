"""Check that two result folders hold the same results (by hash, ignoring run time)."""
import json
from pathlib import Path


def load_hashes(results_dir) -> dict:
    return {p.stem: json.loads(p.read_text())["result_hash"] for p in Path(results_dir).glob("*.json")}


def compare(dir_a, dir_b) -> dict:
    a, b = load_hashes(dir_a), load_hashes(dir_b)
    common = set(a) & set(b)
    mismatched = sorted(j for j in common if a[j] != b[j])
    return {
        "jobs_a": len(a), "jobs_b": len(b), "common": len(common),
        "only_a": sorted(set(a) - set(b)), "only_b": sorted(set(b) - set(a)),
        "mismatched": mismatched,
        "identical": not mismatched and set(a) == set(b),
    }
