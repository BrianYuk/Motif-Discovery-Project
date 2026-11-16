"""Export the chosen JASPAR 2024 count matrices to data/profiles.json.

Run once (needs `pip install pyjaspar`); the result is committed so that
nothing else in the project needs pyjaspar or network access.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from motifbench.profiles import HIGH_IC, LOW_IC  # noqa: E402

EXTRA = ["MA0139.2"]  # CTCF, used in the demo


def main():
    from pyjaspar import jaspardb

    db = jaspardb(release="JASPAR2024")
    out = {"source": "JASPAR2024 via pyjaspar", "profiles": {}}
    for mid in LOW_IC + HIGH_IC + EXTRA:
        m = db.fetch_motif_by_id(mid)
        out["profiles"][mid] = {
            "name": m.name,
            "counts": [[float(x) for x in m.counts[b]] for b in "ACGT"],
        }
        print(mid, m.name, m.length)
    path = ROOT / "data" / "profiles.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
