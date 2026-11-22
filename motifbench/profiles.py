import json
from pathlib import Path

import numpy as np

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "profiles.json"

# Width-10 CORE profiles, chosen so that the two strength groups differ in
# information content but not in width.
LOW_IC = ["MA1991.2", "MA0162.5", "MA0062.4"]   # HNF1A, EGR1, GABPA (~11.3-11.7 bits)
HIGH_IC = ["MA0466.4", "MA0841.2", "MA0605.3"]  # CEBPB, NFE2, ATF3 (~16.1-17.1 bits)
STRENGTH_GROUPS = {"low": LOW_IC, "high": HIGH_IC}

PSEUDOCOUNT_TOTAL = 0.8  # spread equally over the four bases


def counts_to_pwm(counts: np.ndarray, pseudocount_total: float = PSEUDOCOUNT_TOTAL) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    return (counts + pseudocount_total / 4) / (counts.sum(axis=0) + pseudocount_total)


def load_profiles(path: Path = DATA_FILE) -> dict:
    raw = json.loads(Path(path).read_text())
    return {
        mid: {"name": entry["name"], "pwm": counts_to_pwm(np.array(entry["counts"]))}
        for mid, entry in raw["profiles"].items()
    }


def get_pwm(matrix_id: str) -> np.ndarray:
    return load_profiles()[matrix_id]["pwm"]
