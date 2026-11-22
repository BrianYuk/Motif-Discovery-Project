import hashlib
import itertools

from motifbench.profiles import STRENGTH_GROUPS


def seed_from_id(name: str) -> int:
    # Stable 32-bit seed from a string (SHA-256, so it is the same on every machine).
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")


# Settings per method for one job (one random start each; restarts are separate jobs).
METHOD_SETTINGS = {
    "em": {"n_starts": 1, "max_iter": 500, "n_candidates": 20},
    "gibbs": {"n_starts": 1, "n_sweeps": 200, "burn_in": 50},
    "cnn": {"n_filters": 16, "epochs": 100, "patience": 15},
}

PRESETS = {
    # Full study grid from the proposal: 3 x 3 x 2 = 18 conditions.
    "full": dict(n_values=(100, 500, 2000), fracs=(1.0, 0.5, 0.2), groups=("high", "low"),
                 replicates=10, methods=("em", "gibbs", "cnn"), restarts=3),
    # Pilot: one replicate per condition, for timing every method and size.
    "pilot": dict(n_values=(100, 500, 2000), fracs=(1.0, 0.5, 0.2), groups=("high", "low"),
                  replicates=1, methods=("em", "gibbs", "cnn"), restarts=1),
    # Tiny: seconds to run; used by the tests and the runner demo.
    "tiny": dict(n_values=(40,), fracs=(1.0, 0.5), groups=("high",), replicates=2,
                 methods=("em", "gibbs"), restarts=1,
                 overrides={"gibbs": {"n_sweeps": 20, "burn_in": 5}, "em": {"max_iter": 100}}),
}


def make_jobs(n_values, fracs, groups, replicates, methods, restarts,
              seq_len=200, overrides=None):
    overrides = overrides or {}
    jobs = []
    for n, frac, group in itertools.product(n_values, fracs, groups):
        cond = f"N{n}_f{int(frac * 100):03d}_{group}"
        for r in range(replicates):
            profile = STRENGTH_GROUPS[group][r % len(STRENGTH_GROUPS[group])]
            dataset_id = f"{cond}_r{r:02d}"
            for method, k in itertools.product(methods, range(restarts)):
                job_id = f"{dataset_id}_{method}_s{k}"
                settings = {**METHOD_SETTINGS[method], **overrides.get(method, {})}
                jobs.append({
                    "job_id": job_id,
                    "dataset": {"id": dataset_id, "profile": profile, "n_seqs": n,
                                "frac_with_site": frac, "seq_len": seq_len,
                                "seed": seed_from_id(dataset_id)},
                    "method": method,
                    "restart": k,
                    "seed": seed_from_id(job_id),
                    "settings": settings,
                })
    return jobs


def preset_jobs(name: str):
    return make_jobs(**PRESETS[name])
