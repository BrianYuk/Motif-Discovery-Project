"""Run one benchmark job and build its result record."""
import hashlib
import json
import time

import numpy as np

from motifbench import find_motif, generate_dataset, get_pwm, information_content, pwm_similarity, site_metrics
from motifbench.metrics import align_sites

ROUND = 6   # results are rounded before hashing so tiny float noise does not count as a difference


def _r(x):
    return round(float(x), ROUND)


def _run_test_job(job: dict) -> dict:
    """Test-only jobs, so failure tests do not wait on real motif finding.
    "sleep": wait settings["seconds"], then return a fixed result.
    "crash": raise, which ends the worker as a crash would."""
    if job["method"] == "crash":
        raise RuntimeError(f"test job {job['job_id']} crashed its worker on purpose")
    t0 = time.perf_counter()
    time.sleep(job["settings"].get("seconds", 1.0))
    deterministic = {"job_id": job["job_id"], "method": "sleep"}
    return {**deterministic, "result_hash": result_hash(deterministic),
            "runtime_s": time.perf_counter() - t0}


def make_test_job(job_id: str, method: str = "sleep", seconds: float = 1.0) -> dict:
    return {"job_id": job_id, "method": method, "settings": {"seconds": seconds}}


def run_job(job: dict) -> dict:
    if job["method"] in ("sleep", "crash"):
        return _run_test_job(job)
    d = job["dataset"]
    planted = get_pwm(d["profile"])
    ds = generate_dataset(planted, d["n_seqs"], seq_len=d["seq_len"],
                          frac_with_site=d["frac_with_site"], seed=d["seed"])
    t0 = time.perf_counter()
    found = find_motif(ds.sequences, ds.width, job["method"], seed=job["seed"], **job["settings"])
    runtime = time.perf_counter() - t0

    sim = pwm_similarity(found["pwm"], planted)
    sites = site_metrics(found["sites"], ds.sites)
    aligned = site_metrics(align_sites(found["sites"], sim["offset"], sim["strand"]), ds.sites)
    deterministic = {
        "job_id": job["job_id"],
        "method": job["method"],
        "dataset_id": d["id"],
        "profile": d["profile"],
        "pwm": [[_r(x) for x in row] for row in np.asarray(found["pwm"])],
        "sites": found["sites"],
        "score": _r(found["score"]),
        "similarity": _r(sim["score"]),
        "offset": sim["offset"],
        "strand": sim["strand"],
        "precision": _r(sites["precision"]),
        "recall": _r(sites["recall"]),
        "f1": _r(sites["f1"]),
        "f1_aligned": _r(aligned["f1"]),
        "planted_ic": _r(information_content(planted)),
        "found_ic": _r(information_content(found["pwm"])),
    }
    return {
        **deterministic,
        "result_hash": result_hash(deterministic),
        "runtime_s": runtime,          # not hashed: it depends on the machine and its load
    }


def result_hash(record: dict) -> str:
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
