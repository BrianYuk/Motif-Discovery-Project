"""Session 9: heartbeats, leases and retries; and the faster Gibbs sampler."""
import sys
import threading
import time
from pathlib import Path

import numpy as np

from runner.compare import compare
from runner.coordinator import Coordinator
from runner.grid import preset_jobs
from runner.local import run_local
from runner.tasks import make_test_job
from runner.worker import run_worker

QUIET = lambda *a, **k: None  # noqa: E731
FAST = dict(heartbeat_s=0.2, missed_limit=3)     # lease expires after 0.6 s of silence


def _thread_worker(coord, name, heartbeats=True, swallow=False):
    def target():
        try:
            run_worker(*coord.address, name=name, log=QUIET, heartbeats=heartbeats)
        except Exception:
            if not swallow:
                raise
    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t


# -- leases, unit level --------------------------------------------------------
def test_lease_expires_heartbeat_renews_and_wrong_worker_is_ignored(tmp_path):
    jobs = [make_test_job("a"), make_test_job("b")]
    coord = Coordinator(jobs, tmp_path, log=QUIET, heartbeat_s=1.0, missed_limit=3)
    msg = coord._next_for("w1")
    assert msg["type"] == "ASSIGN" and msg["heartbeat_s"] == 1.0
    start = coord.assigned["a"]["last"]
    assert coord.check_leases(now=start + 2.9) == []              # still within 3 s
    coord._heartbeat("a", "w2")                                   # not its worker: ignored
    assert coord.assigned["a"]["last"] == start
    coord._heartbeat("a", "w1")                                   # renews the lease
    renewed = coord.assigned["a"]["last"]
    assert renewed > start
    assert coord.check_leases(now=renewed + 3.1) == [("a", "requeued")]
    assert list(coord.pending)[0] == "a"                          # back at the front
    coord.server.server_close()


# -- a worker process killed in the middle of a job ------------------------------
def test_killed_worker_job_is_requeued_and_results_match_one_worker(tmp_path):
    # A slow job first, then the tiny grid. The worker that takes the slow job is
    # killed while it runs; the other worker must finish everything.
    jobs = [make_test_job("slow", seconds=3.0)] + preset_jobs("tiny")
    reference = run_local(None, tmp_path / "ref", 1, log=QUIET, jobs=jobs, **FAST)
    assert reference["requeues"] == 0

    killed = []

    def kill_slow_worker(coord, procs):
        lease = coord.assigned.get("slow")
        if lease and not killed and procs[lease["worker"]].is_alive():
            procs[lease["worker"]].kill()
            killed.append(lease["worker"])

    run = run_local(None, tmp_path / "kill", 2, log=QUIET, jobs=jobs, on_tick=kill_slow_worker, **FAST)
    assert killed, "the slow job was never seen running"
    assert run["requeues"] >= 1 and run["failed"] == []
    report = compare(tmp_path / "ref", tmp_path / "kill")
    assert report["identical"], report


# -- a hung worker: alive but silent ------------------------------------------------
def test_silent_worker_loses_its_job_and_the_later_result_is_discarded(tmp_path):
    coord = Coordinator([make_test_job("s1", seconds=1.5)], tmp_path, log=QUIET, **FAST).start()
    silent = _thread_worker(coord, "silent", heartbeats=False)
    deadline = time.time() + 5
    while "s1" not in coord.assigned and time.time() < deadline:
        time.sleep(0.02)
    healthy = _thread_worker(coord, "healthy")
    assert coord.wait(10)
    silent.join(10)
    healthy.join(10)
    assert coord.requeues == 1                   # taken away from the silent worker
    assert coord.duplicates == 1                 # one of the two results was discarded
    assert (tmp_path / "s1.json").exists() and coord.done == {"s1"}
    coord.stop()


# -- a job that crashes every worker that takes it ---------------------------------
def test_job_that_fails_every_attempt_is_recorded_and_the_grid_finishes(tmp_path):
    jobs = [make_test_job("boom", method="crash"),
            make_test_job("ok1", seconds=0.3), make_test_job("ok2", seconds=0.3)]
    coord = Coordinator(jobs, tmp_path, log=QUIET, max_attempts=2, **FAST).start()
    threads = [_thread_worker(coord, f"w{i}", swallow=True) for i in range(3)]
    assert coord.wait(15)
    for t in threads:
        t.join(5)
    assert coord.failed == {"boom"} and coord.attempts["boom"] == 2
    assert (tmp_path / "failed" / "boom.json").exists()
    assert coord.done == {"ok1", "ok2"}
    coord.stop()


# -- the faster Gibbs sampler gives the same answers as the old one ----------------
def test_fast_gibbs_matches_reference_implementation():
    sys.path.insert(0, str(Path(__file__).parent))
    from reference_gibbs import find_motif_gibbs as reference
    from motifbench import generate_dataset, get_pwm
    from motifbench.alphabet import encode_all
    from motifbench.gibbs import find_motif_gibbs

    for profile, n, frac, seed in [("MA0841.2", 60, 1.0, 1), ("MA0162.5", 120, 0.5, 2), ("MA0841.2", 200, 0.2, 3)]:
        ds = generate_dataset(get_pwm(profile), n, frac_with_site=frac, seed=seed)
        X = encode_all(ds.sequences)
        new = find_motif_gibbs(X, 10, seed=seed, n_sweeps=30, burn_in=5)
        old = reference(X, 10, seed=seed, n_sweeps=30, burn_in=5)
        assert new["sites"] == old["sites"]
        assert np.array_equal(new["pwm"], old["pwm"])
