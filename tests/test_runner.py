import threading

from runner.compare import compare
from runner.coordinator import Coordinator
from runner.grid import preset_jobs, seed_from_id
from runner.local import run_local
from runner.tasks import run_job
from runner.worker import run_worker


def test_seeds_depend_only_on_names():
    assert seed_from_id("N100_f100_high_r00_em_s0") == seed_from_id("N100_f100_high_r00_em_s0")
    assert seed_from_id("a") != seed_from_id("b")
    jobs = preset_jobs("full")
    assert len(jobs) == 18 * 10 * 3 * 3          # conditions x replicates x methods x restarts
    assert len({j["job_id"] for j in jobs}) == len(jobs)


def test_job_result_is_reproducible():
    job = preset_jobs("tiny")[0]
    assert run_job(job)["result_hash"] == run_job(job)["result_hash"]


def test_one_and_two_workers_give_identical_results(tmp_path):
    quiet = lambda *a, **k: None  # noqa: E731
    run_local("tiny", tmp_path / "one", 1, log=quiet)
    run_local("tiny", tmp_path / "two", 2, log=quiet)
    report = compare(tmp_path / "one", tmp_path / "two")
    assert report["identical"] and report["common"] == len(preset_jobs("tiny"))


def test_duplicate_result_is_discarded_and_finished_jobs_are_skipped(tmp_path):
    jobs = preset_jobs("tiny")[:2]
    quiet = lambda *a, **k: None  # noqa: E731
    coord = Coordinator(jobs, tmp_path, log=quiet).start()
    t = threading.Thread(target=run_worker, args=coord.address, kwargs={"name": "t", "log": quiet})
    t.start()
    assert coord.wait(60)
    t.join()
    first = run_job(jobs[0])
    assert coord._store(jobs[0]["job_id"], first) is False and coord.duplicates == 1
    coord.stop()
    # A new coordinator on the same folder finds nothing left to do.
    again = Coordinator(jobs, tmp_path, log=quiet)
    assert not again.pending and again.finished.is_set()
    again.server.server_close()
