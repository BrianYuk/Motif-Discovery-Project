import multiprocessing as mp
import time

from .coordinator import DEFAULT_HEARTBEAT_S, DEFAULT_MAX_ATTEMPTS, DEFAULT_MISSED_LIMIT, Coordinator
from .grid import preset_jobs
from .worker import worker_process


def start_workers(host, port, n_workers, prefix="w"):
    ctx = mp.get_context("spawn")
    procs = {f"{prefix}{i + 1}": ctx.Process(target=worker_process, args=(host, port, f"{prefix}{i + 1}"))
             for i in range(n_workers)}
    for p in procs.values():
        p.start()
    return procs


def run_local(grid, results, n_workers, log=print, jobs=None, heartbeat_s=DEFAULT_HEARTBEAT_S,
              missed_limit=DEFAULT_MISSED_LIMIT, max_attempts=DEFAULT_MAX_ATTEMPTS, on_tick=None):
    """Run a grid with n_workers local processes. `on_tick(coord, procs)` is called
    about twice a second while the grid runs (the failure tests use it to kill a worker)."""
    coord = Coordinator(jobs if jobs is not None else preset_jobs(grid), results, "127.0.0.1", 0,
                        log=log, heartbeat_s=heartbeat_s, missed_limit=missed_limit,
                        max_attempts=max_attempts).start()
    host, port = coord.address
    t0 = time.perf_counter()
    procs = start_workers(host, port, n_workers)
    # Dead workers' jobs are requeued by their leases; but if every worker has died,
    # nobody is left to run them, so stop with an error.
    while not coord.finished.wait(0.5):
        if on_tick is not None:
            on_tick(coord, procs)
        if not any(p.is_alive() for p in procs.values()):
            coord.stop()
            codes = {n: p.exitcode for n, p in procs.items()}
            raise RuntimeError(f"all workers exited before the grid finished (exit codes {codes}); "
                               f"{len(coord.done)}/{len(coord.jobs)} jobs done")
    coord.wait(0)
    for p in procs.values():
        p.join()
    coord.stop()
    elapsed = time.perf_counter() - t0
    log(f"{len(coord.done)} jobs done with {n_workers} worker(s) in {elapsed:.1f}s; "
        f"requeued: {coord.requeues}; failed: {len(coord.failed)}; duplicates discarded: {coord.duplicates}")
    return {"elapsed_s": elapsed, "done": len(coord.done), "duplicates": coord.duplicates,
            "requeues": coord.requeues, "failed": sorted(coord.failed)}
