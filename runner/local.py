"""Run a coordinator and N worker processes on one machine."""
import multiprocessing as mp
import time

from .coordinator import Coordinator
from .grid import preset_jobs
from .worker import worker_process


def run_local(grid, results, n_workers, log=print, jobs=None):
    coord = Coordinator(jobs if jobs is not None else preset_jobs(grid), results,
                        "127.0.0.1", 0, log=log).start()
    host, port = coord.address
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker_process, args=(host, port, f"w{i + 1}"))
             for i in range(n_workers)]
    for p in procs:
        p.start()
    # Wait for the grid to finish, but stop if every worker has died, since there
    # are no leases yet (session 9) to hand their jobs to anyone else.
    while not coord.finished.wait(0.5):
        if not any(p.is_alive() for p in procs):
            coord.stop()
            codes = [p.exitcode for p in procs]
            raise RuntimeError(f"all workers exited before the grid finished (exit codes {codes}); "
                               f"{len(coord.done)}/{len(coord.jobs)} jobs done")
    coord.wait(0)
    for p in procs:
        p.join()
    coord.stop()
    elapsed = time.perf_counter() - t0
    log(f"{len(coord.done)} jobs done with {n_workers} worker(s) in {elapsed:.1f}s; "
        f"duplicates discarded: {coord.duplicates}")
    return {"elapsed_s": elapsed, "done": len(coord.done), "duplicates": coord.duplicates}
