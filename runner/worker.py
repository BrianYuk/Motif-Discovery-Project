"""Worker: asks the coordinator for jobs, runs each on one thread, returns results."""
import os

# One thread per job, set before NumPy loads, so floating-point results do not
# depend on how many cores the machine has.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import socket  # noqa: E402
import time  # noqa: E402

from . import protocol as P  # noqa: E402
from .tasks import run_job  # noqa: E402


def run_worker(host: str, port: int, name: str = None, log=print, connect_timeout=30.0):
    name = name or f"{socket.gethostname()}-{os.getpid()}"
    deadline = time.time() + connect_timeout
    while True:
        try:
            sock = socket.create_connection((host, port))
            break
        except OSError:
            if time.time() > deadline:
                raise
            time.sleep(0.2)
    stream = sock.makefile("rwb")
    n_jobs = 0
    try:
        while True:
            P.send(stream, {"type": P.REQUEST, "worker": name})
            msg = P.recv(stream)
            if msg is None or msg["type"] == P.DONE:
                break
            if msg["type"] == P.WAIT:
                time.sleep(msg.get("seconds", 0.5))
                continue
            job = msg["job"]
            result = run_job(job)
            result["worker"] = name
            P.send(stream, {"type": P.RESULT, "worker": name, "job_id": job["job_id"], "result": result})
            P.recv(stream)  # ACK
            n_jobs += 1
            log(f"[{name}] finished {job['job_id']} in {result['runtime_s']:.2f}s")
    finally:
        stream.close()
        sock.close()
    return n_jobs


def worker_process(host: str, port: int, name: str):
    """Entry point for a worker started with multiprocessing."""
    run_worker(host, port, name=name)
