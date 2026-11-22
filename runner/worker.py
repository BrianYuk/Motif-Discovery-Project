import os

# One thread per job, set before NumPy loads, so floating-point results do not
# depend on how many cores the machine has.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import socket  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

from . import protocol as P  # noqa: E402
from .tasks import run_job  # noqa: E402


class _Heartbeat:
    def __init__(self, send, worker, job_id, interval):
        self._send, self._msg, self._interval = send, {"type": P.HEARTBEAT, "worker": worker, "job_id": job_id}, interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self._interval):
            try:
                self._send(self._msg)
            except OSError:
                return                      # connection gone; the main thread will notice

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def run_worker(host: str, port: int, name: str = None, log=print, connect_timeout=30.0,
               heartbeats: bool = True):
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
    write_lock = threading.Lock()

    def send(msg):
        with write_lock:
            P.send(stream, msg)

    n_jobs = 0
    try:
        while True:
            send({"type": P.REQUEST, "worker": name})
            msg = P.recv(stream)
            if msg is None or msg["type"] == P.DONE:
                break
            if msg["type"] == P.WAIT:
                time.sleep(msg.get("seconds", 0.5))
                continue
            job = msg["job"]
            if heartbeats:
                with _Heartbeat(send, name, job["job_id"], msg.get("heartbeat_s", 2.0)):
                    result = run_job(job)
            else:
                result = run_job(job)
            result["worker"] = name
            send({"type": P.RESULT, "worker": name, "job_id": job["job_id"], "result": result})
            P.recv(stream)  # ACK
            n_jobs += 1
            log(f"[{name}] finished {job['job_id']} in {result['runtime_s']:.2f}s")
    finally:
        stream.close()
        sock.close()
    return n_jobs


def worker_process(host: str, port: int, name: str):
    run_worker(host, port, name=name)
