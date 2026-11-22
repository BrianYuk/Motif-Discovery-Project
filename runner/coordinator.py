import json
import os
import socketserver
import threading
import time
from collections import deque
from pathlib import Path

from . import protocol as P

DEFAULT_HEARTBEAT_S = 2.0
DEFAULT_MISSED_LIMIT = 3
DEFAULT_MAX_ATTEMPTS = 3


class Coordinator:
    def __init__(self, jobs, results_dir, host="127.0.0.1", port=0, log=print,
                 heartbeat_s=DEFAULT_HEARTBEAT_S, missed_limit=DEFAULT_MISSED_LIMIT,
                 max_attempts=DEFAULT_MAX_ATTEMPTS):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir = self.results_dir / "failed"
        self.jobs = {j["job_id"]: j for j in jobs}
        done = {p.stem for p in self.results_dir.glob("*.json")}
        self.pending = deque(j["job_id"] for j in jobs if j["job_id"] not in done)
        self.assigned = {}        # job_id -> {"worker": name, "last": monotonic time of last heartbeat}
        self.attempts = {}        # job_id -> number of times it was assigned
        self.done = set(done) & set(self.jobs)
        self.failed = set()
        self.duplicates = 0
        self.requeues = 0
        self.heartbeat_s = heartbeat_s
        self.lease_s = heartbeat_s * missed_limit
        self.max_attempts = max_attempts
        self.lock = threading.Lock()
        self.finished = threading.Event()
        self._stopping = threading.Event()
        self.log = log
        self._check_finished()

        coord = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                coord._serve(self.rfile, self.wfile)

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self.server = Server((host, port), Handler)
        self.address = self.server.server_address

    # Message handling
    def _serve(self, rfile, wfile):
        while True:
            try:
                msg = P.recv(rfile)
            except (OSError, ValueError):
                return
            if msg is None:
                return
            kind = msg.get("type")
            if kind == P.REQUEST:
                P.send(wfile, self._next_for(msg.get("worker", "?")))
            elif kind == P.HEARTBEAT:
                self._heartbeat(msg.get("job_id"), msg.get("worker"))
            elif kind == P.RESULT:
                stored = self._store(msg["job_id"], msg["result"])
                P.send(wfile, {"type": P.ACK, "job_id": msg["job_id"], "stored": stored})

    def _next_for(self, worker):
        with self.lock:
            if self.pending:
                job_id = self.pending.popleft()
                self.assigned[job_id] = {"worker": worker, "last": time.monotonic()}
                self.attempts[job_id] = self.attempts.get(job_id, 0) + 1
                return {"type": P.ASSIGN, "job": self.jobs[job_id], "heartbeat_s": self.heartbeat_s}
            if self.assigned:
                return {"type": P.WAIT, "seconds": min(0.5, self.heartbeat_s)}
            return {"type": P.DONE}

    def _heartbeat(self, job_id, worker):
        with self.lock:
            lease = self.assigned.get(job_id)
            if lease is not None and lease["worker"] == worker:
                lease["last"] = time.monotonic()

    def _store(self, job_id, result):
        if job_id not in self.jobs:
            return False
        with self.lock:
            self.assigned.pop(job_id, None)
            if job_id in self.done:
                self.duplicates += 1           # first result wins
                return False
            # A late result can arrive while the job waits in the queue for a retry, or after it was given up; either way it is a valid first result.
            try:
                self.pending.remove(job_id)
            except ValueError:
                pass
            if job_id in self.failed:
                self.failed.discard(job_id)
                (self.failed_dir / f"{job_id}.json").unlink(missing_ok=True)
            path = self.results_dir / f"{job_id}.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(result, indent=1))
            os.replace(tmp, path)              # atomic: never a half-written result file
            self.done.add(job_id)
            self._check_finished()
            n_done, n_all = len(self.done), len(self.jobs)
        self.log(f"[coordinator] {job_id} stored ({n_done}/{n_all})")
        return True

    def _check_finished(self):
        if not self.pending and not self.assigned:
            self.finished.set()

    # Leases
    def check_leases(self, now=None):
        """Requeue or give up on every job whose lease has expired. Returns the
        list of (job_id, action) taken; called by the monitor thread."""
        now = time.monotonic() if now is None else now
        actions = []
        with self.lock:
            expired = [j for j, lease in self.assigned.items() if now - lease["last"] > self.lease_s]
            for job_id in expired:
                worker = self.assigned.pop(job_id)["worker"]
                if self.attempts.get(job_id, 0) >= self.max_attempts:
                    self.failed.add(job_id)
                    self.failed_dir.mkdir(exist_ok=True)
                    record = {"job_id": job_id, "attempts": self.attempts[job_id],
                              "last_worker": worker, "reason": "lease expired on every attempt"}
                    (self.failed_dir / f"{job_id}.json").write_text(json.dumps(record, indent=1))
                    actions.append((job_id, "failed"))
                else:
                    self.pending.appendleft(job_id)
                    self.requeues += 1
                    actions.append((job_id, "requeued"))
            self._check_finished()
        for job_id, action in actions:
            self.log(f"[coordinator] {job_id} {action} (lease expired after {self.lease_s:.1f}s)")
        return actions

    def _monitor(self):
        period = min(1.0, self.heartbeat_s / 2)
        while not self._stopping.wait(period):
            self.check_leases()

    # Lifecycle
    def start(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        threading.Thread(target=self._monitor, daemon=True).start()
        return self

    def wait(self, timeout=None):
        ok = self.finished.wait(timeout)
        time.sleep(0.2)          # let workers receive DONE before the server closes
        return ok

    def stop(self):
        self._stopping.set()
        self.server.shutdown()
        self.server.server_close()
