"""Coordinator: hands out jobs to workers over TCP and stores their results.

Session 8 scope: FIFO dispatch, results keyed by job ID (a duplicate is
discarded), and a simple resume: jobs whose result file already exists are
skipped at start-up. Heartbeats and leases (session 9), the write-ahead log
(sessions 9-10) and LPT scheduling (session 10) build on this.
"""
import json
import os
import socketserver
import threading
import time
from collections import deque
from pathlib import Path

from . import protocol as P


class Coordinator:
    def __init__(self, jobs, results_dir, host="127.0.0.1", port=0, log=print):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.jobs = {j["job_id"]: j for j in jobs}
        done = {p.stem for p in self.results_dir.glob("*.json")}
        self.pending = deque(j["job_id"] for j in jobs if j["job_id"] not in done)
        self.assigned = {}                       # job_id -> worker name
        self.done = set(done) & set(self.jobs)
        self.duplicates = 0
        self.lock = threading.Lock()
        self.finished = threading.Event()
        self.log = log
        if not self.pending:
            self.finished.set()

        coord = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                coord._serve(self.rfile, self.wfile)

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self.server = Server((host, port), Handler)
        self.address = self.server.server_address

    # -- message handling ---------------------------------------------------
    def _serve(self, rfile, wfile):
        while True:
            msg = P.recv(rfile)
            if msg is None:
                return
            if msg["type"] == P.REQUEST:
                P.send(wfile, self._next_for(msg.get("worker", "?")))
            elif msg["type"] == P.RESULT:
                stored = self._store(msg["job_id"], msg["result"])
                P.send(wfile, {"type": P.ACK, "job_id": msg["job_id"], "stored": stored})

    def _next_for(self, worker):
        with self.lock:
            if self.pending:
                job_id = self.pending.popleft()
                self.assigned[job_id] = worker
                return {"type": P.ASSIGN, "job": self.jobs[job_id]}
            if self.assigned:
                return {"type": P.WAIT, "seconds": 0.5}
            return {"type": P.DONE}

    def _store(self, job_id, result):
        with self.lock:
            self.assigned.pop(job_id, None)
            if job_id in self.done:
                self.duplicates += 1           # first result wins
                return False
            path = self.results_dir / f"{job_id}.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(result, indent=1))
            os.replace(tmp, path)              # atomic: never a half-written result file
            self.done.add(job_id)
            if not self.pending and not self.assigned:
                self.finished.set()
            n_done, n_all = len(self.done), len(self.jobs)
        self.log(f"[coordinator] {job_id} stored ({n_done}/{n_all})")
        return True

    # -- lifecycle ----------------------------------------------------------
    def start(self):
        t = threading.Thread(target=self.server.serve_forever, daemon=True)
        t.start()
        return self

    def wait(self, timeout=None):
        ok = self.finished.wait(timeout)
        time.sleep(0.2)          # let workers receive DONE before the server closes
        return ok

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
