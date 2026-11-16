"""Message protocol: one JSON object per line over a TCP connection.

Messages (field "type"):
  worker -> coordinator: REQUEST {worker}, RESULT {worker, job_id, result}
  coordinator -> worker: ASSIGN {job}, WAIT {seconds}, DONE {}, ACK {job_id, stored}
HEARTBEAT and leases are added in session 9.
"""
import json

REQUEST, RESULT, ASSIGN, WAIT, DONE, ACK = "REQUEST", "RESULT", "ASSIGN", "WAIT", "DONE", "ACK"


def send(stream, msg: dict) -> None:
    stream.write((json.dumps(msg, separators=(",", ":")) + "\n").encode())
    stream.flush()


def recv(stream):
    line = stream.readline()
    if not line:
        return None          # connection closed
    return json.loads(line)
