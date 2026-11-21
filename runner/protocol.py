"""Message protocol: one JSON object per line over a TCP connection.

Messages (field "type"):
  worker -> coordinator:
    REQUEST   {worker}                        ask for a job
    HEARTBEAT {worker, job_id}                "still working on job_id"; no reply
    RESULT    {worker, job_id, result}        a finished job
  coordinator -> worker:
    ASSIGN    {job, heartbeat_s}              run this job; send a heartbeat every heartbeat_s
    WAIT      {seconds}                       nothing to hand out right now; ask again later
    DONE      {}                              the grid is finished; disconnect
    ACK       {job_id, stored}                result received (stored=False: a duplicate)
"""
import json

REQUEST, HEARTBEAT, RESULT = "REQUEST", "HEARTBEAT", "RESULT"
ASSIGN, WAIT, DONE, ACK = "ASSIGN", "WAIT", "DONE", "ACK"


def send(stream, msg: dict) -> None:
    stream.write((json.dumps(msg, separators=(",", ":")) + "\n").encode())
    stream.flush()


def recv(stream):
    line = stream.readline()
    if not line:
        return None          # connection closed
    return json.loads(line)
