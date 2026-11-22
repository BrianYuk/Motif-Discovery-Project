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
