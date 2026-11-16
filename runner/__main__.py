"""Command line for the job runner.

  python -m runner coordinator --grid tiny --results results/a --port 5555
  python -m runner worker --host 192.168.1.10 --port 5555
  python -m runner local --grid tiny --workers 2 --results results/b
  python -m runner compare results/a results/b
"""
import argparse
import json
import sys

from .grid import PRESETS, preset_jobs


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m runner")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("coordinator", help="hand out a grid of jobs and collect results")
    c.add_argument("--grid", choices=sorted(PRESETS), default="tiny")
    c.add_argument("--results", required=True)
    c.add_argument("--host", default="0.0.0.0")
    c.add_argument("--port", type=int, default=5555)

    w = sub.add_parser("worker", help="run jobs for a coordinator")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=5555)
    w.add_argument("--name")

    l = sub.add_parser("local", help="coordinator plus N worker processes on this machine")
    l.add_argument("--grid", choices=sorted(PRESETS), default="tiny")
    l.add_argument("--results", required=True)
    l.add_argument("--workers", type=int, default=2)

    k = sub.add_parser("compare", help="check two result folders are identical")
    k.add_argument("dir_a")
    k.add_argument("dir_b")

    args = ap.parse_args(argv)

    if args.cmd == "coordinator":
        from .coordinator import Coordinator
        coord = Coordinator(preset_jobs(args.grid), args.results, args.host, args.port).start()
        print(f"coordinator listening on {coord.address[0]}:{coord.address[1]}, "
              f"{len(coord.pending)} jobs to run")
        coord.wait()
        coord.stop()
        print(f"all {len(coord.done)} jobs done; duplicates discarded: {coord.duplicates}")
    elif args.cmd == "worker":
        from .worker import run_worker
        n = run_worker(args.host, args.port, name=args.name)
        print(f"worker finished {n} jobs")
    elif args.cmd == "local":
        from .local import run_local
        run_local(args.grid, args.results, args.workers)
    elif args.cmd == "compare":
        from .compare import compare
        report = compare(args.dir_a, args.dir_b)
        print(json.dumps(report, indent=1))
        return 0 if report["identical"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
