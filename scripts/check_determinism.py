"""Run a small grid (all three methods) on 1 and on 2 local workers and check that
every result is identical. Usage: python scripts/check_determinism.py [out_dir]"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runner.compare import compare  # noqa: E402
from runner.grid import make_jobs  # noqa: E402
from runner.local import run_local  # noqa: E402


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
    for sub in ("one", "two"):
        shutil.rmtree(out / sub, ignore_errors=True)
    jobs = make_jobs(n_values=(100,), fracs=(1.0, 0.5), groups=("high", "low"),
                     replicates=1, methods=("em", "gibbs", "cnn"), restarts=1)
    quiet = lambda *a, **k: None  # noqa: E731
    one = run_local(None, out / "one", 1, log=quiet, jobs=jobs)
    two = run_local(None, out / "two", 2, log=quiet, jobs=jobs)
    report = compare(out / "one", out / "two")
    print(f"{len(jobs)} jobs: 1 worker {one['elapsed_s']:.1f}s, 2 workers {two['elapsed_s']:.1f}s")
    print(f"identical results: {report['identical']}  mismatched: {report['mismatched']}")
    for p in sorted((out / "two").glob("*.json")):
        d = json.loads(p.read_text())
        print(f"  {d['job_id']:<28} similarity {d['similarity']:.2f}  F1 {d['f1']:.2f}  "
              f"aligned F1 {d['f1_aligned']:.2f}  ({d['worker']})")
    return 0 if report["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
