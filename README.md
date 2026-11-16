# Motif Discovery: EM, Gibbs Sampling and a CNN on Planted Motifs

Group project for SCIE6062001 Computational Biology, COMP6696001 Research Methodology,
Artificial Intelligence and COMP6705001 Distributed Systems.
Brian Yuktipada (2802523004) · Jerico Tjan (2802521112)

This repository holds the benchmark (`motifbench/`) and the distributed job runner
(`runner/`) described in the three proposals. Everything is Python.

## Status: sessions 7 to 8

| Timeline item | Planned | Status |
| --- | --- | --- |
| Generator and metrics | S7 to S8 | Done: planted-motif generator, information content, Tomtom-style similarity, site precision/recall/F1 (raw and offset-aligned) |
| EM and Gibbs in Python | S7 to S9 | Working: ZOOPS EM with start screening and phase shifts; ZOOPS Gibbs site sampler with phase shifts. Speed-up of Gibbs left for S9 |
| CNN motif model | S8 to S10 | First version: one conv layer, max-pool, trained against dinucleotide shuffles; best filter turned into a PWM. Tuning on the pilot grid is S9 to S10 |
| Distributed job runner | S8 to S10 | Protocol, coordinator and workers done: JSON over TCP, FIFO dispatch, results keyed by job ID (duplicates discarded), skip finished jobs on restart, seeds derived from job IDs. Heartbeats, leases, write-ahead log and LPT are S9 to S10 |

Checks run so far (2-core cloud machine):

- 19 automated tests pass (`pytest`), including recovery of a strong motif by EM, Gibbs sampling and the CNN, and determinism of each method given its seed.
- `scripts/check_determinism.py`: 12 jobs (all three methods) on 1 worker and on 2 workers give byte-identical result hashes; 29.2 s on 1 worker against 18.7 s on 2.
- Timing of one job at 2,000 sequences (half with a site, NFE2): EM about 25 to 55 s, Gibbs sampling about 107 s, CNN about 6 s. With 3 restarts per method, the full grid is roughly 10 CPU-hours, so about 1.5 hours on 8 workers. These are single runs, to be replaced by the pilot grid.

## Setup

```bash
python -m pip install -r requirements.txt     # numpy, scipy, torch (CNN only), pytest
python -m pytest -q                           # about 20 s
```

The JASPAR 2024 matrices are already in `data/profiles.json`, so nothing is downloaded at run time.
`scripts/build_profiles.py` rebuilds that file with pyjaspar if needed.

## Live demo

```bash
python scripts/demo.py                         # NFE2 in 150 of 300 sequences
python scripts/demo.py --n 100 --frac 0.5      # small data: watch the CNN struggle
python scripts/demo.py --profile MA0162.5      # a weak motif (EGR1, 11.5 bits)
python scripts/demo.py --no-cnn                # without PyTorch
```

It prints the planted motif and what each method found: consensus, information content
per column, similarity to the planted matrix, site F1 and run time.

## Using the methods

```python
from motifbench import find_motif, generate_dataset, get_pwm, pwm_similarity, site_metrics

planted = get_pwm("MA0841.2")                                   # NFE2
ds = generate_dataset(planted, n_seqs=500, frac_with_site=0.5, seed=1)
res = find_motif(ds.sequences, width=10, method="em", seed=0)   # "em", "gibbs" or "cnn"
pwm_similarity(res["pwm"], planted)      # {"score": ..., "offset": ..., "strand": ...}
site_metrics(res["sites"], ds.sites)     # precision, recall, F1
```

## Running the grid on two laptops

```bash
# Laptop A (coordinator; its own workers can join too)
python -m runner coordinator --grid pilot --results results/pilot --port 5555
python -m runner worker --host 127.0.0.1 --port 5555

# Laptop B (same network; use laptop A's IP address)
python -m runner worker --host 192.168.1.10 --port 5555

# One machine, N worker processes
python -m runner local --grid tiny --workers 2 --results results/tiny

# Are two runs identical? (compares result hashes, ignoring run time)
python -m runner compare results/run_a results/run_b
```

Grids (`runner/grid.py`): `tiny` (8 quick jobs, for tests), `pilot` (18 conditions x 1 replicate
x 3 methods, for timing), `full` (18 conditions x 10 replicates x 3 methods x 3 restarts = 1,620 jobs).
Every seed comes from a SHA-256 hash of the dataset or job ID, so a job gives the same result on any worker.

## Layout

```
motifbench/   alphabet, profiles, generator, shuffle, metrics, em, gibbs, cnn, api (find_motif)
runner/       protocol, grid, tasks (run one job), coordinator, worker, local, compare, CLI
scripts/      demo.py, check_determinism.py, build_profiles.py
data/         profiles.json (JASPAR 2024 count matrices)
tests/        pytest suite
```

## Findings so far, worth discussing in the reports

- **EM converges out of phase.** Without phase shifts, EM on 2,000 sequences found NFE2 three or four columns off (similarity 0.6 to 0.7, raw site F1 0, offset-aligned F1 0.7 to 0.9). Adding a phase-shift search after convergence, as Gibbs samplers do, fixed it (similarity 1.00, F1 0.97 to 0.99).
- **ZOOPS overestimates the share of sequences with a site.** With 20% carriers, EM's estimated share can approach 50%, and weak look-alike sites lower precision. This is method behaviour, not a bug, and is what RQ2 measures.
- **The CNN needs data.** From 300 sequences up it recovers NFE2 reliably (similarity about 0.9); at 100 sequences it often fails, especially when only half carry a site. This is the pattern H1 predicts. Its training settings (patience 15, learning rate 0.01) are provisional until the pilot tuning.

## Next (session 9)

- Runner: heartbeats, leases and requeueing of jobs from dead workers; start the write-ahead log.
- Gibbs: vectorise the per-sequence update to cut the 107 s per job at N = 2,000.
- Run the pilot grid on both laptops; tune the CNN and EM settings on pilot datasets only.
- Both strands for the real ChIP-seq data (synthetic sites are forward-strand only).
