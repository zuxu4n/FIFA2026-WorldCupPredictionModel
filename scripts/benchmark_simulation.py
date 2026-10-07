"""Time the Monte Carlo simulator on the pre-tournament scenario.

    wcpred train --asof 2026-06-11        # once
    python scripts/benchmark_simulation.py [--runs 1000 10000] [--repeats 3]

Only `simulate_world_cup` is timed (data loading and feature building are done
once beforehand). Reports the best of `--repeats` wall-clock timings.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from wcpred.data import load_shootouts
from wcpred.pipeline import load_predictor
from wcpred.simulation import TournamentFormat, simulate_world_cup

ASOF = pd.Timestamp("2026-06-11")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, nargs="+", default=[1_000, 10_000])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    predictor, results = load_predictor(ASOF)
    fmt, shootouts = TournamentFormat.load(), load_shootouts(asof=ASOF)
    for n in args.runs:
        timings = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            simulate_world_cup(results, fmt, predictor, n_runs=n, seed=0, shootouts=shootouts)
            timings.append(time.perf_counter() - start)
        print(
            f"{n:>7,} runs: best {min(timings):6.2f}s  (all: "
            + ", ".join(f"{t:.2f}" for t in timings)
            + ")"
        )


if __name__ == "__main__":
    main()
