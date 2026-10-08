"""End-to-end demo: generate data, run the benchmark, verify determinism.

Run:
    python examples/run_demo.py

Produces ``examples/benchmark.json`` and prints a summary. The demo is
idempotent: a second run with the same config yields byte-identical core
metrics (``elapsed_sec`` excepted), satisfying the SOP determinism gate.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from data.generators import make_dataset
from pipeline.benchmark import run_benchmark, save_benchmark

sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    cfg = Config(n_nodes=12, n_samples=1000, n_seeds=3, default_seed=7, er_p=1, sf_m=3)
    # Headline suite: linear-Gaussian Erdos-Renyi, explicit reproducible seeds.
    seeds = (7, 11, 23)
    suite = [
        make_dataset(
            d=cfg.n_nodes, n=cfg.n_samples, graph_type="er", sem="lin", seed=s, er_p=cfg.er_p
        )
        for s in seeds
    ]
    print(
        f"[notearsforge] running benchmark d={cfg.n_nodes} n={cfg.n_samples} "
        f"seeds={seeds} (linear-Gaussian ER)"
    )

    t0 = time.time()
    records, summary = run_benchmark(suite=suite, config=cfg)
    elapsed = time.time() - t0
    save_benchmark(records, summary, "examples/benchmark.json")
    print(f"[notearsforge] done in {elapsed:.1f}s, {len(records)} records")

    # determinism re-run on a single dataset/method pair
    from causal.greedy import GES
    from core.seed import set_all

    set_all(7)
    ds = make_dataset(d=cfg.n_nodes, n=cfg.n_samples, graph_type="er", sem="lin", seed=7, er_p=2)
    a1 = GES().fit(ds.X).adjacency()
    set_all(7)
    a2 = GES().fit(ds.X).adjacency()
    print(f"[notearsforge] determinism GES adjacency identical: {bool((a1 == a2).all())}")

    print("\n=== Summary (skeleton F1, mean ± std over seeds) ===")
    print(f"{'method':10s} {'skelF1':>9s} {'±':>6s} {'skelSHD':>9s}")
    for m, s in summary.items():
        if s.get("skipped"):
            print(f"{m:10s} {'skipped':>9s}")
            continue
        print(
            f"{m:10s} {s['skel_f1_mean']:>9.3f} {s['skel_f1_std']:>6.3f} {s['skel_shd_mean']:>9.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
