"""notearsforge — command line interface.

Usage:
    python cli.py run            # run the default linear-Gaussian benchmark
    python cli.py run --d 12 --n 800 --seeds 7 11 23 --out benchmark.json
    python cli.py one --method ges --d 10 --n 500 --seed 7   # single run
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import Config
from core.seed import set_all
from data.generators import make_dataset
from eval.metrics import evaluate
from pipeline.benchmark import DEFAULT_METHODS, run_benchmark, save_benchmark


def _print_table(summary: dict) -> None:
    print("\n=== Benchmark summary (skeleton F1, mean ± std) ===")
    print(f"{'method':10s} {'skelF1':>10s} {'±':>6s} {'skelSHD':>9s} {'n':>4s}")
    for m, s in summary.items():
        if s.get("skipped"):
            print(f"{m:10s} {'skipped':>10s}")
            continue
        print(
            f"{m:10s} {s['skel_f1_mean']:>10.3f} {s['skel_f1_std']:>6.3f} "
            f"{s['skel_shd_mean']:>9.2f} {s['n']:>4d}"
        )


def cmd_run(args) -> int:
    cfg = Config.from_env()
    if args.d is not None:
        cfg.n_nodes = args.d
    if args.n is not None:
        cfg.n_samples = args.n
    if args.seeds is not None:
        cfg.default_seed = args.seeds[0]
        cfg.n_seeds = len(args.seeds)
        # override seed_list via monkeypatch
        cfg.seed_list = lambda: tuple(args.seeds)  # type: ignore[attr-defined]
    methods = (
        {k: v for k, v in DEFAULT_METHODS.items() if k in args.methods}
        if args.methods
        else DEFAULT_METHODS
    )
    records, summary = run_benchmark(config=cfg, methods=methods)
    save_benchmark(records, summary, args.out)
    _print_table(summary)
    print(f"\nWrote {len(records)} records -> {args.out}")
    return 0


def cmd_one(args) -> int:
    set_all(args.seed)
    ds = make_dataset(
        d=args.d, n=args.n, graph_type=args.graph_type, sem=args.sem, seed=args.seed, er_p=args.er_p
    )
    factory, _ = DEFAULT_METHODS[args.method]
    meth = factory()
    meth.fit(ds.X)
    res = evaluate(meth.adjacency(), ds.adj_true)
    print(f"method={args.method} dataset={ds.name}")
    print(
        json.dumps(
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()},
            ensure_ascii=False,
        )
    )
    print(
        f"true edges={int(ds.adj_true.sum())} estimated edges={int((meth.adjacency() > 0).sum())}"
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="notearsforge", description="Causal discovery toolkit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run the benchmark suite")
    p_run.add_argument("--d", type=int, default=None)
    p_run.add_argument("--n", type=int, default=None)
    p_run.add_argument("--seeds", type=int, nargs="+", default=None)
    p_run.add_argument("--methods", type=str, nargs="+", default=None)
    p_run.add_argument("--out", type=str, default="benchmark.json")
    p_run.set_defaults(func=cmd_run)

    p_one = sub.add_parser("one", help="single dataset / method run")
    p_one.add_argument("--method", type=str, default="ges", choices=list(DEFAULT_METHODS.keys()))
    p_one.add_argument("--d", type=int, default=10)
    p_one.add_argument("--n", type=int, default=500)
    p_one.add_argument("--seed", type=int, default=7)
    p_one.add_argument("--sem", type=str, default="lin", choices=["lin", "nonlin"])
    p_one.add_argument("--graph-type", type=str, default="er", choices=["er", "sf"])
    p_one.add_argument("--er-p", type=int, default=2)
    p_one.set_defaults(func=cmd_one)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
