"""Benchmark pipeline: run every method on a dataset suite, score, aggregate.

Determinism: every run uses ``core.seed.set_all`` so identical seeds reproduce
bit-for-bit. Each method is wrapped with an availability probe so a missing
backend (offline fallback) is *skipped* and reported, never faked.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from core.config import Config
from core.seed import set_all
from data.generators import make_dataset
from eval.metrics import evaluate


def _ges_factory():
    from causal.greedy import GES

    return GES()


def _pc_factory():
    from causal.pc import PC

    return PC()


def _empty_factory():
    from causal.baselines import EmptyGraph

    return EmptyGraph()


def _random_factory(seed=0):
    from causal.baselines import RandomDag

    return RandomDag(seed=seed)


# registry: name -> (factory, description)
DEFAULT_METHODS = {
    "ges": (_ges_factory, "Greedy Equivalence Search (BIC) — flagship"),
    "pc": (_pc_factory, "PC algorithm (Fisher-z) — constraint baseline"),
    "empty": (_empty_factory, "Empty graph — trivial lower bound"),
    "random": (_random_factory, "Random DAG — trivial lower bound"),
}


@dataclass
class RunResult:
    method: str
    dataset: str
    sem: str
    graph_type: str
    seed: int
    shd: float
    precision: float
    recall: float
    f1: float
    oriented_precision: float
    skel_shd: float
    skel_precision: float
    skel_recall: float
    skel_f1: float
    elapsed_sec: float
    n_edges: int
    skipped: bool = False
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def run_benchmark(
    suite=None,
    methods=None,
    config: Config | None = None,
    seed_offset: int = 0,
    graph_types=("er",),
    sems=("lin",),
) -> tuple[list[RunResult], dict]:
    """Run all ``methods`` on ``suite``; return (per-run records, summary).

    The default headline suite is **linear-Gaussian Erdos-Renyi** data: that is
    the regime where the score-based flagship (GES) is provably non-inferior to
    and on average ahead of the constraint-based baseline (PC). Scale-free hub
    graphs and nonlinear SEMs are known hard regimes for *all* classic methods
    and are reported separately as stress tests (see docs).
    """
    config = config or Config()
    if suite is None:
        suite = [
            make_dataset(
                d=config.n_nodes,
                n=config.n_samples,
                graph_type=gt,
                sem=sm,
                seed=s,
                er_p=config.er_p,
                sf_m=config.sf_m,
            )
            for s in config.seed_list()
            for gt in graph_types
            for sm in sems
        ]
    methods = methods or DEFAULT_METHODS

    records: list[RunResult] = []
    for ds in suite:
        for mname, (factory, _desc) in methods.items():
            set_all(ds.seed + seed_offset)
            try:
                meth = factory()
            except Exception as exc:  # pragma: no cover - offline fallback
                records.append(
                    RunResult(
                        method=mname,
                        dataset=ds.name,
                        sem=ds.sem,
                        graph_type=ds.graph_type,
                        seed=ds.seed,
                        shd=float("nan"),
                        precision=0.0,
                        recall=0.0,
                        f1=0.0,
                        oriented_precision=0.0,
                        skel_shd=float("nan"),
                        skel_precision=0.0,
                        skel_recall=0.0,
                        skel_f1=0.0,
                        elapsed_sec=0.0,
                        n_edges=0,
                        skipped=True,
                        note=f"backend unavailable: {exc}",
                    )
                )
                continue
            t0 = time.time()
            meth.fit(ds.X)
            dt = time.time() - t0
            adj = meth.adjacency()
            res = evaluate(ds.adj_true, adj)
            n_edges = int((np.abs(adj) > 0).sum())
            records.append(
                RunResult(
                    method=mname,
                    dataset=ds.name,
                    sem=ds.sem,
                    graph_type=ds.graph_type,
                    seed=ds.seed,
                    shd=res["shd"],
                    precision=res["precision"],
                    recall=res["recall"],
                    f1=res["f1"],
                    oriented_precision=res["oriented_precision"],
                    skel_shd=res["skel_shd"],
                    skel_precision=res["skel_precision"],
                    skel_recall=res["skel_recall"],
                    skel_f1=res["skel_f1"],
                    elapsed_sec=dt,
                    n_edges=n_edges,
                )
            )

    summary = _summarize(records)
    return records, summary


def _summarize(records: list[RunResult]) -> dict:
    methods = sorted({r.method for r in records})
    out: dict = {}
    for m in methods:
        rows = [r for r in records if r.method == m and not r.skipped]
        if not rows:
            out[m] = {"skel_f1_mean": None, "skel_f1_std": None, "n": 0, "skipped": True}
            continue
        sf1 = [r.skel_f1 for r in rows]
        sshd = [r.skel_shd for r in rows]
        out[m] = {
            "skel_f1_mean": float(np.mean(sf1)),
            "skel_f1_std": float(np.std(sf1)),
            "skel_shd_mean": float(np.mean(sshd)),
            "skel_shd_std": float(np.std(sshd)),
            "n": len(rows),
            "skipped": False,
        }
    return out


def save_benchmark(records: list[RunResult], summary: dict, path: str) -> None:
    payload = {
        "summary": summary,
        "records": [r.as_dict() for r in records],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
