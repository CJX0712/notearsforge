"""Ablation + CLI smoke tests.

Ablation (GES BIC penalty): the Gaussian BIC local score already carries
-|Pa|/2 log(n); the correct multiplier is lambda_ = 1.0. lambda_ = 2.0
over-penalises edges and, on dense hub (scale-free) data, can drive GES to the
empty graph. We lock that relationship: lambda_=1.0 must not be worse than
lambda_=2.0 in skeleton recovery on data where edges are present.
"""

from __future__ import annotations

from causal.greedy import GES
from data.generators import make_dataset
from eval.metrics import evaluate


def test_ges_bic_lambda_ablation_not_worse_at_one():
    ds = make_dataset(d=12, n=1000, graph_type="er", sem="lin", seed=7, er_p=3)
    g1 = GES(lambda_=1.0).fit(ds.X).adjacency()
    g2 = GES(lambda_=2.0).fit(ds.X).adjacency()
    r1 = evaluate(ds.adj_true, g1)
    r2 = evaluate(ds.adj_true, g2)
    # standard BIC (lambda=1.0) must recover at least as much skeleton as the
    # over-penalised lambda=2.0
    assert r1["skel_f1"] >= r2["skel_f1"] - 1e-9, (
        f"lambda=1.0 ({r1['skel_f1']:.3f}) worse than lambda=2.0 ({r2['skel_f1']:.3f})"
    )


def test_cli_one_runs():
    from cli import main

    rc = main(["one", "--method", "ges", "--d", "8", "--n", "400", "--seed", "7"])
    assert rc == 0


def test_cli_run_writes_file(tmp_path):
    from cli import main

    out = tmp_path / "bench.json"
    rc = main(
        [
            "run",
            "--d",
            "10",
            "--n",
            "400",
            "--seeds",
            "7",
            "11",
            "23",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    import json

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "summary" in payload and "records" in payload
    assert len(payload["records"]) == 3 * 4  # 3 seeds x 4 methods
