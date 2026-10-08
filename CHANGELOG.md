# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/); this project adheres to SemVer.

## [0.1.0] - 2026-10-08

### Added
- **GES** (Greedy Equivalence Search, Chickering 2002) — Gaussian-BIC
  score-based causal discovery. Flagship method; no spurious-zero trap, reliable
  skeleton recovery on linear-Gaussian data.
- **PC** (Spirtes et al. 2000) — constraint-based baseline with Fisher-z
  conditional-independence tests, v-structure orientation and Meek-rule passes.
- **NOTEARS** (Zheng et al. 2018) — linear structural DAG learning with the
  smooth acyclicity constraint `h(W)=tr((I+W⊙W/d)^d)-d`; analytic gradient
  verified vs finite differences to ~1e-9. Included as a faithful reference
  implementation (see Known Limitations).
- Weak baselines: `EmptyGraph`, `RandomDag`.
- Synthetic data engine: Erdos-Renyi + scale-free graphs crossed with
  linear-Gaussian + nonlinear SEMs, with known ground-truth DAGs.
- Deterministic benchmark pipeline (`pipeline/benchmark.py`) with offline-fallback
  skip path (no fabricated numbers when a backend is unavailable).
- CLI (`cli.py run` / `cli.py one`), end-to-end demo (`examples/run_demo.py`).
- Structure-recovery metrics: SHD, directed precision/recall/F1, oriented
  precision, skeleton precision/recall/F1.
- pytest suite (determinism, offline fallback, NOTEARS gradient FD, GES/PC
  recovery, DGP regression, metrics, CLI smoke, BIC-lambda ablation).
- Engineering: `pyproject.toml` (ruff 0.16.10 pinned, hard gate), `requirements`
  + lock, `Dockerfile` (build-time gate), `Makefile`, `.github/workflows/ci.yml`
  (matrix py3.12/3.13 × ubuntu/windows), `docs/architecture.md`,
  `docs/model_card.md`, MIT `LICENSE`.

### Fixed (verified pre-delivery)
- **PC early-stop bug**: the skeleton phase broke out at `ncond=0` (where
  nothing is removed on correlated data) and never reached the conditional-
  independence passes, leaving a complete graph. Removed the premature break.
- **NOTEARS gradient orientation**: the acyclicity constraint has a *non-symmetric*
  matrix `A = I + W⊙W/d`; the correct gradient is `2·W·(A^{d-1})ᵀ` (transpose
  mandatory). The originally published `Σ Aⁱ (W⊙W) A^{d-1-i}` form was wrong
  (finite-difference error 9.7e0); the transposed form matches to 1.5e-9.
- **DGP causal-signal bug (ER)**: `_er_dag` must use `triu` (parent index < child
  index) so the SEM, which evaluates nodes in index order, has each parent's
  value ready when it generates the child. A `tril` version produced noise-only
  data and every algorithm recovered the empty graph (skelF1 = 0).
- **DGP causal-signal bug (SF)**: `_sf_dag` wrote edges with reversed orientation
  (`B[child, parent]`); corrected to `B[parent, child]` (parent index < child
  index), so scale-free data now carries real causal signal.
- **benchmark default-suite import**: `make_dataset` was referenced but not
  imported in `pipeline/benchmark.py`; added to the import.

### Known Limitations
- NOTEARS (augmented-Lagrangian + L-BFGS-B) collapses to the empty graph on the
  polynomial acyclicity constraint for sparse linear-Gaussian data (a known
  spurious-zero local minimum, `h(OLS)≈15.9`); GES is the production flagship.
- Scale-free hub graphs and nonlinear SEMs are genuinely hard for *all* classic
  causal-discovery methods; reported as stress tests, not as headline wins.
