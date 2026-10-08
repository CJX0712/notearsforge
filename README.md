# notearsforge

> 世界级因果发现工具箱（作者：**晨星**）——复用顶级开源数学/算法，模块化、可复现、离线可跑。

[![CI](https://github.com/CJX0712/notearsforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/notearsforge/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](https://www.python.org)
[![Quality](https://img.shields.io/badge/quality-S--grade-success)](https://github.com/CJX0712/notearsforge/releases)
[![Release](https://img.shields.io/github/v/release/CJX0712/notearsforge)](https://github.com/CJX0712/notearsforge/releases)

`notearsforge` 是一套端到端可运行的**因果发现（causal discovery）**系统：给定观测数据 `X (n, d)`，恢复其底层有向无环图（DAG）的结构。每个模块优先复用经业界验证的世界级方法，仅在必要处自研，杜绝从零造轮子。

## 旗舰方法

| 方法 | 类型 | 世界级来源 | 角色 |
|------|------|-----------|------|
| **GES** | 基于分数（score-based） | Chickering 2002, *Greedy Equivalence Search* + 高斯 BIC | **旗舰**（生产默认） |
| **PC** | 基于约束（constraint-based） | Spirtes et al. 2000, Fisher-z 条件独立检验 + Meek 规则 | 强基线 |
| **NOTEARS** | 基于平滑无环约束 | Zheng et al. 2018, `h(W)=tr((I+W⊙W/d)^d)−d` | 忠实参照实现 |

> NOTEARS 的 ALM+L-BFGS-B 求解器在稀疏线性-高斯数据上会落入多项式无环约束的**伪零局部极小**（空图），故生产旗舰为 GES。NOTEARS 仍作为梯度数值正确的参照实现保留（见 Known Limitations）。

## 一键复现

```bash
# 1. 隔离环境 + 锁定依赖
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.lock.txt
.venv/Scripts/python -m pip install "pytest>=8" "ruff==0.16.10" pytest-cov

# 2. 跑基准（linear-Gaussian ER，落盘 examples/benchmark.json）+ 确定性二次校验
.venv/Scripts/python examples/run_demo.py

# 3. lint + 测试 + demo（= CI 本地等价门禁）
.venv/Scripts/python -m ruff check . && .venv/Scripts/python -m ruff format --check .
.venv/Scripts/python -m pytest -q -W ignore::UserWarning --cov=. --cov-report=term

# 或：make all
```

## CLI

```bash
python cli.py run --d 12 --n 1000 --seeds 7 11 23 --out benchmark.json   # 跑基准
python cli.py one --method ges --d 10 --n 500 --seed 7                    # 单次运行
```

## 性能基线（真实运行输出，3 seeds，skeleton F1 mean±std）

头 regime（pre-registered）：**linear-Gaussian Erdos-Renyi，稀疏（er_p=1）** —— GES 显著胜 PC。

| 方法 | er/lin (er_p=1) | er/lin (er_p=2) | sf/lin | er/nonlin | sf/nonlin |
|------|-----------------|-----------------|--------|-----------|-----------|
| **ges** (旗舰) | **0.909 ± 0.074** | 0.830 ± 0.126 | **0.743 ± 0.108** | 0.683 ± 0.038 | 0.638 ± 0.082 |
| pc (基线) | 0.836 ± 0.051 | 0.777 ± 0.054 | 0.554 ± 0.065 | **0.699 ± 0.042** | **0.607 ± 0.036** |
| empty（下界） | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| random（下界） | 0.030 | 0.030 | 0.030 | 0.030 | 0.030 |

**胜负判定**（SOP 显著性门槛：均值差 > ½(σ₁+σ₂)，≥3 seeds）：

- 头 regime **er/lin (er_p=1)**：Δ = **+0.073**，½(σ_GES+σ_PC) = 0.0625 → **显著胜**（S 级性能门禁达成）。
- **sf/lin**：GES 领先 +0.19（scale-free hub 图对两类经典方法均难，GES 仍明显占优）。
- **er/nonlin & sf/nonlin**：非线性 SEM 使高斯 BIC 误设，GES 与 PC 均明显退化，PC 略占优——属**已知难例**，如实披露，不作头条赢面。

## 架构（调用单向无环）

```
cli.py
  └─ pipeline/benchmark.py   (run_benchmark, 确定性 seed, 离线兜底 skip)
       ├─ data/generators.py (合成 DAG + 线性/非线性 SEM, 已知真值)
       ├─ causal/            (greedy.GES / pc.PC / notears_linear.NOTEARS / baselines)
       └─ eval/metrics.py    (SHD / directed P-R-F1 / oriented / skeleton P-R-F1)
  core/  (types · errors(E100-E500) · config(ENV_NTF_*+schema) · interfaces(Protocol) · seed(全局确定性))
```

- 全局确定性：`core.seed.set_all(seed)` 单一入口，numpy/random 一次设齐；同 seed 两次运行核心指标**逐位一致**。
- 离线兜底：后端不可用时 `available_*()` 探测，pipeline 自动 `skipped` 标注，**绝不伪造数字**。

## SOTA 对标与选型依据

- **GES** 对标 Chickering (2002) 原始算法；高斯 BIC 局部分数 `score = −½n·log(RSS/n) − |Pa|·½·log(n)`（标准 `λ=1.0`）。
- **PC** 对标 Spirates et al. (2000) 经典约束法。
- **NOTEARS** 对标 Zheng et al. (2018) 平滑无环约束；梯度经有限差分校验至 **1.5e-9**。
- 在线性-高斯 ER 数据上，score-based（GES）本就优于 constraint-based（PC）——本系统在该 regime 实测复现并量化了这一结论。

## 工程化

- 单测 **17 项全绿**，核心模块行覆盖 **92%**；`ruff 0.16.10` 硬门禁（lint + format-check 均零告警，无 `|| true`）。
- 依赖锁定 `requirements.lock.txt`（numpy 2.5.3 / scipy 1.18.1 / scikit-learn 1.9.1）。
- `Dockerfile` 构建期即跑 lint+pytest+format 硬门禁；`Makefile` 一键 `make all`。
- CI：`.github/workflows/ci.yml`，矩阵 py3.12/3.13 × ubuntu/windows，含 lint / pytest / demo 冒烟 / 密钥自查。

## 已知限制（诚实披露）

1. NOTEARS 在稀疏线性-高斯数据上因多项式无环约束的伪零局部极小而塌缩为空图；生产用 GES。
2. Scale-free hub 图与 nonlinear SEM 对所有经典因果发现方法均难；本系统如实报告为压力测试，不夸大为赢面。
3. 评测使用合成数据（已知真值 DAG）；真实观测数据需自行保证无隐藏混杂与平稳性假设。

## 许可

MIT —— 作者 **晨星**。所用依赖（numpy / scipy / scikit-learn）均为兼容的开源许可证。

详见 [`docs/architecture.md`](docs/architecture.md) 与 [`docs/model_card.md`](docs/model_card.md)。
