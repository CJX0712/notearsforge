# Architecture — notearsforge

## 1. 设计原则

- **复用优先**：每个算法优先采用经业界验证的世界级开源方法（GES / PC / NOTEARS 的 canonical 形式），仅自研粘合层与确定性/评测基础设施。
- **单向无环调用**：`cli → pipeline → {data, causal, eval} → core`，无反向依赖。
- **全局确定性**：唯一随机入口 `core.seed.set_all(seed)`，保证同 seed 两次运行逐位一致。
- **离线可降级**：强后端缺失时自动 `skipped`，不伪造数字。

## 2. 模块职责

### core/ （基础设施，零领域逻辑）
| 模块 | 职责 |
|------|------|
| `types.py` | `Dataset` / `DagResult` dataclass，构造期 schema 校验 |
| `errors.py` | 稳定错误码 `E100–E500`（Data/NotFitted/Convergence/Config/Backend） |
| `config.py` | `Config` 数据类，`ENV_NTF_*` 覆盖 + 字段白名单 schema 校验 |
| `interfaces.py` | `StructureLearner` Protocol（fit / adjacency 契约） |
| `seed.py` | `set_all(seed)` 全局确定性入口（numpy + random 一次设齐） |

### data/ （合成数据生成，已知真值）
- `_er_dag`：Erdos-Renyi DAG，使用 **`triu`**（parent index < child index），保证 SEM 按索引序生成时父节点已就绪。
- `_sf_dag`：Barabasi-Albert 偏好连接，**`B[parent, child]=1` 且 parent<child**，拓扑序成立。
- `_linear_sem` / `_nonlinear_sem`：按拓扑序 `X[:,j] = f(X[:,parents]) + noise`。
- `make_dataset` / `make_benchmark_suite`：返回带 `adj_true` 的数据集。

### causal/ （领域算法）
- `greedy.py` → `GES`：forward（插入）/ backward（删除）两阶段，高斯 BIC 局部分数，防环 `has_path`。
- `pc.py` → `PC`：skeleton（Fisher-z）+ v-structure + Meek 规则；无向边双向表示以对齐骨架指标。
- `notears_linear.py` → `NOTEARSLinear`：ALM + L-BFGS-B（变量加倍），平滑无环约束 `h(W)`。
- `baselines.py` → `EmptyGraph` / `RandomDag`：弱基线。

### eval/ （结构恢复指标）
`shd`、`directed_prf1`、`oriented_precision`、`skeleton_prf1`、`skeleton_shd`、`evaluate`。
骨架指标对 PC 的 CPDAG 公平（方向无关）。

### pipeline/ （基准流水线）
`run_benchmark(suite, methods, config, seed_offset, graph_types, sems)`：
- 每个 `method` 包 `try/except`：后端异常 → 记 `skipped=True`，不崩溃、不造假。
- `_summarize` 聚合 mean±std；`save_benchmark` 落盘 JSON。
- `DEFAULT_METHODS` 注册 ges/pc/empty/random。

### cli.py
`run`（基准）/ `one`（单次）；`sys.path` 注入 repo 根以兼容直接运行。

## 3. 关键不变量（CI 门禁覆盖）

1. **NOTEARS 梯度**：`∂h/∂W = 2·W·(A^{d-1})ᵀ`，A 非对称 → 必须转置；与有限差分误差 < 1.5e-9（`tests/test_notears_gradient.py`）。
2. **NOTEARS 无环**：严格上三角 `W` 的 `h(W)=0`（`tests/test_notears_gradient.py`）。
3. **DGP 因果信号**：`_er_dag`/`_sf_dag` 严格上三角；生成数据 GES `skelF1>0.5`（`tests/test_generators.py`）。
4. **确定性**：同 seed 两次 `GES`/`PC`/`NOTEARS`/`make_dataset` 输出逐位一致（`tests/test_determinism.py`）。
5. **离线兜底**：缺失后端记录 `skipped`（`tests/test_offline_fallback.py`）。
6. **消融诚实**：`GES(λ=1.0)` 不劣于 `GES(λ=2.0)`（标准 BIC 惩罚）（`tests/test_ablation_cli.py`）。

## 4. 数据流

```
make_dataset(seed) ──► Dataset(X, adj_true)
        │
        ├─► GES().fit(X).adjacency()  ─┐
        ├─► PC().fit(X).adjacency()   ├─► evaluate(adj_true, adj) ──► RunResult
        ├─► NOTEARS.fit(X).adjacency()┘
        │
        └─► run_benchmark ──► summary(mean±std) ──► benchmark.json
```

## 5. 复现预算

- 头 regime 端到端（d=12, n=1000, 3 seeds, 4 方法）≈ 2.4s（CPU）。
- 内存峰值 < 1GB；无任何网络/权重下载。
