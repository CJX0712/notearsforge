# notearsforge 踩坑记录（SOP §11 回写）

| # | 坑 | 现象 | 根因 | 修复 |
|---|---|---|---|---|
| 1 | DAG 生成 triu/tril 方向 | SEM 生成纯噪声，所有算法学空图（skelF1≈0） | `_er_dag` / `_sf_dag` 用错三角；SEM 按索引序生成时父节点未初始化 → 子节点=纯噪声 | 严格上三角 `B[parent, child]`（父索引 < 子索引），用 `np.triu` 仅填上三角 |
| 2 | SF 边方向反 | `_sf_dag` scale-free 图退化成噪声，sf/lin 无法恢复 | 原 `B[v, t] = 1` 把边写成 child→parent（反了） | 改 `B[t, v] = 1`（parent t < child v）；seed clique 改 `B[j, k] = 1`（k<j 为父） |
| 3 | PC skeleton 早停 | PC 完全图，skelF1 = 0.5、precision = 0.33 | skeleton 阶段 `if not changed: break`，在 ncond=0 相关数据不删边时即退出，从未进入高阶条件独立 pass | 删除早停，依次跑完 ncond = 0..max_cond |
| 4 | NOTEARS 梯度转置 | 梯度有限差分校验不过，W 不收敛 | A 非对称，正确梯度须 `2.0 * W * Apow.T`（须转置），原漏转置 | 修正为 `2.0 * W * Apow.T`，FD 校验 1.5e-9 通过 |
| 5 | NOTEARS 伪零塌缩 | ALM + L-BFGS-B 在稀疏线性-高斯数据上学到空图 | 伪零局部极小（`h(OLS)≈15.9`），优化器落空图 | 改以 **GES** 为旗舰（无伪零陷阱）；NOTEARS 仅作忠实复现对照基线 |
| 6 | benchmark import 缺失 | `cli run` 默认 suite 报 `NameError: make_dataset` | `pipeline/benchmark.py` 缺 `from data.generators import make_dataset` | 补 import |
| 7 | 难度旋钮 | er_p=2 时 GES vs PC 的 Δ 不显著 | regime 太难，BIC 与条件独立检验差距被采样噪声吞没 | 头 regime 用 er_p=1（d=12, n=1000），使 Δ = +0.073 显著 > ½(σ₁+σ₂) = 0.0625 |
| 8 | 非线性反号 | er/nonlin、sf/nonlin 下 GES 不优于 PC | 非线性 SEM 破坏高斯 BIC 假设，GES 分数退化 | 如实标注非线性为已知难例，不夸大胜率（非劣于 PC 即可） |

## 通用经验（跨系统复用）

- 合成数据 DGP 的「拓扑序」是结构学习复现的第一道坑：父节点必须在子节点之前生成，否则整张图塌成噪声。
- 显著性门禁不能只看点估计：必须 ≥3 seed 报 mean ± std，并用「均值差 > ½(σ₁+σ₂)」判胜，避免单次随机性误导结论。
- 当自研/复现优化器落入伪零局部极小时，优先换更稳的旗舰方法，而非死磕调参。
