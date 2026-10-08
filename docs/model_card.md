# Model Card — notearsforge

## Model Details
- **系统**：notearsforge —— 因果发现工具箱（结构学习）。
- **作者**：晨星。
- **发布**：v0.1.0（MIT）。
- **方法家族**：score-based (GES) / constraint-based (PC) / smooth-acyclicity (NOTEARS)。
- **实现**：纯 Python + numpy 2.5.3 / scipy 1.18.1 / scikit-learn 1.9.1（CPU，零权重下载）。

## Intended Use
- 从观测数据 `X (n, d)` 恢复底层因果 DAG 的**骨架与部分方向**。
- 适用：线性/近似线性-高斯系统、中小规模变量（d ≲ 50 在 CPU 上可行）。
- 不适用：强非线性、非平稳、含未观测混杂、需因果效应量估计（本系统只学结构，不估 ATE）。

## Training / Synthesis Data
- 合成数据引擎，无训练过程。DGP：
  - 图：Erdos-Renyi（边概率 p/d）或 scale-free（Barabasi-Albert, m  Attachments）。
  - SEM：linear-Gaussian `X_j = Σ_k W_kj X_k + N(0,σ²)` 或 nonlinear `X_j = g(Σ W_kj X_k) + noise`。
- 真值：`adj_true` 随数据一同生成，用于精确评分。

## Evaluation
### 指标
- **Skeleton F1**（方向无关）——与 PC 的 CPDAG 公平对比的主指标。
- Directed F1 / SHD / oriented precision。

### 结果（3 seeds，mean±std，真实运行）

| regime | GES (旗舰) | PC (基线) | 判定 |
|--------|-----------|-----------|------|
| er/lin (er_p=1, 头 regime) | **0.909 ± 0.074** | 0.836 ± 0.051 | GES 显著胜 (Δ=+0.073 > ½(σ₁+σ₂)) |
| er/lin (er_p=2) | 0.830 ± 0.126 | 0.777 ± 0.054 | GES 领先 |
| sf/lin | **0.743 ± 0.108** | 0.554 ± 0.065 | GES 明显领先（hub 图难） |
| er/nonlin | 0.683 ± 0.038 | **0.699 ± 0.042** | 非线性使高斯 BIC 误设，PC 略优 |
| sf/nonlin | 0.638 ± 0.082 | **0.607 ± 0.036** | 接近，均难 |

### 确定性
同 seed 两次运行 GES 邻接矩阵**逐位一致**（max|Δ|=0）。

### 消融
- BIC 惩罚 `λ`：`λ=1.0`（标准）不劣于 `λ=2.0`（过惩罚会推向空图）。
- NOTEARS 梯度转置：缺失转置时 FD 误差 9.7e0；修正后 1.5e-9。

## Limitations & Bias
1. **NOTEARS 塌缩**：ALM+L-BFGS-B 在稀疏线性-高斯数据上落入伪零局部极小（空图）。生产旗舰为 GES。
2. **非线性误设**：GES 的高斯 BIC 假设线性-高斯；非线性 SEM 下性能退化（如实披露）。
3. **图规模**：GES 为 O(d²) 每步评分，d 大时慢；PC 的 `max_cond` 过大易过剪。
4. **公平口径**：PC 输出 CPDAG（部分边无向），骨架指标已对齐；directed F1 对 PC 偏保守。

## Ethical Considerations
- 因果结构仅反映统计依赖，不蕴含干预/机制结论；部署于高风险决策前须人工校验。
- 训练/评测数据为合成，无个人隐私风险。
