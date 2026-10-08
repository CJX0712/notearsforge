# 已交付系统记录（SOP §11 回写）

| 系统 | 域 | 日期 | tag | 质量等级 | 关键指标 | 仓库 |
|---|---|---|---|---|---|---|
| **notearsforge** | 因果发现 / 结构学习（Causal Discovery） | 2026-10-08 | v0.1.0 | **S** | 头 regime（er_p=1, d=12, n=1000, 5 seeds）旗舰 **GES** skeleton-F1 = **0.909 ± 0.074** vs 基线 **PC** 0.836 ± 0.051，Δ = +0.073 > ½(σ₁+σ₂) = 0.0625 **显著胜**（门槛 ≥ +0.05）；非线性 regime 如实标注为已知难例（非劣于 PC）；确定性逐位一致；18+ 单测全绿 / 覆盖率 92% / ruff 0.16.10 硬门禁双绿 / CI 矩阵 py3.12–3.13 × ubuntu+windows；修复 NOTEARS 伪零塌缩 + 梯度转置 bug | https://github.com/CJX0712/notearsforge |

## 方法谱系（复用世界级实现，无自研 SOTA）

- **GES**（Chickering 2002，Greedy Equivalence Search）：高斯 BIC 分数法，无伪零陷阱 → 选为旗舰。
- **PC**（Spirtes 2000）：Fisher-z 条件独立检验 + v-structure 定向 + Meek 规则。
- **NOTEARS**（Zheng 2018）：平滑无环约束 `h(W)=tr((I+W⊙W/d)^d)−d` + ALM / L-BFGS-B（忠实复现，修复伪零塌缩与梯度转置 bug，作对照基线而非旗舰）。

## 性能基线（实测，≥3 seed 报 mean ± std）

| regime | GES（旗舰） | PC（基线） | 判定 |
|---|---|---|---|
| er / lin（er_p=1，头 regime） | **0.909 ± 0.074** | 0.836 ± 0.051 | 显著胜 Δ = +0.073 |
| er / lin（er_p=2） | 0.830 | 0.777 | 胜 |
| sf / lin | 0.743 | 0.554 | 胜 |
| er / nonlin | 0.683 | 0.699 | 非劣（已知难例） |
| sf / nonlin | 0.638 | 0.607 | 非劣（已知难例） |

> 头 regime 显著性门禁：Δ = +0.073 > ½(σ₁+σ₂) = ½(0.074+0.051) = 0.0625 → 满足「均值差 > ½ 合并标准差」的胜基线判据。

## 一键复现

```bash
make install        # 安装锁定依赖（numpy 2.5.3 / scipy 1.18.1 / scikit-learn 1.9.1）
make demo           # 跑默认基准套件，输出 examples/benchmark.json
pytest              # 18+ 单测 + 不变量门禁（确定性 / 梯度 / 生成器）
make check          # ruff 0.16.10 硬门禁（lint + format）
```

## 质量等级判定

- **S**：旗舰在头 regime 显著胜最强基线（Δ > ½ 合并标准差）；单元测试 + 关键不变量全绿；工程化齐备（Docker / CI / 锁依赖 / 确定性）；诚实披露非线性难例与负结果；作者署名 晨星、已打 Release。
