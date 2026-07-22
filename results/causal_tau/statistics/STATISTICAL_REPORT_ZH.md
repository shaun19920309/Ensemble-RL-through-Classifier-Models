# 2020年因果动态Tau实验统计分析

## 分析口径

本分析只使用DJ30的2020年实验记录，不使用任何2021年观测。主要统计单位是15个RL组合-分类器组配置。每个配置中的30次运行共享同一市场路径和同一组确定性RL候选，只反映分类器重拟合波动，因此没有被当成30个独立市场样本。

配对层指标使用2020年完整路径，只能解释已经发生的结果；机制层的已实现收益、命中率和后验胜率也包含结果信息。本报告只做事后统计诊断，不在这里构造或验证准入条件。

## 总体结果

- 15个配置中，12个平均Sharpe超过因果单RL，7个超过事后最强单RL。
- 1,800个评估块中有1,119个启用集成。启用块相对因果fallback为758胜、229平、132负。
- 历史策略优势与下一块真实优势的Pearson相关为0.0345，Spearman相关为0.0192；按15个配置聚类bootstrap后的Spearman 95%区间为[-0.1220, 0.1746]。
- 只有128个启用块在Tau两侧各有至少5个有效样本，其中只有24个保持预期交叉关系，持续率为18.8%。

## RL组合层结果

| RL组合 | 全年Sharpe差距 | 日收益相关性 | 弱模型日胜率 | 支配指数 | 四块赢家序列 | 胜因果单模型配置数 | 胜事后最强配置数 | 平均Sharpe差-因果 | 平均Sharpe差-最强 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A2C+PPO | 0.3266 | 0.9030 | 0.4762 | 0.0545 | ppo;a2c;a2c;ppo | 4 | 0 | 0.0235 | -0.3139 |
| A2C+SAC | 0.0199 | 0.8127 | 0.5119 | 0.0001 | a2c;sac;a2c;sac | 4 | 4 | 0.2422 | 0.2223 |
| PPO+SAC | 0.3067 | 0.8766 | 0.5079 | 0.0389 | ppo;sac;ppo;sac | 4 | 3 | 0.1810 | -0.0275 |

A2C+SAC是2020年最均衡的组合：全年基础Sharpe差仅0.0199，基础日收益相关性0.8127，5个分类器组中有4个超过事后最强单模型。A2C+PPO的基础Sharpe差为0.3266、相关性为0.9030，没有任何分类器组超过最强PPO。

但“基础模型表现接近”不是必要条件：PPO+SAC的基础Sharpe差仍有0.3067，却有3/5配置超过最强单模型。四个63日块中，三个组合的块赢家都发生交替。因此，全年强弱差距能够解释一部分结果，但不能单独解释分类器组之间的成功与失败。这里只有3个独立RL组合，不能对这些配对特征给出可靠总体显著性结论。

## 分类器组结果

| 分类器组 | RL组合数 | 平均Sharpe差-因果 | 胜因果组合数 | 平均Sharpe差-最强 | 胜最强组合数 | 激活日比例 | 激活块比例 | 分支分化率 | 模式命中率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 0.1268 | 2 | -0.0618 | 1 | 0.7196 | 0.7194 | 1.0000 | 0.5599 |
| 2 | 3 | 0.2108 | 2 | 0.0222 | 2 | 0.8340 | 0.8333 | 1.0000 | 0.5646 |
| 3 | 3 | 0.0037 | 2 | -0.1849 | 0 | 0.1420 | 0.1417 | 0.3455 | 0.7743 |
| 4 | 3 | 0.2831 | 3 | 0.0945 | 2 | 0.8257 | 0.8250 | 1.0000 | 0.5758 |
| 5 | 3 | 0.1201 | 3 | -0.0685 | 2 | 0.5895 | 0.5889 | 0.7586 | 0.6436 |

Group 4的平均表现最好：相对因果单模型平均Sharpe差为+0.2831，三个RL组合全部为正；相对事后最强单模型平均差为+0.0945。Group 2次之。

Group 3是最清晰的失效模式：激活日比例只有14.2%，分支分化率只有34.6%，三个组合均未超过最强单模型；但其条件模式命中率反而达到77.4%。这说明只看激活后的命中率会产生误导，覆盖率不足时，高条件命中率不能转化为全年收益优势。

## 同一RL组合内部的统计关联

| 结果变量 | 特征 | 特征性质 | 组合内Spearman | 组合内置换p | BH校正q |
| --- | --- | --- | --- | --- | --- |
| Delta Sharpe vs causal single | Realized block log advantage | realized_mechanism | 0.8036 | 0.0014 | 0.0193 |
| Delta Sharpe vs stronger single | Realized block log advantage | realized_mechanism | 0.8036 | 0.0017 | 0.0244 |
| Delta Sharpe vs causal single | Active day rate | decision_structure | 0.7191 | 0.0040 | 0.0280 |
| Delta Sharpe vs stronger single | Active day rate | decision_structure | 0.7191 | 0.0048 | 0.0337 |
| Delta Sharpe vs causal single | Selected-block win rate | realized_mechanism | 0.6476 | 0.0076 | 0.0357 |
| Delta Sharpe vs stronger single | Selected-block win rate | realized_mechanism | 0.6476 | 0.0083 | 0.0389 |
| Delta Sharpe vs causal single | Branch divergence rate | decision_structure | 0.6065 | 0.0490 | 0.1716 |
| Delta Sharpe vs stronger single | Branch divergence rate | decision_structure | 0.6065 | 0.0503 | 0.1762 |
| Delta Sharpe vs causal single | Mean selected tau | decision_structure | 0.5125 | 0.1039 | 0.2909 |
| Delta Sharpe vs stronger single | Mean selected tau | decision_structure | 0.5125 | 0.1057 | 0.2959 |

在先扣除RL组合平均差异、再只在同一组合的5个分类器组内部置换后，激活日比例与相对因果单模型Sharpe差的Spearman相关为0.7191（p=0.0040，q=0.0280）；相对事后最强单模型得到相同方向。

分支分化率的组合内相关也为正，rho=0.6065，但多重校正后q=0.1716，只属于提示性证据。历史policy-advantage均值与最终Sharpe差没有稳定关系：相对因果单模型组合内rho=0.1679，q=0.6441。

## 块级历史信号

| historical_advantage_quartile | selected_blocks | historical_advantage_mean | realized_advantage_mean | realized_win_rate | supported_both_sides | crossover_persisted |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 280 | -0.0001 | 0.0001 | 0.8714 | 90 | 24 |
| 2 | 280 | -0.0000 | 0.0001 | 0.5571 | 19 | 0 |
| 3 | 279 | -0.0000 | 0.0001 | 0.6631 | 1 | 0 |
| 4 | 280 | 0.0000 | 0.0001 | 0.6179 | 18 | 0 |

如果历史优势估计可以迁移到下一块，四分位数升高时真实优势或胜率应大体单调上升。实际结果没有这种关系：历史优势最低的第一四分位反而具有最高的真实胜率。结合接近零的连续相关和跨零的聚类区间，当前history policy advantage不能解释下一块的相对收益。

## 当前可支持的统计结论

1. 2020年的集成收益首先受到RL候选组合结构影响；A2C+SAC的低相关、低全年差距和块级轮换与更高成功率同时出现，但样本只有3个组合。
2. 在固定RL组合后，分类器组造成的主要差异不是条件命中率，而是集成是否有足够的激活覆盖和分支分化机会。
3. 激活块的已实现优势与全年Sharpe改善高度一致，这是收益分解关系，不是事前预测证据。其成功/失败效应量Cliff's delta为0.9643。
4. 当前历史优势估计和拟合交叉点的下一块可迁移性很弱；这部分无法解释为什么某个组合会在下一块继续有效。
5. 所有结论均是2020年单一市场路径上的探索性统计。任何后续阈值若由这些结果直接确定，都属于事后设计，必须另行做严格的因果回放。

## 图表

- `figure_2020_configuration_outcomes.pdf`: 15个配置相对两个基准的Sharpe差热力图。
- `figure_2020_statistical_profiles.pdf`: 配对差距、激活覆盖、块胜率及历史-真实优势关系。
- `figure_2020_feature_correlations.pdf`: 先去除RL组合均值后的组合内Spearman相关汇总。
