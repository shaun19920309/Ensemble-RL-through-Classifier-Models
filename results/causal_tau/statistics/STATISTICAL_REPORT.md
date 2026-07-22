# 2020 causal crossover-tau statistical analysis

## Scope and statistical unit

This is a retrospective diagnostic analysis of the 2020 DJ30 mechanism-only causal crossover-tau experiment. It uses no 2021 observations. The primary inferential unit is the 15 RL-pair/classifier-group configurations. The 30 classifier refits within each configuration quantify conditional classifier-fit variation on the same realized market path; they are not treated as 30 independent market samples.

Pair-level base-model descriptors use independently evolved component account curves and are explicitly retrospective. Block-level results use same-state aggressive/conservative/fallback feedback but remain clustered by configuration and common market dates. Reported tests are exploratory associations, not causal estimates and not pre-deployment rules.

## Headline

- Mean Sharpe exceeds the causal single-RL baseline in **12/15** configurations and the retrospectively stronger constituent in **7/15** configurations.
- The selected controller is active in **1,119** of 1,800 blocks. Its same-state advantage over fallback is positive in **758**, tied in **229**, and negative in **132** selected blocks.
- Historical policy advantage has Pearson correlation **0.0345** and Spearman correlation **0.0192** with next-block realized advantage. The configuration-cluster bootstrap 95% interval for Spearman correlation is **[-0.1220, 0.1746]**.
- Only **128** selected blocks contain at least five informative observations on both sides of tau, and the fitted crossover persists in **24/128 (18.8%)**.
- After Benjamini-Hochberg correction across the configuration-varying features, **1** pooled correlations and **6** within-pair correlations remain at q <= 0.05. Pair-level descriptors have only three independent values and are reported without configuration-level significance claims.

## Complete configuration outcomes

| Pair | Group | DeltaS causal | DeltaS stronger | Active days | Active blocks | Selected-block win rate | Mode hit rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A2C+PPO | 1 | 0.0763 | -0.2611 | 0.9331 | 0.9333 | 0.6500 | 0.5556 |
| A2C+PPO | 2 | -0.0560 | -0.3934 | 1.0000 | 1.0000 | 0.7500 | 0.5601 |
| A2C+PPO | 3 | 0.0094 | -0.3279 | 0.1588 | 0.1583 | 0.3021 | 0.7390 |
| A2C+PPO | 4 | 0.0476 | -0.2898 | 1.0000 | 1.0000 | 0.8417 | 0.5643 |
| A2C+PPO | 5 | 0.0401 | -0.2972 | 0.9087 | 0.9083 | 0.9167 | 0.6160 |
| A2C+SAC | 1 | 0.3311 | 0.3112 | 0.6925 | 0.6917 | 0.6552 | 0.5805 |
| A2C+SAC | 2 | 0.4003 | 0.3804 | 0.7510 | 0.7500 | 0.6667 | 0.5829 |
| A2C+SAC | 3 | -0.0114 | -0.0313 | 0.1665 | 0.1667 | 0.1250 | 0.8640 |
| A2C+SAC | 4 | 0.3803 | 0.3604 | 0.7260 | 0.7250 | 0.6667 | 0.5825 |
| A2C+SAC | 5 | 0.1106 | 0.0907 | 0.2339 | 0.2333 | 0.6429 | 0.6409 |
| PPO+SAC | 1 | -0.0270 | -0.2355 | 0.5333 | 0.5333 | 0.4688 | 0.5437 |
| PPO+SAC | 2 | 0.2880 | 0.0795 | 0.7510 | 0.7500 | 0.6667 | 0.5508 |
| PPO+SAC | 3 | 0.0130 | -0.1955 | 0.1007 | 0.1000 | 0.1250 | 0.7201 |
| PPO+SAC | 4 | 0.4213 | 0.2129 | 0.7510 | 0.7500 | 0.6667 | 0.5806 |
| PPO+SAC | 5 | 0.2095 | 0.0011 | 0.6258 | 0.6250 | 0.5467 | 0.6740 |

## Pair-level retrospective structure

| Pair | Sharpe gap | Return corr. | Weak daily win rate | Dominance index | Block winners | Configs > causal | Configs > stronger | Mean DeltaS causal | Mean DeltaS stronger |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A2C+PPO | 0.3266 | 0.9030 | 0.4762 | 0.0545 | ppo;a2c;a2c;ppo | 4 | 0 | 0.0235 | -0.3139 |
| A2C+SAC | 0.0199 | 0.8127 | 0.5119 | 0.0001 | a2c;sac;a2c;sac | 4 | 4 | 0.2422 | 0.2223 |
| PPO+SAC | 0.3067 | 0.8766 | 0.5079 | 0.0389 | ppo;sac;ppo;sac | 4 | 3 | 0.1810 | -0.0275 |

The pair table is useful for explaining the realized 2020 outcomes, but it cannot by itself establish an ex-ante filter: Sharpe gap, full-year return correlation, daily winner share, and block winner sequence all use the completed 2020 path.

## Classifier-group summary

| Group | Pairs | Mean DeltaS causal | Pairs > causal | Mean DeltaS stronger | Pairs > stronger | Active days | Active blocks | Branch divergence | Mode hit rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 0.1268 | 2 | -0.0618 | 1 | 0.7196 | 0.7194 | 1.0000 | 0.5599 |
| 2 | 3 | 0.2108 | 2 | 0.0222 | 2 | 0.8340 | 0.8333 | 1.0000 | 0.5646 |
| 3 | 3 | 0.0037 | 2 | -0.1849 | 0 | 0.1420 | 0.1417 | 0.3455 | 0.7743 |
| 4 | 3 | 0.2831 | 3 | 0.0945 | 2 | 0.8257 | 0.8250 | 1.0000 | 0.5758 |
| 5 | 3 | 0.1201 | 3 | -0.0685 | 2 | 0.5895 | 0.5889 | 0.7586 | 0.6436 |

Each group row contains only three pair-level configuration means, so it should be read descriptively rather than as a population comparison.

## Configuration-level associations with delta Sharpe

### Against the causal single-RL baseline

| feature_label | feature_stage | spearman_rho | permutation_p | permutation_q_bh | within_pair_spearman_rho | within_pair_permutation_p | within_pair_q_bh | analysis_unit | units |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Realized block log advantage | realized_mechanism | 0.8429 | 0.0003 | 0.0042 | 0.8036 | 0.0014 | 0.0193 | 15_pair_group_configurations | 15 |
| Supported crossover persistence | realized_mechanism | 0.8456 | 0.0187 | 0.1402 | 0.5892 | 0.2318 | 0.3245 | 15_pair_group_configurations | 8 |
| Branch divergence rate | decision_structure | 0.4110 | 0.1287 | 0.5593 | 0.6065 | 0.0490 | 0.1716 | 15_pair_group_configurations | 15 |
| Two-sided mode-gap balance | realized_mechanism | -0.3929 | 0.1491 | 0.5593 | -0.3429 | 0.2222 | 0.3245 | 15_pair_group_configurations | 15 |
| Selected-block win rate | realized_mechanism | 0.3084 | 0.2617 | 0.7765 | 0.6476 | 0.0076 | 0.0357 | 15_pair_group_configurations | 15 |
| Mean selected tau | decision_structure | 0.2724 | 0.3234 | 0.7765 | 0.5125 | 0.1039 | 0.2909 | 15_pair_group_configurations | 15 |
| Mean threshold quantile | decision_structure | 0.2502 | 0.3624 | 0.7765 | 0.3914 | 0.2268 | 0.3245 | 15_pair_group_configurations | 15 |
| Historical policy advantage | realized_mechanism | 0.1929 | 0.4929 | 0.7917 | 0.1679 | 0.5980 | 0.6441 | 15_pair_group_configurations | 15 |

### Against the retrospectively stronger constituent

| feature_label | feature_stage | spearman_rho | permutation_p | permutation_q_bh | within_pair_spearman_rho | within_pair_permutation_p | within_pair_q_bh | analysis_unit | units |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Realized block log advantage | realized_mechanism | 0.7071 | 0.0039 | 0.0585 | 0.8036 | 0.0017 | 0.0244 | 15_pair_group_configurations | 15 |
| Two-sided mode-gap balance | realized_mechanism | -0.5821 | 0.0245 | 0.1840 | -0.3429 | 0.2213 | 0.3228 | 15_pair_group_configurations | 15 |
| Selected-agent switch rate | decision_structure | -0.3560 | 0.1870 | 0.6460 | 0.1626 | 0.5433 | 0.6338 | 15_pair_group_configurations | 15 |
| Holding dispersion | decision_structure | 0.3213 | 0.2508 | 0.6460 |  |  |  | 15_pair_group_configurations | 15 |
| Mean threshold quantile | decision_structure | 0.2931 | 0.2848 | 0.6460 | 0.3914 | 0.2231 | 0.3228 | 15_pair_group_configurations | 15 |
| Historical policy advantage | realized_mechanism | 0.2857 | 0.3040 | 0.6460 | 0.1679 | 0.5964 | 0.6423 | 15_pair_group_configurations | 15 |
| Mean selected tau | decision_structure | 0.2724 | 0.3253 | 0.6460 | 0.5125 | 0.1057 | 0.2959 | 15_pair_group_configurations | 15 |
| Branch divergence rate | decision_structure | 0.2619 | 0.3445 | 0.6460 | 0.6065 | 0.0503 | 0.1762 | 15_pair_group_configurations | 15 |

The within-pair columns first remove each RL pair's mean and then permute outcomes only inside the five classifier groups of that pair. They isolate classifier/decision-block differences from the much larger pair-level performance differences.

## Successful-versus-failed profile against the stronger constituent

| feature_label | feature_stage | successful_mean | failed_mean | mean_difference | cliffs_delta | primary_permutation_p | permutation_q_bh | analysis_unit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Realized block log advantage | realized_mechanism | 0.0001 | 0.0000 | 0.0001 | 0.9643 | 0.0200 | 0.3800 | classifier_groups_permuted_within_rl_pair |
| Branch divergence rate | decision_structure | 0.9264 | 0.7284 | 0.1980 | 0.3214 | 0.0800 | 0.6333 | classifier_groups_permuted_within_rl_pair |
| Mean selected tau | decision_structure | 0.2632 | 0.2041 | 0.0591 | 0.2500 | 0.1400 | 0.6333 | classifier_groups_permuted_within_rl_pair |
| Mean threshold quantile | decision_structure | 0.5698 | 0.4172 | 0.1526 | 0.2500 | 0.1400 | 0.6333 | classifier_groups_permuted_within_rl_pair |
| Weaker-model daily win rate | pair_retrospective | 0.5102 | 0.4886 | 0.0216 | 0.7143 | 0.1667 | 0.6333 | 3_rl_pair_clusters |
| Mode hit rate | realized_mechanism | 0.5989 | 0.6453 | -0.0464 | -0.0357 | 0.3200 | 0.6333 | classifier_groups_permuted_within_rl_pair |
| Base block-rank switches | pair_retrospective | 3.0000 | 2.3750 | 0.6250 | 0.6250 | 0.3333 | 0.6333 | 3_rl_pair_clusters |
| Base dominance index | pair_retrospective | 0.0167 | 0.0438 | -0.0271 | -0.7143 | 0.3333 | 0.6333 | 3_rl_pair_clusters |

Positive Cliff's delta means the feature tends to be larger among the seven successful configurations. The primary permutation keeps the RL-pair structure intact: pair-level descriptors are permuted as three whole clusters, while configuration-varying features are permuted only among the five classifier groups inside each pair. Features classified as realized_mechanism already contain next-block return information and must not be interpreted as deployable predictors.

## Historical-signal quartiles

| historical_advantage_quartile | selected_blocks | historical_advantage_mean | realized_advantage_mean | realized_win_rate | supported_both_sides | crossover_persisted |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 280 | -0.0001 | 0.0001 | 0.8714 | 90 | 24 |
| 2 | 280 | -0.0000 | 0.0001 | 0.5571 | 19 | 0 |
| 3 | 279 | -0.0000 | 0.0001 | 0.6631 | 1 | 0 |
| 4 | 280 | 0.0000 | 0.0001 | 0.6179 | 18 | 0 |

A monotone increase in this table would support transport of the historical advantage estimate. The pooled correlation and cluster interval above provide the corresponding continuous diagnostic.

## Interpretation boundary

The analysis can identify associations worth testing in a future causal admission rule, but it cannot validate such a rule on the same 2020 path. Any thresholds derived from these tables would be post-hoc. A valid next step must freeze the candidate features and cutoffs using only pre-2020 completed blocks, then replay the 2020 decision sequence without retuning.
