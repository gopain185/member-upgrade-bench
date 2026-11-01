## Findings

- **The blend sees about `share x solo gain`.** For large solo gains and correlated members, retention is close to the share. It falls below the share only when members are nearly independent.
- **One submission rarely shows it.** A 0.044 solo gain at share 0.2 is +0.004 to +0.009 in the blend, against a paired sd of 0.004-0.006 on 1,200 public rows. 80% power takes 2,000-17,000 rows.
- **A member below the rest of the blend can hurt.** At k = 4, rho = 0.7, a new member of solo 0.55 changes the blend by 0.000 / -0.002 / -0.010 / -0.029 at share 0.1 / 0.2 / 0.33 / 0.5.

## The real blend

On Name That LLM out-of-fold rows the DeBERTa member held 17-22% of the weight, and its wrong-class scores correlated 0.83-0.86 with the LoRA members and 0.64-0.70 with the n-gram SVCs.

| blend | new member's share | flip | sd on 1,200 rows | model expects | public change |
| --- | --- | --- | --- | --- | --- |
| new member replaces the old | 0.216 | 5.3% | 0.0067 | about +0.009 | -0.005 |
| half old, half new | 0.108 | 2.9% | 0.0049 | about +0.005 | -0.002 |
| quarter-weight old, half new | 0.114 | 2.6% | 0.0047 | about +0.005 | -0.002 |
| old kept, half new added | 0.103 | 3.6% | 0.0055 | about +0.005 | -0.006 |

The changes are compatible with zero (-0.4 to -1.1 sd) and below the model's expectation (-1.4 to -2.2 sd), so the expectation is probably too high for this upgrade. They are correlated, not four independent checks. The simulation leaves out the noise of the solo gain itself (sd about 0.012), the Sinkhorn balancing used only in the solo submissions (the gain without it is +0.015 out of fold), and any gain concentrated on rows the other members already get right.

## Limits

Gaussian exchangeable errors, equal members, fixed accuracies of 0.60 and 0.52, and standardised scores rather than raw log-probabilities. Use the rule as an expectation to check against, not a guarantee.
