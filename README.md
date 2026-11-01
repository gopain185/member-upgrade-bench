# member-upgrade-bench

One blend member gets much better on its own. How much of that gain should the blend see, and could you tell from one
leaderboard submission?

I retrained one member of a Name That LLM blend on cleaner data. Alone, its public score rose from 0.522 to 0.566.
Put back in the blend, the score did not rise (0.626 before, 0.620 to 0.624 after, four variants). This package
simulates that situation so the next upgrade comes with an expectation attached.

## The model

18 balanced classes. Each member scores each class as `a * (is true class) + sqrt(rho) * shared noise +
sqrt(1 - rho) * private noise`, so solo accuracy depends only on the signal `a` (computed by numerical integration)
and `rho` only sets how its errors line up with the other members'. The blend is a soft vote: the target member holds
`share` of the weight, the mean of `k` other members (solo accuracy 0.60) holds the rest. A retrained member keeps half
of its private noise variance.

**Replace grid:** the target goes from solo 0.52 to 0.52 + gain (0.01 / 0.02 / 0.044 / 0.08), keeping its weight.
**Add grid:** a new member of solo 0.40-0.65 joins the blend, against the blend alone. Both: k = 2 / 4 / 8,
rho = 0.3-0.9, target rho equal to or half the others', share = 0.1-0.5, 36,000 rows, 4 seeds, common random numbers.

## Findings

- **The blend sees about `share x solo gain`.** For solo gains of 0.044 and 0.08 at rho >= 0.7, retention (blend gain / solo gain) is 0.09-0.10 at share 0.1, 0.17-0.21 at share 0.2, 0.30-0.34 at share 0.33, 0.48-0.50 at share 0.5, for all k. Small gains are noisier: at gain 0.01 the range at share 0.2 widens to 0.11-0.29. Retention falls below the share when the members are nearly independent (k = 8, rho = 0.3: 0.05 / 0.10 / 0.16 / 0.33). Halving the target's rho barely changes it (median absolute change 0.014).
- **So a big solo jump is a small blend gain.** A solo gain of 0.044 at share 0.2 gives +0.0036 to +0.0091.
- **One submission usually cannot see it.** For that case the two blends disagree in correctness on 1.9-4.7% of rows, which
  gives a paired sd of 0.004-0.006 on 1,200 public rows and z of 0.8-2.2. 80% power takes 2,000-17,000 rows.
- **A member below the rest of the blend can hurt at any real weight.** At k = 4, rho = 0.7 (the rest alone at 0.69), a
  new member of solo 0.55 changes the blend by 0.000 / -0.002 / -0.010 / -0.029 at share 0.1 / 0.2 / 0.33 / 0.5.
  Solo 0.60: +0.005 / +0.007 / +0.006 / -0.005. Solo 0.65: +0.010 / +0.017 / +0.022 / +0.020.
- **Tuning the weight mostly decides whether the member belongs at all.** At k = 4, rho = 0.7 the best share is 0 for a
  member of solo 0.52, 0.10 for solo 0.564 and 0.26 for solo 0.60.

## Against the real blend

Name That LLM, out-of-fold (`src/member_upgrade_check.py` in my competition repo, plain blends without grouped-test heads):
the DeBERTa member holds 17-22% of the weight, its wrong-class scores correlate 0.83-0.86 with the LoRA members and
0.64-0.70 with the n-gram SVCs. So the model's rho = 0.7-0.9 cells at share 0.2 are the closest match.

| blend | new member's share | flip against the base | sd on 1,200 rows | model expects | public change |
| --- | --- | --- | --- | --- | --- |
| op1: new member replaces the old | 0.216 | 5.3% | 0.0067 | about +0.009 | -0.005 |
| op2: half old, half new | 0.108 | 2.9% | 0.0049 | about +0.005 | -0.002 |
| op3: quarter-weight old, half new | 0.114 | 2.6% | 0.0047 | about +0.005 | -0.002 |
| op4: old kept, half new added | 0.103 | 3.6% | 0.0055 | about +0.005 | -0.006 |

Against the expectation, the observed changes are -2.2 sd (op1), -1.4 (op2), -1.5 (op3) and -1.9 (op4); against zero, -0.7, -0.4, -0.4 and -1.1. **The results are compatible with no change and sit below the model's expectation, so the expectation is probably too high for this upgrade.** The four are correlated, so they are not four independent checks. Using the out-of-fold solo gain (+0.015) instead of the public one lowers the expectation for op1 to +0.003 (-1.2 sd). Possible reasons the simulation leaves out:

- The solo public gain is itself noisy: with a 16% flip rate between the two solo members its sd is about 0.012.
- The solo submissions used Sinkhorn balancing and the blends did not. On out-of-fold rows, the gain without balancing is
  +0.015 (0.5209 to 0.5361), not +0.044.
- The gain may sit on rows that the other members already classify correctly. The model has no such structure.

## Limits

- Synthetic Gaussian scores: real errors concentrate on particular class pairs and prompts.
- All other members are equal, and the real blend's members are not.
- The real shares are computed from raw weights on log-probabilities, while the model standardises the rest of the blend.
- The other members' accuracy (0.60) and the old member's (0.52) are fixed. Only the retention pattern has been shown,
  not that it holds at other levels.

## Use

    pip install numpy pandas scipy pytest
    python -m pytest tests
    python -m upgradebench.run          # about 3 minutes; writes results/replace.csv and results/add.csv

## License

MIT for the code. The results are CC0 (see `kaggle/DATASHEET.md`).
