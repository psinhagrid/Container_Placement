# Container Placement — TODO

## Current Best Results
| Strategy | Train | Test | Score |
|---|---|---|---|
| XGBoost (18 features, greedy data) | 0.7624 | TBD | TBD |
| XGBoost (19 features) | 0.7624 | 0.7326 | 12.9/40 |
| Greedy baseline | 0.7873 | 0.7687 | 11.3/40 |

**Target: 25+ / 40**

---

## NEXT ACTION: Run Iteration Loop
The val RMSE keeps improving (0.6742 — best ever) but simulation score
is stuck at ~0.769. Root cause: only 5929 training rows with 18 features.
More data = better generalization.

```bash
cp data/train/placement_features.csv data/train/placement_features_accum.csv
python -m solution.iteration_loop --iterations 5
```

Expected: each iteration adds ~6000 rows, val RMSE stays good, simulation improves.

---

## Feature Removal (DO AFTER ITERATION LOOP)
Wait until we have 30K+ rows before removing — importances stabilize with more data.

Remove (redundant — subsumed by same_group_in_stack):
- [ ] same_vessel (2.59%): same_group_in_stack is more specific
- [ ] same_port (2.76%): same_group_in_stack covers this
- [ ] weight_ok (2.60%): weight_rank_inc + weight_rank_top already capture this

Keep despite low % (genuinely different signals):
- unsafe_rank_count (2.49%): rank-based vs ETD-based, different from unsafe_count
- weight_rank_inc/top: model needs both sides of weight comparison

---

## Features Still to Consider Adding
- vessel_containers_in_yard: count of same-vessel containers in yard (grouping urgency)
- stack_purity: same_vessel_count / stack_height (0-1, quality of grouping)
- etd_rank_percentile: intra_vessel_rank normalized by vessel size

---

## Deliverables Remaining
- [ ] results/results.json — run XGBoost on test data after loop
- [ ] docs/design.md — write after final score is known
- [ ] tests/ — unit tests (optional, scored)
