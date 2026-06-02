# Container Placement — TODO

## Current Best Results
| Strategy | Train | Test | Score |
|---|---|---|---|
| **19-feat greedy only** | 0.7624 | **0.7326** | **12.9/40 ← BEST TEST** |
| 18-feat + 10% loop | 0.7472 | 0.7383 | 12.6/40 |
| 15-feat + shuffle R4 | **0.7398** | 0.7405 | 12.5/40 |
| Greedy baseline | 0.7873 | 0.7687 | 11.3/40 |

**Target: 25+ / 40**

---

## RUNNING NOW: Path B
```bash
python -m solution.parallel_collector --workers 3 --rounds 10
```
- 15 features, 15% exploration, shuffled initial states
- Val RMSE still falling at 0.5298 — not saturated yet
- DO NOT TOUCH code files while this runs

---

## AFTER B FINISHES: Path A
Add back same_vessel, same_port, weight_ok → 18 features
Hypothesis: those 3 features captured test-specific signals
that were lost when we removed them.

```bash
# 1. Add back 3 features to all 4 pipeline files (greedy_collector,
#    xgb_collector, xgb_strategy, train_xgboost)
# 2. Regenerate clean training data
python -m src.run --strategy solution.greedy_collector.GreedyCollector --data-dir data/train -v
# 3. Train baseline model
python -m solution.train_xgboost
# 4. Run shuffle pipeline
cp data/train/placement_features.csv data/train/placement_features_accum.csv
python -m solution.parallel_collector --workers 3 --rounds 10
# 5. Test best model
python -m src.run --strategy solution.xgb_strategy.XGBStrategy --data-dir data/test -v -o results/results.json
```

---

## Future Diversity Techniques (implement after A/B results)
- [ ] Vessel arrival order shuffle (medium complexity, high impact)
      Shuffle which vessel discharges first, keep LOAD always after DISCHARGE
- [ ] Bootstrap initial state sampling (easy)
      Sample N of 4800 initial containers per worker

---

## Deliverables Remaining
- [ ] results/results.json — update after best model found (currently 12.9/40)
- [ ] docs/design.md — write after final score known
- [ ] tests/ — unit tests (optional, scored)

## Safe to do while B runs
- [ ] Write docs/design.md (just .md file, no code)
- [ ] Write unit tests in tests/
