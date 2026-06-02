# Container Placement — TODO

## Current Best Results
| Strategy | Train | Test | Score |
|---|---|---|---|
| XGBoost 19-feat (greedy only) | 0.7624 | **0.7326** | **12.9/40** ← best test |
| XGBoost 18-feat + 5 iter loop | 0.7472 | 0.7383 | 12.6/40 |
| XGBoost 15-feat (5.9K rows) | 0.7698 | — | — |
| Greedy baseline | 0.7873 | 0.7687 | 11.3/40 |

**Target: 25+ / 40**

---

## Current State
- Features: **15** (removed same_vessel, same_port, weight_ok)
- Training data: 5,929 rows (fresh greedy, 15-feature format)
- Accumulated CSV: 5,929 rows (loop data was lost when reseeding)

---

## Two Options to Run Next

### Option A — Rebuild 35K rows, 10% exploration (standard)
```bash
python -m solution.iteration_loop --iterations 5
```
- 5 iterations × 6K rows = 35K total with 15 clean features
- Same as before but with 3 fewer redundant features
- Expected: ≈ 0.7472 train or better, possibly better test
- Risk: might still overfit to train distribution
- Time: ~10 min

### Option B — Start from 5.9K, 20% exploration (more diverse)
```bash
# First: change line in xgb_collector.py:
# EXPLORE_RATE = 0.20
python -m solution.iteration_loop --iterations 5
```
- 20% random choices → more varied scenarios
- Model sees more edge cases → may generalize better to test
- Collection score slightly worse (more random) but training signal richer
- Key reason: test has DIFFERENT initial state than train → more exploration
  may help model be robust across different initial states
- Time: ~10 min

### Which to pick
```
Option A: safe, rebuilds known-good approach with cleaner features
Option B: riskier but potentially breaks the train/test generalization gap
          Test initial state (day 20) ≠ train initial state (day 0)
          20% exploration generates data from more diverse yard states
```
→ **Try Option B first** (more interesting, addresses root cause of test gap)
→ If worse: fall back to Option A

---

## After Loop Completes
- [ ] Test whichever model scores better on train
- [ ] If beats 0.7326 on test → new best, save results
- [ ] Consider: vessel_containers_in_yard, stack_purity features
- [ ] Consider: 5 more iterations at whichever rate worked better

---

## Deliverables Remaining
- [ ] results/results.json — update after best model found
- [ ] docs/design.md — write after final score known
- [ ] tests/ — unit tests (optional, scored)
