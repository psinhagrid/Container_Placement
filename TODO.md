# Container Placement — TODO

## Current Best
| Strategy | Train | Test | Score |
|---|---|---|---|
| XGBoost (19 features) | 0.7624 | TBD | 11.6/40 |
| Greedy baseline | 0.7873 | 0.7687 | 11.3/40 |

**Target: 25+ / 40**

---

## Active: Self-Improvement Loop
- [x] Fix xgb_collector.py → same 19 features as greedy_collector
- [x] Fix NaN crash (column mismatch in accumulated CSV)
- [ ] Run: greedy_collector → train → iteration_loop x5

## Next Features to Add
- [ ] hours_until_load: (vessel_load_start - current_time) / 3600
      → needs on_event() to track current time
      → powerful: if vessel loads in 2h, placement must be accessible NOW
- [ ] same_group_in_stack: containers with same (vessel, port, weight) in stack
      → 3 same-group below = retrieved together = 0 extra reshuffles
- [ ] initial_below_count: initial-state containers in stack with ETD < ours
      → more precise than unsafe_count for the initial state bottleneck

## Initial State Bottleneck
Cannot eliminate (~2500 fixed reshuffles from 4800 pre-placed containers).
Can REDUCE by:
- Not placing our containers above initial containers loading soon
- hours_until_load + initial_below_count features teach XGBoost to avoid this

## Deliverables Remaining
- [ ] results/results.json — run XGBoost on test data after loop
- [ ] docs/design.md — write after final score is known
- [ ] tests/ — unit tests (optional but scored)
