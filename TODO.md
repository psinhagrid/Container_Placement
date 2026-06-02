# Container Placement — TODO

## Current Best
| Model | Train | Test | Score |
|---|---|---|---|
| 18-feat + shuffle + 184K rows | 0.7328 | **0.7260** | **13.2/40** |
| Target | — | ≤ 0.45 | 25+ /40 |

---

## RUNNING NOW
```bash
python -m solution.parallel_collector --workers 3 --rounds 10
```
- 18 features, 15% exploration
- Position shuffle + event order shuffle (NEW) per worker
- Each round: 3 workers × 6K rows = 18K new rows
- Accumulator: ~183K rows → ~363K after this run

---

## When run finishes
1. Test standard model:
   `python -m src.run --strategy solution.xgb_strategy.XGBStrategy --data-dir data/test -v -o results/results.json`
2. Test ensemble (if breakthrough happened):
   `python -m src.run --strategy solution.ensemble_strategy.EnsembleStrategy --data-dir data/test -v`
3. Keep better result

---

## Next: CP-SAT Vessel Stacking (path to 25+)
Use OR-Tools CP-SAT to find optimal stack arrangement for each vessel BEFORE it loads:
- Know from vessel_schedule: which containers, which ports, which weights
- CP-SAT solves: arrange containers in stacks so LOAD retrieves in exact order → 0 reshuffles
- LOAD events = 75% of retrievals → near-0 for those → score estimate ~36/40

Implementation plan:
1. For each vessel approaching its load window: collect its containers in yard
2. Run CP-SAT: assign each container to a stack position (port+weight ordered)
3. Route future placements to planned positions
4. Handle conflicts when positions are occupied

---

## Also available (easy wins)
- [ ] XGBoost ranking objective (rank:pairwise instead of regression)
- [ ] Bootstrap initial state sampling (sample N of 4800 per worker)
- [ ] Run more rounds if val RMSE still falling

---

## Deliverables
- [x] results/results.json — 0.7260 test (13.2/40)
- [x] report.md — brief technical report on final model
- [ ] docs/design.md — full design document (write last)
- [ ] tests/ — unit tests (optional, scored)
