# Container Placement — TODO & Strategy Tracker

## Current Best Results
| Strategy | Train | Test | Score |
|---|---|---|---|
| Greedy baseline | 0.7873 | 0.7687 | 11.3/40 |
| XGBoost (unsafe_count) | 0.7664 | TBD | TBD |
| MinETD precomputed | 0.8357 | 0.8358 | 10.0/40 |

**Target: 25+ / 40** (requires reshuffles/retrieval ≤ 0.45)

---

## What We've Used
- [x] stack_height as primary placement criterion (greedy core)
- [x] top container ETD check (only top tier — missed lower tiers)
- [x] vessel/port as tiebreakers
- [x] weight ordering (HEAVY on top)
- [x] XGBoost learned weights from training data
- [x] Precomputed stack_min_etd from initial state (all tiers via O(1) lookup)
- [x] unsafe_count feature for XGBoost (containers with ETD < ours in stack)

## What We Haven't Used (HIGH PRIORITY)

### 1. Exact Retrieval Rank from Vessel Schedule
The problem spec says LOAD retrieves: port-by-port, within each port HEAVY→MEDIUM→LIGHT.
Vessel schedule has `ports` array = loading order.
We can compute for every container: its EXACT retrieval rank.
rank_score = etd + port_idx * 200 + weight_offset * 60
Stack safe if stack_min_rank >= incoming_rank.
→ IMPLEMENTING NOW: ranked_strategy.py

### 2. on_event() callback for ALL events
Called before every event including LOAD/TRUCK_DLVR.
We see the container_id about to be retrieved BEFORE place_container() is called next.
Can maintain a real-time buffer of upcoming retrievals.
→ Add to ranked_strategy.py

### 3. get_containers_by_vessel(vessel_id) 
yard_state method we've never called.
Returns all current containers for a vessel (accurate even after simulator reshuffles).
Use to cluster same-vessel containers at placement time.
→ Add to ranked_strategy.py

### 4. snapshot()/restore() for lookahead
Provided by yard_state explicitly for this purpose.
For top-5 candidates: snapshot → simulate 3 events → count reshuffles → restore.
→ Add if ranked_strategy doesn't reach target

### 5. Port order determination from train data
Analyze train events to confirm: does ports array order = loading order?
Could improve ranked_strategy if our assumption is wrong.
→ Analysis task

### 6. Retrieval rank as XGBoost feature
Add rank_score to greedy_collector features, retrain XGBoost.
Could improve XGBoost significantly (more precise than unsafe_count alone).
→ After ranked_strategy is working

## Cleanup (Done)
- [x] Deleted: lookahead_strategy.py (invalid — read future event files)
- [x] Deleted: ortools_strategy.py (OR-Tools numpy compatibility + failed)
- [x] Deleted: vessel_port_strategy.py (consistently worse than greedy)
- [x] Deleted: etd_smart_strategy.py (superseded by min_etd + ranked)
- [x] Deleted: departure_time_strategy.py (all variants failed)

## Files to Keep
- solution/heuristic_strategy.py — v6 reference heuristic
- solution/xgb_strategy.py — best overall strategy
- solution/min_etd_strategy.py — min-ETD reference
- solution/greedy_collector.py — training data generation
- solution/train_xgboost.py — model training
- solution/ranked_strategy.py — NEW main strategy

## Deliverables Checklist
- [ ] solution/ranked_strategy.py — main algorithm
- [ ] results/results.json — test data output
- [ ] docs/design.md — 1-3 page design document
- [ ] tests/ — unit tests (optional, scored)
