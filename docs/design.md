# Container Yard Placement — Design Document

**Result:** 0.7260 reshuffles/retrieval · 13.2 / 40 · 0 constraint violations

---

## Algorithm Description

I implemented a **self-improving XGBoost scoring strategy** that learns which stack positions minimize future reshuffles from historical simulation data.

At each placement decision:
1. Find all stacks at the **global minimum height** — never sacrifice height balance
2. Extract 18 features describing each candidate stack
3. XGBoost predicts expected reshuffles for each candidate
4. Return the position with the lowest predicted reshuffles

Height-first candidate selection is the most critical design choice. Every approach I tried that overrode height balance (ETD hard rules, block assignment, vessel grouping) was worse than greedy. The insight: shorter stacks have fewer containers above each retrieval target, directly reducing reshuffles regardless of container ordering.

---

## Training Data Generation

I built a **self-improvement loop** that grows the training dataset iteratively:

```
1. Run simulation with XGBoost + 15% random exploration
2. At each placement: record 18 features of the chosen stack
3. At each retrieval: record actual reshuffles caused
4. Append ~6,000 new (features, reshuffles) pairs
5. Retrain XGBoost on ALL accumulated data
6. Keep model only if holdout score improves
7. Repeat
```

The 15% random exploration prevents the feedback loop from collapsing into a narrow distribution — 85% of placements use the current model, 15% are random, ensuring training data covers diverse scenarios.

**Final training data:** ~360,000 labeled examples across 20+ parallel runs.

**To avoid overfitting** to one fixed initial state, I implemented two shuffle techniques:

- **Position shuffle** (`initial_state_shuffler.py`): Randomly reassign the 4,800 initial containers to different physical positions before each collection run, keeping container IDs intact so all LOAD/TRUCK_DLVR events work correctly.
- **Parallel workers**: 3 workers run simultaneously with different random seeds, each starting from a different shuffled yard → 3× more diverse training data per time unit.

**Observed effect:** Without shuffling, the model saturates at ~35K rows with a train-test gap of 0.030. With shuffling, it continues improving to 360K rows with a gap of only 0.007 — the model learns generalizable principles rather than memorized positions.

---

## The 18 Features

All features are defined in `solution/features.py` (single source of truth imported by all pipeline files).

| Feature | Purpose |
|---|---|
| `stack_height` | Primary signal — lower stacks = fewer reshuffles |
| `unsafe_count` | Containers in stack with ETD earlier than ours — they block us |
| `intra_vessel_rank` | Exact loading position: `port_idx × 3 + weight_offset` (0-17) |
| `hours_until_load` | Time until vessel's LOAD window — urgency of correct placement |
| `same_group_in_stack` | Same (vessel + port + weight) containers already present |
| `initial_below_count` | Initial-state containers below us with earlier ETD |
| `unsafe_rank_count` | More precise conflict count using loading rank |
| `unsafe_x_height` | Interaction: ETD conflicts amplified by stack height |
| `min_height_pct` | Yard context — fraction of stacks at minimum height |
| `top_etd_gap_days` | ETD gap between us and the top container |
| `same_vessel / same_port` | Vessel and port grouping quality |
| `weight_ok / weight_rank_inc / top` | Weight ordering (HEAVY retrieved first during LOAD) |
| `block_occ` | Block occupancy — spread load across blocks |
| `days_until_dep` | Days until this container departs |
| `rank_gap_to_top` | Loading rank difference to top container |

---

## Trade-offs and Alternatives Rejected

**Hand-tuned heuristics (6 variants):** Tried ETD ordering, vessel grouping, block assignment, weight ordering as hard rules. Every constraint that skipped "wrong" stacks forced containers onto taller stacks — consistently worse than greedy. Manual weight tuning has no convergence guarantee. *Rejected in favour of learning weights from data.*

**Deep RL:** Requires millions of training episodes. Each simulation takes ~90s. Training would take days. *Rejected due to compute constraints.*

**OR-Tools CP-SAT block assignment:** Assigned vessels to dedicated blocks. Failed because 144 initial empty stacks were insufficient for 10,000+ placements — greedy fallback contaminated everything. *Rejected: wrong abstraction for this problem.*

**Ensemble of two checkpoints:** Built `ensemble_strategy.py` averaging predictions from two model snapshots. Both models learned from the same accumulated data and agreed on every candidate — no improvement. *Available but inactive.*

---

## Train Data Analysis

From the training simulation (days 1-20):
- **10,385 placement events** (DISCHARGE + TRUCK_RECV)
- **10,207 retrieval events** (LOAD + TRUCK_DLVR) — 76% are LOAD (ship containers)
- **Initial state:** 4,800 containers creating a fixed reshuffle floor regardless of strategy
- **Reshuffle distribution:** 60% of retrievals have 0 reshuffles, 28% have 1, 9% have 2, 3% have 3-4

Key finding: the initial state generates ~2,500 unavoidable reshuffles. The remaining ~4,500 reshuffles come from placement decisions, making those decisions the primary optimization target.

The most impactful feature discovered during training: `unsafe_count` (containers in the chosen stack with ETD earlier than ours). This directly predicts future reshuffles caused by our placement and became the second most important feature at 3-5% importance after stack_height.

---

## Time and Space Complexity

**Per placement decision:**
- Candidate collection: O(blocks × bays × rows) = O(1,920) — find global minimum height
- Feature extraction: O(|candidates| × constant) ≈ O(200 × 18)
- XGBoost prediction: O(|candidates| × trees × depth) ≈ O(200 × 100 × 6)
- Total: **O(1,920) per placement**, ~0.01s wall time

**Training:**
- Data collection: O(events × stacks) per simulation run = O(20,000 × 1,920) ≈ O(38M)
- XGBoost training: O(rows × features × trees) = O(360K × 18 × 100) — completes in ~60s
- Total training pipeline: **~3 minutes per round** (3 parallel workers)

**Memory:** Training CSV ~80MB at 360K rows × 19 columns. Model: ~5MB. Yard state tracking: O(containers) ≈ O(10,000) dicts.
