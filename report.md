# Container Placement — Technical Report

## Final Model: XGBoost + Self-Improvement Loop

**Result:** 0.7260 reshuffles/retrieval (13.2/40)
**Strategy:** `solution/xgb_strategy.py` using model trained by `solution/parallel_collector.py`

---

## Why XGBoost

The core task is: given ~200 candidate stacks at minimum height, pick the one that will cause the fewest future reshuffles. This is a tabular ranking/scoring problem.

**Why not hand-tuned heuristics:** We tried 6 variants of scoring functions with manually chosen weights. Every approach that enforced hard rules (skip stacks with ETD violations, restrict to assigned blocks) forced containers onto taller stacks than greedy — consistently worse. Manual weight tuning is guesswork with no convergence guarantee.

**Why not deep learning / RL:** Requires millions of training episodes. Each simulation run takes ~90s. Getting meaningful signal from RL would take days of compute. XGBoost on tabular features trains in seconds and achieves strong results with thousands of examples.

**Why XGBoost specifically:** Tabular data, ~350K training rows, 18 features. XGBoost consistently outperforms neural networks on tabular data at this scale, is interpretable through feature importances, and trains in under a minute.

**Key design choice:** Only score candidates at the **global minimum height** (never sacrifice height balance). This preserves greedy's core insight — shorter stacks mean fewer containers above each retrieval target — while using XGBoost to make smarter choices among equal-height candidates.

---

## The 18 Features

Defined in `solution/features.py` (single source of truth — all pipeline files import from here).

| Feature | Why |
|---|---|
| `stack_height` | Primary signal — lower is always better |
| `unsafe_count` | Containers in stack with ETD < ours → we block their retrieval |
| `intra_vessel_rank` | Exact loading position within vessel (port_idx × 3 + weight_offset) |
| `hours_until_load` | Time until vessel's LOAD window — urgency signal |
| `same_group_in_stack` | Same (vessel + port + weight) containers already there → 0 extra reshuffles |
| `initial_below_count` | Initial-state containers below us with earlier ETD → we block them |
| `unsafe_x_height` | Interaction: ETD violations amplified by stack height |
| `min_height_pct` | Yard context — what fraction of stacks are at minimum height |
| `top_etd_gap_days` | ETD gap between us and the top container |
| `same_vessel / same_port` | Grouping quality signals |
| `weight_ok` | Weight ordering check (HEAVY retrieved first during LOAD) |
| `weight_rank_inc / top` | Incoming and top container weight classes |
| `block_occ` | Block occupancy — spread containers evenly |
| `days_until_dep` | Days until our container departs |
| `unsafe_rank_count` | More precise: containers with intra_vessel_rank < ours |
| `rank_gap_to_top` | Rank difference between us and top container |

---

## How Training Data Is Generated

**Each collection run:**
1. `XGBCollector` runs a full 20-day simulation on the training data
2. At every placement: records 18 features of the chosen stack
3. At every retrieval: records the actual reshuffles caused
4. Saves ~6,000 (features, reshuffles) pairs per run

**Self-improvement loop:**
```
Collect (XGBoost + 15% random exploration)
  ↓
Append to accumulated CSV (never discard old data)
  ↓
Retrain XGBoost on ALL accumulated data
  ↓
Evaluate on original initial_state.json
  ↓
Keep model only if it improves
  ↓
Repeat
```

The 15% exploration ensures training data contains diverse placements — not just the model's own decisions — preventing feedback loop collapse.

**Total training data:** ~363,000 examples across ~20 runs from diverse scenarios.

---

## How Shuffling Improved Generalization

**The problem:** Every run starts from the same `initial_state.json`. The model memorizes position-specific patterns ("stack B01/bay3/row2 is always risky on Jan 5") rather than learning generalizable principles.

**Solution — two shuffle layers applied per worker:**

**1. Position shuffle** (`solution/initial_state_shuffler.py`):
Keep the same 4,800 container IDs and attributes, but randomly reassign their physical positions. Container IDs are preserved so all LOAD/TRUCK_DLVR events still work correctly. Each worker starts from a genuinely different yard configuration.

**2. Event order shuffle** (`solution/event_order_shuffler.py`):
Within each vessel rotation's discharge window, shuffle which container (by weight/port) arrives first. HEAVY might arrive first in one worker, LIGHT first in another. Keeps all containers discharged and loaded correctly.

**Effect:** The model can no longer memorize position-specific or arrival-order-specific patterns. It must learn features that generalize: ETD ordering, vessel grouping quality, urgency timing, weight compatibility. These features transfer from training (days 1-20) to test (days 21-40).

**Observed result:**
- Without shuffle: model saturates at ~35K rows, train-test gap = 0.030
- With shuffle: still improving at 363K rows, train-test gap = 0.007

---

## Parallel Workers

Three workers run simultaneously, each with a unique random seed:

```
Worker 0 (seed 1000): shuffled yard A + shuffled discharge order A + exploration path A
Worker 1 (seed 1001): shuffled yard B + shuffled discharge order B + exploration path B
Worker 2 (seed 1002): shuffled yard C + shuffled discharge order C + exploration path C
```

Each worker produces ~6,000 training examples from a genuinely different scenario. Three workers in parallel = same wall-clock time as one worker, but 3× more diverse training data per round.

---

## Ensemble (Optional Enhancement)

`solution/ensemble_strategy.py` averages predictions from two model checkpoints:
- **Primary:** current best model (`xgb_model.pkl`)
- **Secondary:** previous best model (`xgb_model_ensemble.pkl`, saved automatically on each breakthrough)

Averaging reduces variance in borderline decisions where one model is uncertain. The secondary model is automatically saved by the parallel collector whenever a new best is found.
