# Container Placement — Journey Log

---

## Phase 1: Heuristic (Manual Weight Tuning)

**Goal**: Build a scoring function that places containers intelligently using domain knowledge.
**Outcome**: Best score 0.8239 reshuffles/retrieval (train). Could never beat greedy (0.7873).
**Lesson**: Manual weight tuning is guesswork. ETD hard rules that override height balance always hurt. Greedy's global shortest-stack search is hard to beat manually.

---

## Attempt 1: Heuristic with Guessed Weights

**Strategy**: Score every candidate stack on 5 signals — ETD ordering, vessel grouping, port grouping, weight ordering, stack height.

**Weights we guessed:**
- ETD violation: -200
- Same vessel (top): +150
- Weight bad: -80
- Height penalty: -8

**Result on train data:**
- Reshuffles/retrieval: **0.9525** (worse than random at 0.80)
- Score: **10.0 / 40** (only violations component, reshuffles scored 0)

**What went wrong:**
The vessel grouping bonus (+150) was far stronger than the weight ordering penalty (-80). This caused same-vessel containers to get packed into tall stacks, but in the wrong weight order. Since LOAD retrieves HEAVY first, having LIGHT on top of HEAVY in a stack guarantees reshuffles on every retrieval. The height penalty (-8) was also too weak to prevent tall stacks from forming.

---

## Attempt 2: Strict Rules for ETD and Weight Ordering

**Key change**: ETD ordering and weight ordering are now treated as hard rules, not soft preferences.

**New weights:**
- ETD violation: -1000 (strict — beats any grouping bonus)
- Weight bad: -500 (strict — never place lighter on heavier for ship vessels)
- Same vessel (top): +80 (reduced, now a tiebreaker)
- Height penalty: -25 (stronger)
- Block occupancy cached per call → 6x speed improvement (260s → 43s)
- Truck vessels (VSL019/020) skip weight ordering (no HEAVY-first loading)

**Result on train data:**
- Reshuffles/retrieval: **0.9525** — identical to Attempt 1, no improvement

**What went wrong:**
Weight changes had zero effect. Root cause identified by comparing with greedy baseline (0.7873 on train): we were scanning **all tiers** in each stack for ETD violations. Since the initial 4800 containers have mixed ETDs across tiers, almost every non-empty stack got a -1000 penalty and was rejected. We always fell back to empty stacks, and always picked the **first empty slot in B01** (iteration order bias). Result: all containers piled into B01 sequentially, leaving other blocks empty. Greedy's balanced distribution was better.

**Key insight**: Trial and error on weights is wrong. Weights are NOT the problem — the logic is. XGBoost will learn weights from data in Phase 2 anyway.

---

## Attempt 3: Fix the Logic Bug — Check Only Top Container

**Root cause fix**: Only the TOP container matters for a placement decision.

**Changes:**
- Removed full-stack tier scan — only look at top container
- ETD and weight checks become hard `continue` (skip stack entirely)
- Proportional block occupancy spread, ETD proximity reward
- Weights: same vessel +200, same port +80, height -15/tier

**Result on train data:**
- Reshuffles/retrieval: **0.9477** — tiny improvement, still worse than greedy

**What went wrong:**
Same problem: vessel bonus (+200) still dominated height penalty (-15). Built tall same-vessel stacks even though logic was cleaner.

---

## Attempt 4: Height as Primary, ETD/Weight as Hard Filters

**Key change**: Removed all numeric weights. Made height the PRIMARY criterion (like greedy). ETD and weight are now hard skip rules, vessel/port are tiebreakers for equal height only.

**Logic:**
1. Hard skip: top_etd < inc_etd (we'd block it)
2. Hard skip: lighter on heavier (ship vessels only)
3. Among valid stacks: pick SHORTEST (greedy core)
4. Tiebreaker for equal height: same vessel → same port

**Result on train data:**
- Reshuffles/retrieval: **0.8875** — meaningful improvement but still above greedy (0.7873)

**What went wrong:**
ETD hard filtering sometimes forced us onto TALLER stacks when the shortest stacks had early-ETD containers on top (which we'd skip). Greedy uses those short stacks freely. We traded "fewer ETD violations" for "taller stacks" — not always a good trade.

**Key realization:**
Manual weight tuning is hitting a wall. Decided to stop iterating on weights and move to a fundamentally different strategy. XGBoost will handle weight learning in Phase 2.

---

## Attempt 5: ETD-Block Assignment (v5)

**Fundamental change in thinking**: Stop scoring stacks globally. Instead, assign each container to a DEDICATED BLOCK based on its departure_time, then use greedy within that block.

**Block assignment:**
- Ship vessels (VSL001-018): 8 blocks (B01-B08), 2-day ETD windows cycle through blocks
- Truck vessels (VSL019-020): B09, B10

**Within assigned block:**
- Hard skip: top departs >1 day before us
- Hard skip: lighter on heavier (ship only)
- Pick globally shortest valid stack (greedy within block)
- Tiebreaker: same vessel, then same port

**Why this should work:**
All containers for a vessel end up in the same block (same ETD → same block). During LOAD, vessel containers are already geographically grouped. With weight ordering maintained within stacks, reshuffles ≈ 0 for vessel loading.

This is what the problem rubric calls "departure-time-aware" (~0.30-0.40, 17-21 pts).

**Result on train data:**
- Reshuffles/retrieval: **0.9860** — worst result yet (even worse than v1)
- Speed: 3.96s (fastest so far, only scans one block)

**What went wrong:**
Restricting to one block removed the flexibility that makes greedy work. If the assigned block has contaminated stacks (initial-state containers with wrong ETD/weight), we're stuck with bad options. The spill-over logic helps but creates disorganized placements. Block assignment is the wrong abstraction.

**Key lesson:** Every time we overrode greedy's global height-first search with a constraint (ETD hard skip, block restriction), the score got WORSE. Greedy wins because it finds the globally shortest stack. We need to preserve that while adding domain knowledge.

---

## Attempt 6: Greedy + Domain Tiebreakers (v6)

**Final heuristic approach.** Stop fighting greedy. Join it.

Keep greedy's height-first logic (always pick globally shortest stack) and add domain knowledge ONLY as tiebreakers for equal-height stacks:

```
Primary  : shortest height  (greedy — never compromised)
Tie 1    : top ETD >= incoming ETD  (no ETD violation on top)
Tie 2    : same vessel
Tie 3    : same port
Tie 4    : weight ordering correct (ship only)
```

No hard skip rules. No block restrictions. Domain knowledge nudges decisions only when height is equal — never forces a taller stack.

**Result on train data:**
- Reshuffles/retrieval: **0.8239** — best heuristic result, closest to greedy (0.7873)
- Speed: 17.62s

**Analysis:**
Finally beat all previous attempts. The 0.0366 gap vs greedy comes from tiebreakers occasionally preferring a vessel-matched stack over the first-found equal-height stack, creating slight non-uniformity vs greedy's perfectly uniform distribution.

**Decision: Stop heuristic iteration. Move to XGBoost.**
v6 is good enough to generate meaningful training data. XGBoost will learn the weights we've been guessing manually. The heuristic's job is done.

---

## Phase 2: XGBoost (Learned Score Function)

**Goal**: Train XGBoost to predict reshuffles for any candidate stack, using features collected from the Phase 1 heuristic simulation. Use it to score candidates at inference time instead of hand-tuned weights.

**Why XGBoost beats manual weights:**
- We collected 5,929 labeled examples: (stack features at placement time → actual reshuffles)
- XGBoost learns the real relationship from data — no guessing
- At inference: find all shortest valid stacks, batch-predict reshuffles, pick argmin

**Training data summary:**
- Rows: 5,929 placements that were placed AND retrieved during train simulation
- Features: stack_height, top_etd_gap_days, same_vessel, same_port, weight_ok, weight_rank_inc, weight_rank_top, block_occ, days_until_dep, is_truck
- Label: reshuffles (0=3552, 1=1675, 2=568, 3=134)
- Note: is_truck always 0 (truck containers not retrieved in train window)

**Inference strategy (xgb_strategy.py):**
1. Find global minimum stack height across all valid stacks
2. Collect all candidates at min_height and min_height+1
3. Batch-predict reshuffles with XGBoost for all candidates
4. Return position with lowest predicted reshuffles
5. Falls back to v6 heuristic if model unavailable

---

### Phase 2 — Attempt 1: XGBoost Regressor (400 trees, depth 6)

**Model:** XGBRegressor, n_estimators=400, max_depth=6, lr=0.05, subsample=0.8

**Training result:**
- Val RMSE: 0.6925 (best was 0.6689 at step 50 — model overfit after that)
- Val MAE: 0.4890 vs baseline 0.6559 — learning real signal

**Feature importances learned:**
- stack_height: 54% — confirms greedy's core insight
- same_vessel: 8%, same_port: 6%, days_until_dep: 6%, block_occ: 6%
- top_etd_gap_days: 5%, weight signals: ~5% each
- is_truck: 0% — no truck examples in training data

**Simulation result on train data:**
- Reshuffles/retrieval: **0.7808** — FIRST TIME BEATING GREEDY (0.7873)!
- Score: 0.8 / 30 (just crossed the threshold)

**Issue identified:** Model overfits badly. RMSE peaks at step 50 (0.669) then degrades to 0.693 by step 399. We're training 350 unnecessary trees that hurt generalization.

---

### Phase 2 — Attempt 2: XGBoost with Early Stopping

**Fix:** Add `early_stopping_rounds=30` — stop training when val RMSE doesn't improve for 30 rounds. Model will stop around step ~80 instead of 400, using only the genuinely useful trees.

*Result pending...*
