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

**Fix:** `early_stopping_rounds=30`, stopped at best iteration 60, val RMSE 0.6681.

**Simulation result on train data:**
- Reshuffles/retrieval: **0.7897** — WORSE than attempt 1 (0.7808), slightly worse than greedy (0.7873)

**Why early stopping hurt:**
With only 5929 training examples, the trees from step 60-400 weren't just noise — they were capturing real but subtle patterns. Early stopping cut off useful signal. The 400-tree model's slight overfitting to training distribution actually generalizes better to the sequential simulation.

**Bigger issue identified:**
stack_height importance increased from 54% → 67% with early stopping. This means the model is essentially learning "shorter = better" (greedy's rule) and not much else. Training data generated from v6 heuristic already avoids ETD/vessel mistakes (as tiebreakers), so XGBoost sees little variance in those features → weak signal.

**Root cause of being stuck near greedy:**
Training data from v6 = few ETD violations in features → XGBoost has little to learn beyond height.
Fix: regenerate training data using GREEDY (which makes random ETD/vessel mistakes) → more variance → XGBoost learns stronger ETD/vessel signal.

**Decision: Regenerate training data with greedy, retrain XGBoost.**

---

### Phase 2 — Attempt 3: Greedy Training Data + Min-Height Only Candidates

**Changes:**
- Regenerated training data using GreedyCollector (greedy makes more ETD/vessel mistakes → more signal variance)
- Fixed XGBoost candidate selection: only score stacks at exact min_height (not min_height+1)

**Training result:**
- Val RMSE: 0.7004 (worse than v6 data — greedy data has more noise)
- Feature importances more balanced: stack_height 49%, days_until_dep 9%, block_occ 9%, top_etd_gap_days 8%

**Simulation result on train data:**
- Reshuffles/retrieval: **0.7691** — best result overall, beats greedy (0.7873)
- Score: **11.3 / 40**

---

## Phase 2 — Summary

| Attempt | Train score | Notes |
|---|---|---|
| Greedy baseline | 0.7873 | Reference |
| XGBoost (v6 data, 400 trees) | 0.7808 | First to beat greedy |
| XGBoost (v6 data, early stop) | 0.7897 | Early stopping hurt |
| XGBoost (greedy data, min_h only) | **0.7691** | Best — 11.3/40 |

**Phase 2 conclusion:**
XGBoost as a tiebreaker among shortest stacks beats greedy, but only marginally (1.3/30 pts vs ~0/30 for greedy). The fundamental ceiling: XGBoost learns "shorter = better" (54%+ importance on stack_height) which greedy already handles perfectly. No amount of XGBoost tuning will reach the 17-21 pts range (0.30-0.40 reshuffles/retrieval).

**The gap to 0.30-0.40 requires changing the placement LOGIC, not the scoring model.**
LOAD events = 75% of retrievals. If same-vessel containers are grouped and weight-ordered, LOAD reshuffles → ~0. That alone gets to 0.20-0.35.

---

## Phase 3: Departure-Time Aware Placement

**Goal:** Reach 0.30–0.40 reshuffles/retrieval (17–21 pts) by:
1. Reading vessel schedule → assign each vessel rotation to a dedicated block
2. Within block: enforce strict weight ordering (HEAVY on top, always)
3. Avoid initial-state contaminated stacks — only build on empty/clean stacks
4. XGBoost as final tiebreaker within same-vessel equal-height stacks

**Why this works:**
- LOAD events (75% of retrievals): all same-vessel containers in same block, weight-ordered → 0 reshuffles
- TRUCK_DLVR (25%): still greedy-level, can't control much

**Expected score:** 17–21 pts for reshuffles + 10 pts violations = 27–31 / 40

---

### Phase 3 — Attempt 1: Vessel-Block Assignment + Weight Ordering + Initial-State Avoidance

*Result pending...*

---

## Phase 4: OR-Tools CP-SAT Solver

**What changed in thinking:**
Previous phases tried to score or group containers heuristically. The fundamental problem: every hard rule restricting which stacks to use forces containers to taller stacks → worse than greedy.

**Deep Learning vs OR-Tools:**
- Deep RL: learns a policy through trial and error, needs millions of episodes, weeks to train. Not feasible here.
- OR-Tools CP-SAT: writes constraints as math equations, solver finds the EXACT optimal solution in seconds. No training data needed.
- For this problem: OR-Tools is the right tool.

**Two-phase approach:**

*Phase A (pre-assignment):*
  - Read vessel_schedule.json
  - CP-SAT assigns each of 18 vessels to one of 8 ship blocks
  - Constraint: vessels with OVERLAPPING LOAD WINDOWS must be in different blocks
  - This ensures when vessel A loads from Block B01, vessel B's containers are in B02 (no cross-vessel interference)

*Phase B (per placement):*
  - Route container to vessel's assigned block
  - Within block: same vessel+port grouping + weight ordering
  - Empty-stack-first: never place on initial-state stacks (prevents contamination)

**Why this should work:**
  LOAD events (75% of retrievals) retrieve one vessel at a time from one block.
  If all vessel A containers are in Block B01, weight-ordered:
    HEAVY retrieved first (on top) → 0 reshuffles → cascade cleanly.
  Different blocks for different vessels → no cross-vessel interference.

### Phase 4 — Attempt 1: CP-SAT block assignment + vessel+port grouping

*Result pending...*

---

## Phase 4 — Summary: All Grouping Approaches Failed

**Every grouping approach (vessel+port, block assignment, OR-Tools) scored worse than greedy (0.7873 train).**

Root cause: Initial state has 4800 containers taking ~80% of empty stacks in each block. When empty stacks run out, greedy fallback fires and places containers on group stacks → contaminates group purity → reshuffles.

Key constraint: Initial state is FIXED INPUT. Cannot be changed by our algorithm.

**Wrong assumption we had:** Height must be minimized (like greedy). This is wrong.
Tall stacks with PERFECT ETD ordering beat short stacks with random ordering.
A height-5 stack where each container is retrieved top→bottom → 0 reshuffles.
A height-2 stack with wrong order → 1 reshuffle per retrieval.

---

## Phase 5: ETD-Smart Strategy (Minimum-Violation Fallback)

**Core insight:**
For 0 reshuffles when container X is retrieved: X must be on TOP at retrieval time.
This happens if every container placed above X has ETD ≤ X.etd (retrieved before X).

If all stacks maintain decreasing ETD from bottom to top → 0 reshuffles for ALL retrievals.

**Why previous ETD approaches failed:**
When no ETD-compatible stack existed, we fell to GREEDY fallback (random shortest stack).
Greedy ignores violation SIZE — might pick a stack with 10-day violation vs a stack with 1-day violation.
A 10-day violation guarantees a reshuffle. A 1-hour violation is nearly harmless.

**The fix: Minimum-Violation Fallback**

Pass 1: ETD-compatible stacks (top_etd ≥ inc_etd) + weight-compatible → shortest, vessel tiebreaker
Pass 2: Empty stacks (no ETD constraint, always safe)
Pass 3: Minimum-violation fallback → stack with smallest (violation_days + height) score
Pass 4: Pure greedy (absolute last resort)

**Expected math:**
  - Initial state generates ~2500 reshuffles (fixed, can't control)
  - Our ~6400 new containers: if 80% use Pass 1/2 (0 reshuffles), 20% use Pass 3 (small violations ~0.5):
  - Our reshuffles: 6400 × 0.20 × 0.5 = 640
  - Total: (2500 + 640) / 9647 = 0.325 → 17 pts reshuffles + 10 = 27 total

### Phase 5 — Attempt 1: ETD-Smart with Minimum-Violation Fallback

*Result pending...*

**Result on train data:**
- Reshuffles/retrieval: **0.9043** — worse than greedy (0.7873) and XGBoost
- Train and test scores identical (0.9043 vs 0.9043) — test initial state is NOT cleaner

**Why the math was off:**
Pass 2 (empty stacks) runs out after ~5,244 placements (not 80% of placements).
Pass 3 (min violation) picks TALLER stacks than greedy → more containers below us → more reshuffles.
The math assumed 80% of placements use Pass 1/2 — actual was closer to 30%.

---

## Phase 5 — Attempt 2: Precomputed Min-ETD Strategy

**New insight:** We know ALL 4,800 initial containers and their positions in `initialize()` BEFORE event 1.
We precompute `stack_min_etd[(block,bay,row)]` = minimum ETD of any container in each stack.
This gives O(1) safety check: if stack_min_etd >= inc_etd → safe (0 reshuffles we add).
Live updates: `on_container_retrieved()` recomputes min_etd when containers leave.

**Result on train data:** 0.8357 reshuffles/retrieval
**Result on test data:** 0.8358 reshuffles/retrieval (identical — test initial state has same mixed ETD distribution)

**Why test ≠ better:** Test initial state containers have ETDs in Jan 21–Feb 9 range. New containers also arrive for Jan 21–Feb 9 vessels. Still plenty of conflicts (e.g., Jan 22 top vs Jan 25 incoming = still unsafe). The test data is NOT significantly cleaner for our checks.

---

## Phase 5 — Attempt 3: Ranked Strategy (Exact Retrieval Order)

**Root cause of all failures identified:**
min_etd and ETD ordering treat all same-vessel containers as equal (same ETD).
But the problem spec says LOAD retrieves: **port-by-port, within each port HEAVY→MEDIUM→LIGHT**.
Two containers from the SAME vessel (same ETD) but different ports/weights are retrieved in a KNOWN order.

**The fix:** Assign every container a retrieval rank using vessel schedule:
```
rank_score = etd_seconds + port_idx * 200 + weight_offset * 60

port_idx    = position of port in vessel's ports array (0=first port loaded)
weight_offset = HEAVY:0, MEDIUM:1, LIGHT:2
```

For VSL001 with ports [PORT_01, PORT_03, PORT_04]:
- PORT_01/HEAVY: rank = etd + 0    (retrieved FIRST → on TOP)
- PORT_01/MEDIUM: rank = etd + 60
- PORT_01/LIGHT:  rank = etd + 120
- PORT_03/HEAVY:  rank = etd + 200
- PORT_03/LIGHT:  rank = etd + 320
- PORT_04/LIGHT:  rank = etd + 520  (retrieved LAST → at BOTTOM)

Stack is SAFE for incoming (rank R) if stack_min_rank >= R.
As containers are retrieved, stack_min_rank rises → more stacks become safe.

**Also cleaned up dead code (deleted):**
lookahead_strategy.py, ortools_strategy.py, vessel_port_strategy.py, etd_smart_strategy.py, departure_time_strategy.py

**Created TODO.md** tracking all remaining improvements.

### Phase 5 — Attempt 3: Ranked Strategy

*Result pending...*
