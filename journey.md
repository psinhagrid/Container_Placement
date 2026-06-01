# Container Placement — Journey Log

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

**Root cause fix**: Only the TOP container matters for a placement decision. We place on top of the stack — the top container is the only one directly interacting with our placement. Lower tiers are irrelevant (we can't change them, and they don't block our container).

**Changes:**
- Removed the full-stack tier scan — now only look at top container
- ETD and weight checks become hard `continue` (skip stack) not score penalties
- Added proportional block occupancy spread (`-20 * occ_ratio`) so empty stacks in different blocks are not treated equally — distributes load across all 10 blocks
- ETD proximity reward: prefer stacks where top ETD is close to ours (tighter cluster)
- Weights: same vessel +200, same port +80, correct weight +50, height -15/tier

*Result pending...*
