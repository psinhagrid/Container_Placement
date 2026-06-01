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
- Block occupancy cached per call (performance fix)
- Truck vessels (VSL019/020) skip weight ordering (no HEAVY-first loading)

*Result pending...*
