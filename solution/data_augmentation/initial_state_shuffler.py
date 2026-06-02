"""Initial State Shuffler — generates diverse training scenarios.

Core idea (user insight):
  Every training run starts from the same initial_state.json → model memorizes
  one specific yard configuration → overfits → performance ceiling.

  Solution: keep the SAME containers (same IDs, same attributes, same ETDs) but
  randomly shuffle WHERE they sit in the yard. Each shuffle creates a genuinely
  different training scenario.

Why this is valid:
  - Container IDs are preserved → LOAD/TRUCK_DLVR events still find the right containers
  - Container attributes (vessel, port, weight, ETD) unchanged → event logic works
  - Only physical positions change → different yard state → different training examples
  - The model learns: "how to place containers optimally regardless of starting config"

Why this is a game-changer:
  With 3 workers × different shuffled states:
    Worker 1: [dense B01, sparse B05] → learns one set of patterns
    Worker 2: [sparse B01, dense B08] → learns different patterns
    Worker 3: [mixed distribution]    → learns another set
  Combined: model learns general placement principles, not memorized yard positions

Usage:
  from solution.initial_state_shuffler import shuffle_initial_state, generate_variants

  # Get one shuffled variant
  variant = shuffle_initial_state(initial_state, seed=42)

  # Get N different variants
  variants = generate_variants(initial_state, n=3)
"""

import json
import random
from collections import defaultdict
from typing import Dict, List


def shuffle_initial_state(initial_state: dict, seed: int = None) -> dict:
    """Randomly reassign container positions while keeping all attributes fixed.

    The yard structure (which positions are filled, stack heights) stays the same.
    Containers are randomly redistributed across those same positions.

    Args:
        initial_state: original initial state dict with 'containers' list
        seed: random seed for reproducibility

    Returns:
        New initial state dict with same containers in different positions
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    containers = initial_state.get("containers", [])
    if not containers:
        return initial_state

    # Collect all currently occupied position slots, grouped by stack
    # Stack structure is preserved (same stacks filled, same heights)
    stack_slots: Dict[tuple, List[int]] = defaultdict(list)
    for c in containers:
        pos = c["position"]
        key = (pos["block"], pos["bay"], pos["row"])
        stack_slots[key].append(pos["tier"])

    # Build ordered list of valid position slots
    # Within each stack: tiers sorted ascending (tier 1 = bottom, filled first)
    all_slots = []
    for key in sorted(stack_slots.keys()):  # deterministic ordering
        for tier in sorted(stack_slots[key]):
            all_slots.append({
                "block": key[0],
                "bay":   key[1],
                "row":   key[2],
                "tier":  tier,
            })

    assert len(all_slots) == len(containers), \
        f"Slot count {len(all_slots)} != container count {len(containers)}"

    # Shuffle the container assignment: which container_id goes to which slot
    container_ids = [c["container_id"] for c in containers]
    rng.shuffle(container_ids)

    # Build lookup: container_id → original attributes
    attrs_by_id = {c["container_id"]: c for c in containers}

    # Assemble new initial state: each container keeps its attributes but gets a new slot
    new_containers = []
    for slot, cid in zip(all_slots, container_ids):
        orig = attrs_by_id[cid]
        new_c = {
            "container_id":     orig["container_id"],
            "position":         slot,
            "size":             orig.get("size", 20),
            "weight_class":     orig.get("weight_class", "MEDIUM"),
            "vessel_id":        orig.get("vessel_id", ""),
            "port_of_discharge": orig.get("port_of_discharge", ""),
            "departure_time":   orig.get("departure_time", ""),
        }
        new_containers.append(new_c)

    return {"containers": new_containers}


def generate_variants(initial_state: dict, n: int,
                      base_seed: int = 0) -> List[dict]:
    """Generate N distinct shuffled variants of the initial state.

    Each variant uses a different seed → different container arrangement.
    Used by parallel workers to each start from a unique yard configuration.

    Args:
        initial_state: original initial state
        n: number of variants to generate
        base_seed: base seed (variant i uses seed = base_seed + i)

    Returns:
        List of n shuffled initial state dicts
    """
    return [
        shuffle_initial_state(initial_state, seed=base_seed + i)
        for i in range(n)
    ]


def save_variant(variant: dict, path: str) -> None:
    """Save a shuffled initial state to a JSON file."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(variant, f)


if __name__ == "__main__":
    # Quick test: shuffle and verify container count + IDs are preserved
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/train/initial_state.json"

    with open(path) as f:
        original = json.load(f)

    variant = shuffle_initial_state(original, seed=42)

    orig_ids  = {c["container_id"] for c in original["containers"]}
    var_ids   = {c["container_id"] for c in variant["containers"]}
    orig_positions = {(c["container_id"], c["position"]["block"],
                       c["position"]["bay"], c["position"]["row"],
                       c["position"]["tier"])
                      for c in original["containers"]}
    var_positions  = {(c["container_id"], c["position"]["block"],
                       c["position"]["bay"], c["position"]["row"],
                       c["position"]["tier"])
                      for c in variant["containers"]}

    print(f"Original containers: {len(original['containers'])}")
    print(f"Variant  containers: {len(variant['containers'])}")
    print(f"IDs preserved:       {orig_ids == var_ids}")
    print(f"Positions changed:   {orig_positions != var_positions}")
    changed = len(orig_positions - var_positions)
    print(f"Containers moved:    {changed} / {len(orig_ids)} ({changed/len(orig_ids)*100:.1f}%)")
