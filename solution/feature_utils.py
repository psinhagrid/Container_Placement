"""Shared feature computation for XGBoost training and inference.

Feature design principles (from domain analysis):
  - Retrieval rank features: encode EXACT loading order (port + weight)
  - Stack conflict features: count containers that will cause reshuffles
  - Interaction features: domain-obvious combinations trees learn faster from
  - Yard context features: same stack quality changes based on yard state
  - Avoid: blind polynomial expansion (XGBoost handles non-linearity itself)
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

WEIGHT_RANK   = {"HEAVY": 3, "MEDIUM": 2, "LIGHT": 1}
WEIGHT_OFFSET = {"HEAVY": 0, "MEDIUM": 1, "LIGHT": 2}  # 0=first loaded, 2=last
TRUCK_VESSELS = {"VSL019", "VSL020"}
PORT_STEP     = 200.0
WEIGHT_STEP   = 60.0
ONE_HOUR      = 3_600.0
SCHEDULE_PATH = "data/vessel_schedule.json"


def load_vessel_ports() -> Dict[str, List[str]]:
    """Load vessel_id → ordered list of ports (loading order = array order)."""
    if not os.path.exists(SCHEDULE_PATH):
        return {}
    with open(SCHEDULE_PATH) as f:
        sched = json.load(f)
    return {
        v["vessel_id"]: v.get("ports", [])
        for v in sched.get("vessels", [])
    }


def parse_ts(s: str, cache: Dict[str, float]) -> float:
    if not s:
        return float("inf")
    if s not in cache:
        try:
            cache[s] = datetime.fromisoformat(s).timestamp()
        except Exception:
            cache[s] = float("inf")
    return cache[s]


def rank_score(vessel_id: str, port: str, weight: str, etd_ts: float,
               vessel_ports: Dict[str, List[str]]) -> float:
    """Compute retrieval rank for a container.

    Lower = retrieved sooner = should be on TOP of stack.
    """
    if vessel_id in TRUCK_VESSELS or etd_ts == float("inf"):
        return etd_ts
    ports    = vessel_ports.get(vessel_id, [])
    port_idx = ports.index(port) if port in ports else len(ports)
    w_off    = WEIGHT_OFFSET.get(weight, 1)
    return etd_ts + port_idx * PORT_STEP + w_off * WEIGHT_STEP


def port_order(vessel_id: str, port: str, vessel_ports: Dict[str, List[str]]) -> int:
    """Port's position in vessel's loading sequence (0 = first loaded)."""
    if vessel_id in TRUCK_VESSELS:
        return 0
    ports = vessel_ports.get(vessel_id, [])
    return ports.index(port) if port in ports else len(ports)


def compute_features(
    block: str, bay: int, row: int, h: int,
    event_vessel: str, event_port: str, event_weight: str,
    event_etd_ts: float, event_ts: float,
    block_occ: float,
    top_info,                             # Container info of top container (or None)
    stack_container_ids: set,             # all container IDs in this stack
    container_etd_map: Dict[str, float],  # container_id → etd timestamp
    container_rank_map: Dict[str, float], # container_id → rank_score
    vessel_ports: Dict[str, List[str]],
    min_height: int,                      # global minimum height across yard
    min_height_count: int,                # number of stacks at min height
    total_stacks: int,                    # total non-full stacks
    max_tiers: int = 5,
) -> dict:
    """Compute the full feature vector for one candidate placement.

    Returns a dict with all feature names → values.
    """
    inc_rank = WEIGHT_RANK.get(event_weight, 2)
    inc_port_order = port_order(event_vessel, event_port, vessel_ports)
    inc_w_offset   = WEIGHT_OFFSET.get(event_weight, 1)
    inc_rank_score = rank_score(event_vessel, event_port, event_weight,
                                event_etd_ts, vessel_ports)

    # Days until departure
    days_until = (
        max(0.0, (event_etd_ts - event_ts) / 86_400)
        if event_etd_ts != float("inf") and event_ts != float("inf")
        else 0.0
    )

    # Top container features
    top_etd_gap, same_vessel, same_port, weight_ok, top_rank, top_rank_score = (
        0.0, 0, 0, 1, 0, float("inf")
    )
    if top_info:
        te = container_etd_map.get(top_info.container_id,
                                   float("inf"))
        if te == float("inf"):
            try:
                te = datetime.fromisoformat(top_info.departure_time).timestamp()
            except Exception:
                te = float("inf")
        top_rank       = WEIGHT_RANK.get(top_info.weight_class, 2)
        top_rank_score = container_rank_map.get(top_info.container_id, float("inf"))
        top_etd_gap    = round((te - event_etd_ts) / 86_400, 4) if te != float("inf") else 0.0
        same_vessel    = int(top_info.vessel_id == event_vessel)
        same_port      = int(top_info.port_of_discharge == event_port)
        if event_vessel not in TRUCK_VESSELS:
            weight_ok  = int(inc_rank >= top_rank)

    # Stack-level features using precomputed maps
    unsafe_count      = 0   # containers with ETD < ours (old feature)
    unsafe_rank_count = 0   # containers with rank_score < ours (more precise)
    same_vessel_count = 0   # same-vessel containers in stack
    for cid in stack_container_ids:
        c_etd  = container_etd_map.get(cid, float("inf"))
        c_rank = container_rank_map.get(cid, float("inf"))
        if c_etd < event_etd_ts - ONE_HOUR:
            unsafe_count += 1
        if c_rank < inc_rank_score - ONE_HOUR:
            unsafe_rank_count += 1
        # We'd need container info for vessel check — approximate with ETD
        # (same vessel = same ETD, so close ETD = likely same vessel)
        if abs(c_etd - event_etd_ts) < ONE_HOUR:
            same_vessel_count += 1

    # Interaction features
    rank_gap_to_top = (
        (inc_rank_score - top_rank_score) / 86_400
        if top_rank_score != float("inf") and inc_rank_score != float("inf")
        else 0.0
    )
    # Positive = we're retrieved later than top (bad — we block top)
    # Negative = we're retrieved sooner (good — top blocks us, but we leave first)

    free_slots = max_tiers - h          # slots above our placement
    intra_vessel_rank = inc_port_order * 3 + inc_w_offset  # 0-17

    # Yard context
    min_height_pct = min_height_count / max(total_stacks, 1)

    return {
        # ── Core features (existing) ──────────────────────────────────────
        "stack_height":     h,
        "top_etd_gap_days": top_etd_gap,
        "same_vessel":      same_vessel,
        "same_port":        same_port,
        "weight_ok":        weight_ok,
        "weight_rank_inc":  inc_rank,
        "weight_rank_top":  top_rank,
        "block_occ":        round(block_occ, 4),
        "days_until_dep":   round(days_until, 4),
        "is_truck":         int(event_vessel in TRUCK_VESSELS),
        "unsafe_count":     unsafe_count,

        # ── New retrieval rank features ───────────────────────────────────
        "port_order":           inc_port_order,      # 0=first port loaded, 5=last
        "weight_offset":        inc_w_offset,         # 0=HEAVY(first), 2=LIGHT(last)
        "intra_vessel_rank":    intra_vessel_rank,    # 0-17 within vessel

        # ── New stack conflict features ───────────────────────────────────
        "unsafe_rank_count":    unsafe_rank_count,    # precise: rank < ours
        "same_vessel_in_stack": same_vessel_count,    # grouping quality
        "free_slots":           free_slots,           # pile-on risk: fewer = safer
        "rank_gap_to_top":      round(rank_gap_to_top, 4),  # interaction with top

        # ── Interaction features ──────────────────────────────────────────
        "unsafe_count_x_height":  unsafe_count * h,   # damage multiplied by height
        "free_slots_x_unsafe":    free_slots * unsafe_rank_count,

        # ── Yard context ──────────────────────────────────────────────────
        "min_height_pct":   round(min_height_pct, 4),  # fraction of stacks at min height
    }


# Complete feature list (same order as compute_features output)
ALL_FEATURES = [
    "stack_height", "top_etd_gap_days", "same_vessel", "same_port",
    "weight_ok", "weight_rank_inc", "weight_rank_top",
    "block_occ", "days_until_dep", "is_truck", "unsafe_count",
    "port_order", "weight_offset", "intra_vessel_rank",
    "unsafe_rank_count", "same_vessel_in_stack", "free_slots", "rank_gap_to_top",
    "unsafe_count_x_height", "free_slots_x_unsafe",
    "min_height_pct",
]
