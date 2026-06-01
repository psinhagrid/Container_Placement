"""Departure-Time Aware Placement Strategy — Phase 3.

Core idea:
  Assign each vessel rotation to a dedicated block (from vessel_schedule.json).
  Within that block, enforce strict weight ordering (HEAVY on top).
  Avoid initial-state contaminated stacks — build clean stacks from scratch.

Why this works:
  LOAD events = 75% of all retrievals.
  If same-vessel containers are grouped in one block AND weight-ordered,
  LOAD retrieves: HEAVY (on top, 0 reshuffles) → MEDIUM → LIGHT.
  Result: near-zero reshuffles for vessel loading.

Block assignment:
  - Sort all ship vessel rotations by ETD
  - Assign cyclically to B01-B08 (8 ship blocks)
  - Truck vessels (VSL019/020) → B09, B10

Placement priority for each incoming container:
  1. Clean stacks in assigned block (empty or only our containers)
  2. Clean stacks in any other block (overflow)
  3. Contaminated stacks in assigned block (last resort)
  4. Contaminated stacks anywhere (complete fallback)

Within each tier: greedy height-first, vessel match as tiebreaker.
Weight ordering strictly enforced everywhere (hard skip, not penalty).
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional, Set, Tuple

from src.models import Event, Position
from src.placement_interface import PlacementStrategy
from src.yard_state import YardState

WEIGHT_RANK = {"HEAVY": 3, "MEDIUM": 2, "LIGHT": 1}
TRUCK_VESSELS = {"VSL019", "VSL020"}
SHIP_BLOCKS  = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08"]
TRUCK_BLOCKS = ["B09", "B10"]
ALL_BLOCKS   = SHIP_BLOCKS + TRUCK_BLOCKS

SCHEDULE_PATH = "data/vessel_schedule.json"


class DepartureTimeStrategy(PlacementStrategy):

    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        self._etd_cache: Dict[str, float] = {}

        # Track which (block, bay, row) positions had initial-state containers.
        # These are "contaminated" until cleared by retrieval events.
        self._initial_positions: Set[Tuple[str, int, int]] = set()
        for c in initial_state.get("containers", []):
            p = c["position"]
            self._initial_positions.add((p["block"], p["bay"], p["row"]))

        # Build vessel rotation → block mapping
        self._vessel_block: Dict[Tuple[str, str], str] = {}
        self._build_block_assignment()

    # ── Vessel schedule parsing ────────────────────────────────────────────────

    def _etd(self, s: str) -> float:
        if not s:
            return float("inf")
        if s not in self._etd_cache:
            try:
                self._etd_cache[s] = datetime.fromisoformat(s).timestamp()
            except Exception:
                self._etd_cache[s] = float("inf")
        return self._etd_cache[s]

    def _build_block_assignment(self) -> None:
        if not os.path.exists(SCHEDULE_PATH):
            return

        with open(SCHEDULE_PATH) as f:
            schedule = json.load(f)

        rotations = []
        for vessel in schedule["vessels"]:
            vid = vessel["vessel_id"]
            if vid in TRUCK_VESSELS:
                continue
            for rot in vessel["rotations"]:
                rotations.append((rot["etd"], vid))

        rotations.sort(key=lambda x: x[0])  # sort by ETD string (ISO → lexicographic = chronological)

        for i, (etd_str, vid) in enumerate(rotations):
            block = SHIP_BLOCKS[i % len(SHIP_BLOCKS)]
            self._vessel_block[(vid, etd_str)] = block

    # ── Stack helpers ──────────────────────────────────────────────────────────

    def _is_clean(self, block: str, bay: int, row: int, h: int) -> bool:
        """True if stack has no initial-state containers."""
        if (block, bay, row) not in self._initial_positions:
            return True     # never had initial containers
        return h == 0       # had initial containers but all retrieved → now clean

    def _best_in_block(
        self,
        yard_state: YardState,
        block: str,
        event: Event,
        inc_rank: int,
        apply_weight: bool,
        clean_only: bool,
    ) -> Optional[Position]:
        """Shortest valid stack in block; vessel match as tiebreaker."""
        if block not in yard_state.blocks:
            return None

        bi = yard_state.blocks[block]
        best_pos: Optional[Position] = None
        best_h = float("inf")
        best_vessel = False
        best_port = False

        for bay in range(1, bi.bays + 1):
            for row in range(1, bi.rows + 1):
                h = yard_state.get_stack_height(block, bay, row)
                if h >= bi.tiers:
                    continue

                # Clean filter
                if clean_only and not self._is_clean(block, bay, row, h):
                    continue

                vessel_match = False
                port_match = False

                if h > 0:
                    top_cid = yard_state.get_container_at(block, bay, row, h)
                    top_info = yard_state.get_container_info(top_cid) if top_cid else None
                    if top_info:
                        # Hard weight rule: NEVER place lighter on heavier
                        # (LOAD retrieves HEAVY first → HEAVY must be on top)
                        if apply_weight:
                            top_rank = WEIGHT_RANK.get(top_info.weight_class, 2)
                            if inc_rank < top_rank:
                                continue
                        vessel_match = top_info.vessel_id == event.vessel_id
                        port_match = top_info.port_of_discharge == event.port_of_discharge

                # Greedy height first; vessel/port as tiebreakers
                better = False
                if h < best_h:
                    better = True
                elif h == best_h:
                    if vessel_match > best_vessel:
                        better = True
                    elif vessel_match == best_vessel and port_match > best_port:
                        better = True

                if better:
                    best_h = h
                    best_vessel = vessel_match
                    best_port = port_match
                    best_pos = Position(block, bay, row, h + 1)

        return best_pos

    # ── Main placement ─────────────────────────────────────────────────────────

    def place_container(self, yard_state: YardState, event: Event) -> Position:
        apply_weight = event.vessel_id not in TRUCK_VESSELS
        inc_rank = WEIGHT_RANK.get(event.weight_class, 2)

        # Find the assigned block for this vessel rotation
        key = (event.vessel_id, event.departure_time)
        if event.vessel_id in TRUCK_VESSELS:
            target = TRUCK_BLOCKS[abs(hash(event.vessel_id)) % len(TRUCK_BLOCKS)]
        else:
            target = self._vessel_block.get(key, SHIP_BLOCKS[0])

        # ── Priority 1: clean stacks in assigned block ─────────────────────────
        pos = self._best_in_block(yard_state, target, event, inc_rank, apply_weight, clean_only=True)
        if pos:
            return pos

        # ── Priority 2: clean stacks in any other block (overflow) ────────────
        for block in ALL_BLOCKS:
            if block == target:
                continue
            pos = self._best_in_block(yard_state, block, event, inc_rank, apply_weight, clean_only=True)
            if pos:
                return pos

        # ── Priority 3: contaminated stacks in assigned block ─────────────────
        pos = self._best_in_block(yard_state, target, event, inc_rank, apply_weight, clean_only=False)
        if pos:
            return pos

        # ── Priority 4: any stack anywhere (complete fallback) ─────────────────
        for block in ALL_BLOCKS:
            if block == target:
                continue
            pos = self._best_in_block(yard_state, block, event, inc_rank, apply_weight, clean_only=False)
            if pos:
                return pos

        # Yard full
        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h < bi.tiers:
                        return Position(block_name, bay, row, h + 1)

        return Position(SHIP_BLOCKS[0], 1, 1, 999)

    def on_container_retrieved(self, container_id: str, position: Position,
                                reshuffles: int) -> None:
        pass
