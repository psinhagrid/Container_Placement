"""Vessel + Port Grouping Strategy — fixed version.

Two bugs in previous version:
  Bug 1: Height-first caused containers to go to NEW empty stacks instead of
          joining their group's existing stack. Groups never built up.
  Bug 2: Greedy fallback contaminated group stacks with other-group containers.
          When group containers retrieved, intruders above them → reshuffles.

Fix:
  Priority for ship containers:
    Pass 1: existing SAME-GROUP stack where weight ordering is OK
            → pile up in group stack (HEAVY ends up on top via weight rule)
    Pass 2: empty stack (height 0) — start a new stack for this group
    Pass 3: existing same-group stack ignoring weight order — maintain purity
    Pass 4: empty-only fallback — NEVER place on another group's stack
    Pass 5: true last resort — any slot (rare, near-full yard only)

Why weight ordering within group matters:
  LOAD retrieves HEAVY first, then MEDIUM, then LIGHT.
  Stack should be LIGHT (bottom) → MEDIUM → HEAVY (top).
  When HEAVY is on top → retrieved first → 0 reshuffles → cascade down cleanly.
"""

from datetime import datetime
from typing import Dict, Optional

from src.models import Event, Position
from src.placement_interface import PlacementStrategy
from src.yard_state import YardState

WEIGHT_RANK = {"HEAVY": 3, "MEDIUM": 2, "LIGHT": 1}
TRUCK_VESSELS = {"VSL019", "VSL020"}


class VesselPortStrategy(PlacementStrategy):

    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        self._etd_cache: Dict[str, float] = {}

    def _etd(self, s: str) -> float:
        if not s:
            return float("inf")
        if s not in self._etd_cache:
            try:
                self._etd_cache[s] = datetime.fromisoformat(s).timestamp()
            except Exception:
                self._etd_cache[s] = float("inf")
        return self._etd_cache[s]

    def place_container(self, yard_state: YardState, event: Event) -> Position:
        # Truck vessels: no LOAD weight ordering → just greedy
        if event.vessel_id in TRUCK_VESSELS:
            return self._greedy(yard_state)

        inc_rank = WEIGHT_RANK.get(event.weight_class, 2)
        grp_vessel = event.vessel_id
        grp_port = event.port_of_discharge

        # ── Pass 1: existing same-group stack, weight ordering respected ───────
        # PREFERRED — keep piling containers into the same group stack.
        # Weight rule: inc_rank >= top_rank (incoming is heavier or equal → goes on top correctly)
        p1 = self._find_group_stack(yard_state, grp_vessel, grp_port, inc_rank,
                                     weight_strict=True)
        if p1:
            return p1

        # ── Pass 2: empty stack — start a new stack for this group ─────────────
        # Needed when: no existing group stack, or weight violation in all group stacks.
        p2 = self._find_empty(yard_state)
        if p2:
            return p2

        # ── Pass 3: existing same-group stack, weight order relaxed ────────────
        # Group purity over weight ordering — fewer reshuffles than mixing groups.
        p3 = self._find_group_stack(yard_state, grp_vessel, grp_port, inc_rank,
                                     weight_strict=False)
        if p3:
            return p3

        # ── Pass 4: empty-only fallback — yard nearly full, but keep groups pure
        # Already checked in Pass 2; this is a second scan (positions might have freed)
        p4 = self._find_empty(yard_state)
        if p4:
            return p4

        # ── Pass 5: true last resort — any open slot ───────────────────────────
        return self._greedy(yard_state)

    def _find_group_stack(
        self,
        yard_state: YardState,
        vessel: str,
        port: str,
        inc_rank: int,
        weight_strict: bool,
    ) -> Optional[Position]:
        """Find shortest non-empty stack belonging to this (vessel, port) group."""
        best_pos: Optional[Position] = None
        best_h = float("inf")

        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h == 0 or h >= bi.tiers:
                        continue

                    top_cid = yard_state.get_container_at(block_name, bay, row, h)
                    top_info = (
                        yard_state.get_container_info(top_cid) if top_cid else None
                    )
                    if not top_info:
                        continue

                    # Group purity: top must be same vessel AND same port
                    if top_info.vessel_id != vessel or top_info.port_of_discharge != port:
                        continue

                    # Weight ordering (strict mode only)
                    if weight_strict:
                        top_rank = WEIGHT_RANK.get(top_info.weight_class, 2)
                        if inc_rank < top_rank:
                            continue  # lighter on heavier → wrong order

                    if h < best_h:
                        best_h = h
                        best_pos = Position(block_name, bay, row, h + 1)

        return best_pos

    def _find_empty(self, yard_state: YardState) -> Optional[Position]:
        """Find any empty (height 0) stack."""
        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h == 0:
                        return Position(block_name, bay, row, 1)
        return None

    def _greedy(self, yard_state: YardState) -> Position:
        """Shortest stack regardless of group (true last resort)."""
        best_pos: Optional[Position] = None
        best_h = float("inf")

        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h < bi.tiers and h < best_h:
                        best_h = h
                        best_pos = Position(block_name, bay, row, h + 1)
                        if best_h == 0:
                            return best_pos

        return best_pos or Position(
            list(yard_state.blocks.keys())[0], 1, 1, 999
        )

    def on_container_retrieved(
        self, container_id: str, position: Position, reshuffles: int
    ) -> None:
        pass
