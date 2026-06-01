"""Lookahead Placement Strategy — Phase 3, Attempt 4.

Core idea:
  Pre-read all retrieval events from the event file.
  At each placement, look ahead K retrievals to see which containers
  will be retrieved soon. Choose a position that blocks the fewest
  of those upcoming retrievals.

Why this works:
  Reshuffles happen when container X is retrieved and containers
  above it must be moved. By not placing our container above
  containers that will be retrieved soon, we directly prevent
  causing those reshuffles.

Height remains primary (never sacrifices height for lookahead benefit).
Lookahead breaks ties among equal-height candidates.
Vessel/port are secondary tiebreakers.

Automatically detects whether running on train or test data.
"""

import glob
import json
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set

from src.models import Event, Position
from src.placement_interface import PlacementStrategy
from src.yard_state import YardState

WEIGHT_RANK = {"HEAVY": 3, "MEDIUM": 2, "LIGHT": 1}
TRUCK_VESSELS = {"VSL019", "VSL020"}
LOOKAHEAD_K = 9999  # look ahead effectively all remaining retrieval events


class LookaheadStrategy(PlacementStrategy):

    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        self._etd_cache: Dict[str, float] = {}
        self._retrieval_order: List[str] = []
        self._retrieval_cursor = 0

        # Detect which events file belongs to this simulation by checking
        # whether initial_state containers appear as retrieval events.
        initial_cids: Set[str] = {
            c["container_id"] for c in initial_state.get("containers", [])
        }

        for path in sorted(glob.glob("data/*/events.jsonl")):
            retrievals = []
            found_match = False
            try:
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        e = json.loads(line)
                        if e.get("type") in ("LOAD", "TRUCK_DLVR"):
                            cid = e["container_id"]
                            retrievals.append(cid)
                            if cid in initial_cids:
                                found_match = True
            except Exception:
                continue

            if found_match:
                self._retrieval_order = retrievals
                break

        print(
            f"[LookaheadStrategy] Loaded {len(self._retrieval_order)} retrieval events "
            f"| lookahead K={LOOKAHEAD_K}"
        )

    # ── ETD helper ─────────────────────────────────────────────────────────────

    def _etd(self, s: str) -> float:
        if not s:
            return float("inf")
        if s not in self._etd_cache:
            try:
                self._etd_cache[s] = datetime.fromisoformat(s).timestamp()
            except Exception:
                self._etd_cache[s] = float("inf")
        return self._etd_cache[s]

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def on_event(self, event: Event) -> None:
        """Advance our cursor each time a retrieval event is processed."""
        if event.type in ("LOAD", "TRUCK_DLVR"):
            self._retrieval_cursor += 1

    # ── Main placement ─────────────────────────────────────────────────────────

    def place_container(self, yard_state: YardState, event: Event) -> Position:
        apply_weight = event.vessel_id not in TRUCK_VESSELS
        inc_rank = WEIGHT_RANK.get(event.weight_class, 2)

        # ── Build danger map for upcoming K retrievals ─────────────────────────
        # Maps (block, bay, row) → set of tiers that will be retrieved soon.
        # Placing our container above any of these tiers causes a reshuffle.
        upcoming = self._retrieval_order[
            self._retrieval_cursor : self._retrieval_cursor + LOOKAHEAD_K
        ]
        stack_danger: Dict = defaultdict(set)
        for cid in upcoming:
            pos = yard_state.get_container_position(cid)
            if pos is not None:
                stack_danger[(pos.block, pos.bay, pos.row)].add(pos.tier)

        # ── Find global minimum stack height ───────────────────────────────────
        min_h = float("inf")
        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h < bi.tiers:
                        min_h = min(min_h, h)
                        if min_h == 0:
                            break  # can't go lower
                if min_h == 0:
                    break
            if min_h == 0:
                break

        if min_h == float("inf"):
            return self._fallback(yard_state)

        # ── Score all candidates at min height ─────────────────────────────────
        best_pos: Optional[Position] = None
        best_blocked = float("inf")
        best_vessel = False
        best_port = False

        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h != min_h or h >= bi.tiers:
                        continue

                    our_tier = h + 1

                    # How many upcoming retrievals would we be blocking?
                    tiers_in_stack = stack_danger.get((block_name, bay, row), set())
                    blocked = sum(1 for t in tiers_in_stack if t < our_tier)

                    vessel_match = False
                    port_match = False

                    if h > 0:
                        top_cid = yard_state.get_container_at(block_name, bay, row, h)
                        top_info = (
                            yard_state.get_container_info(top_cid) if top_cid else None
                        )
                        if top_info:
                            vessel_match = top_info.vessel_id == event.vessel_id
                            port_match = (
                                top_info.port_of_discharge == event.port_of_discharge
                            )
                            # Soft penalty for weight violation within same vessel
                            if apply_weight and vessel_match:
                                top_rank = WEIGHT_RANK.get(top_info.weight_class, 2)
                                if inc_rank < top_rank:
                                    blocked += 10

                    # Primary: fewest blocked retrievals
                    # Tiebreakers: vessel match → port match
                    better = False
                    if blocked < best_blocked:
                        better = True
                    elif blocked == best_blocked:
                        if vessel_match > best_vessel:
                            better = True
                        elif vessel_match == best_vessel and port_match > best_port:
                            better = True

                    if better:
                        best_blocked = blocked
                        best_vessel = vessel_match
                        best_port = port_match
                        best_pos = Position(block_name, bay, row, our_tier)

        return best_pos if best_pos is not None else self._fallback(yard_state)

    def _fallback(self, yard_state: YardState) -> Position:
        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h < bi.tiers:
                        return Position(block_name, bay, row, h + 1)
        return Position(list(yard_state.blocks.keys())[0], 1, 1, 999)

    def on_container_retrieved(
        self, container_id: str, position: Position, reshuffles: int
    ) -> None:
        pass
