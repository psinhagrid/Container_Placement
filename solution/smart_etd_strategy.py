"""Smart ETD Strategy — fixes the empty-stack allocation problem.

The Core Insight (from user):
  First-come-first-served for empty stacks is wrong.
  - Late ETD container gets empty stack → many containers pile on → many reshuffles
  - Early ETD container gets empty stack → retrieved soon → 0-1 reshuffles

  But: early ETD containers DON'T NEED empty stacks — most existing stacks are
  compatible for them (existing containers have later ETD = no violation).

  The fix: WIDEN Pass 1 with a grace period so MORE containers find compatible
  non-empty stacks. Empty stacks are then preserved for when truly needed.

Algorithm:
  Pass 1: ANY non-empty stack where stack_min_etd >= inc_etd - GRACE
          → GRACE = 24h by default (accept tiny violations, big gain in coverage)
          → pick globally SHORTEST (height primary, never sacrificed)
          → tiebreaker: same vessel → same port

  Pass 2: empty stacks — ONLY when Pass 1 truly fails (no compatible non-empty)
          → spread by block occupancy

  Pass 3: height-CONSTRAINED minimum violation
          → ONLY at min_height or min_height+1 (never taller than greedy would pick)
          → argmin(unsafe_count) among those height-constrained stacks
          → preserves height balance while minimising damage

  Pass 4: pure greedy (absolute last resort)

Why the grace period helps:
  Without grace (strict):  stack with [Jan3, Jan8] is UNSAFE for Jan4 (Jan3 < Jan4)
  With 24h grace:          stack with [Jan3, Jan8] is SAFE for Jan4 (Jan3 >= Jan4 - 24h)
  → Jan3 and Jan4 are retrieved almost simultaneously → the 1-day violation rarely
    causes an actual reshuffle (Jan3 container is gone by the time Jan4 retrieval fires)
  → But now many more stacks qualify → much higher Pass 1 success rate
  → Empty stacks preserved for containers with no compatible option at all

Why height-constrained Pass 3 fixes previous ETD strategies:
  Old Pass 3: picked height-3 stack with small violation over height-1 with large violation
              → height imbalance → MORE reshuffles overall (each extra tier = more pile-ons)
  New Pass 3: constrained to min_height (like greedy) → height balance preserved
              → picks LEAST UNSAFE among the shortest stacks only
"""

from datetime import datetime
from typing import Dict, Optional

from src.models import Event, Position
from src.placement_interface import PlacementStrategy
from src.yard_state import YardState

WEIGHT_RANK   = {"HEAVY": 3, "MEDIUM": 2, "LIGHT": 1}
TRUCK_VESSELS = {"VSL019", "VSL020"}

# Grace period: accept stacks where min_etd is within GRACE seconds of ours.
# 24h = containers retrieved within 1 day of each other → violation rarely fires.
GRACE = 24 * 3_600.0


class SmartETDStrategy(PlacementStrategy):

    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        self._etd_cache: Dict[str, float] = {}

        # Precompute stack_min_etd from initial state for O(1) lookup
        self._container_etd: Dict[str, float] = {}
        self._stack_containers: Dict[tuple, set] = {}
        self._stack_min_etd:    Dict[tuple, float] = {}

        for c in initial_state.get("containers", []):
            cid = c["container_id"]
            p   = c["position"]
            key = (p["block"], p["bay"], p["row"])
            etd = self._etd(c.get("departure_time", ""))

            self._container_etd[cid] = etd
            self._stack_containers.setdefault(key, set()).add(cid)
            self._stack_min_etd[key] = min(
                self._stack_min_etd.get(key, float("inf")), etd
            )

    def _etd(self, s: str) -> float:
        if not s:
            return float("inf")
        if s not in self._etd_cache:
            try:
                self._etd_cache[s] = datetime.fromisoformat(s).timestamp()
            except Exception:
                self._etd_cache[s] = float("inf")
        return self._etd_cache[s]

    def on_container_retrieved(self, container_id: str, position: Position,
                                reshuffles: int) -> None:
        key = (position.block, position.bay, position.row)
        stack = self._stack_containers.get(key, set())
        stack.discard(container_id)
        self._stack_min_etd[key] = (
            min(self._container_etd.get(c, float("inf")) for c in stack)
            if stack else float("inf")
        )
        self._container_etd.pop(container_id, None)

    def place_container(self, yard_state: YardState, event: Event) -> Position:
        inc_etd      = self._etd(event.departure_time)
        apply_weight = event.vessel_id not in TRUCK_VESSELS
        inc_rank     = WEIGHT_RANK.get(event.weight_class, 2)

        # ── Pass 1: compatible non-empty stacks with grace period ──────────────
        # Accept stacks where stack_min_etd >= inc_etd - GRACE
        # → small violations (< 24h) are nearly harmless
        # → dramatically more stacks qualify → empty stacks preserved
        # Height primary (greedy core), vessel/port as tiebreaker
        best_pos:    Optional[Position] = None
        best_h       = float("inf")
        best_vessel  = False
        best_port    = False

        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h == 0 or h >= bi.tiers:
                        continue

                    key = (block_name, bay, row)
                    min_etd = self._stack_min_etd.get(key, float("inf"))

                    # Accept if min ETD is within GRACE of ours
                    if min_etd < inc_etd - GRACE:
                        continue  # violation too large — skip

                    top_cid  = yard_state.get_container_at(block_name, bay, row, h)
                    top_info = yard_state.get_container_info(top_cid) if top_cid else None

                    # Weight check (ship vessels only)
                    if top_info and apply_weight:
                        top_rank = WEIGHT_RANK.get(top_info.weight_class, 2)
                        if inc_rank < top_rank:
                            continue

                    vessel_match = bool(top_info and top_info.vessel_id == event.vessel_id)
                    port_match   = bool(top_info and top_info.port_of_discharge == event.port_of_discharge)

                    better = False
                    if h < best_h:
                        better = True
                    elif h == best_h:
                        if vessel_match > best_vessel:
                            better = True
                        elif vessel_match == best_vessel and port_match > best_port:
                            better = True

                    if better:
                        best_h      = h
                        best_vessel = vessel_match
                        best_port   = port_match
                        best_pos    = Position(block_name, bay, row, h + 1)

        if best_pos is not None:
            self._update_tracking(event.container_id, best_pos, inc_etd)
            return best_pos

        # ── Pass 2: empty stacks — preserved for when Pass 1 truly fails ───────
        best_occ: float = float("inf")
        for block_name, bi in yard_state.blocks.items():
            occ, cap = yard_state.get_block_occupancy(block_name)
            occ_ratio = occ / cap if cap > 0 else 0.0
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    if yard_state.get_stack_height(block_name, bay, row) == 0:
                        if occ_ratio < best_occ:
                            best_occ = occ_ratio
                            best_pos = Position(block_name, bay, row, 1)

        if best_pos is not None:
            self._update_tracking(event.container_id, best_pos, inc_etd)
            return best_pos

        # ── Pass 3: height-constrained minimum violation ────────────────────────
        # ONLY at min_height (like greedy) — never sacrifice height balance
        # Among shortest stacks: pick the one with fewest containers earlier than us
        min_h = float("inf")
        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h < bi.tiers:
                        min_h = min(min_h, h)

        if min_h < float("inf"):
            best_unsafe = float("inf")
            for block_name, bi in yard_state.blocks.items():
                for bay in range(1, bi.bays + 1):
                    for row in range(1, bi.rows + 1):
                        h = yard_state.get_stack_height(block_name, bay, row)
                        if h != min_h or h >= bi.tiers:
                            continue
                        key = (block_name, bay, row)
                        # Count containers with ETD earlier than ours (true violations)
                        unsafe = sum(
                            1 for cid in self._stack_containers.get(key, set())
                            if self._container_etd.get(cid, float("inf")) < inc_etd - GRACE
                        )
                        if unsafe < best_unsafe:
                            best_unsafe = unsafe
                            best_pos = Position(block_name, bay, row, h + 1)

        if best_pos is not None:
            self._update_tracking(event.container_id, best_pos, inc_etd)
            return best_pos

        # ── Pass 4: pure greedy last resort ────────────────────────────────────
        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h < bi.tiers:
                        pos = Position(block_name, bay, row, h + 1)
                        self._update_tracking(event.container_id, pos, inc_etd)
                        return pos

        return Position(list(yard_state.blocks.keys())[0], 1, 1, 999)

    def _update_tracking(self, cid: str, pos: Position, etd: float) -> None:
        key = (pos.block, pos.bay, pos.row)
        self._container_etd[cid] = etd
        self._stack_containers.setdefault(key, set()).add(cid)
        self._stack_min_etd[key] = min(self._stack_min_etd.get(key, float("inf")), etd)
