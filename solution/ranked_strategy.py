"""Ranked Placement Strategy — uses exact retrieval order from vessel schedule.

rank_score for each container:
  = etd_seconds + port_idx * PORT_STEP + weight_offset * WEIGHT_STEP

  port_idx:      position of port_of_discharge in vessel's ports array
                 (0 = first port loaded during LOAD operation)
  weight_offset: HEAVY=0, MEDIUM=1, LIGHT=2
                 (HEAVY is retrieved first within each port group)

Lower rank  → retrieved sooner  → should be ON TOP of stack.
Higher rank → retrieved later   → should be at BOTTOM.

Stack is SAFE for incoming container (rank R):
  stack_min_rank >= R  (every container in the stack is retrieved after R)
  → incoming goes on top of all of them → retrieved before them → 0 reshuffles

Tracking is kept in sync throughout the simulation:
  - initialize():            pre-compute rank_score for all 4800 initial containers
  - place_container():       record rank + update stack_min_rank after choosing pos
  - on_container_retrieved(): remove container + recompute stack_min_rank
                              (as containers depart, more stacks become SAFE)
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

from src.models import Event, Position
from src.placement_interface import PlacementStrategy
from src.yard_state import YardState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEIGHT_RANK: Dict[str, int] = {"HEAVY": 3, "MEDIUM": 2, "LIGHT": 1}
WEIGHT_OFFSET: Dict[str, int] = {"HEAVY": 0, "MEDIUM": 1, "LIGHT": 2}

TRUCK_VESSELS: Set[str] = {"VSL019", "VSL020"}

PORT_STEP: float = 200.0    # rank gap between consecutive port groups
WEIGHT_STEP: float = 60.0   # rank gap between weight classes within a port
ONE_HOUR: float = 3_600.0

# Path to vessel schedule, relative to the repository root
SCHEDULE_PATH: str = "data/vessel_schedule.json"


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

class RankedStrategy(PlacementStrategy):
    """Placement strategy that uses exact retrieval order to eliminate reshuffles.

    Pre-computes a rank_score for every container based on when it will be
    retrieved (earlier = lower score = should sit on top).  A stack is SAFE
    for an incoming container when every container already in the stack has a
    higher rank (retrieved later), so the incoming container will naturally be
    on top when it is eventually retrieved.
    """

    # ------------------------------------------------------------------
    # initialize
    # ------------------------------------------------------------------

    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        """Set up data structures and pre-compute ranks for initial containers.

        Args:
            yard_layout:   The yard layout JSON (block dimensions).
            initial_state: The initial state JSON (pre-existing containers).
        """
        # Cache for ISO-8601 → Unix timestamp conversions
        self._etd_cache: Dict[str, float] = {}

        # vessel_id → ordered list of ports (port[0] = first loaded)
        self._vessel_ports: Dict[str, List[str]] = {}
        self._load_vessel_schedule()

        # container_id → rank_score
        self._container_rank: Dict[str, float] = {}

        # (block, bay, row) → set of container_ids currently in that stack
        self._stack_containers: Dict[Tuple[str, int, int], Set[str]] = {}

        # (block, bay, row) → minimum rank_score in the stack
        # float('inf') means the stack is empty (safe for any container)
        self._stack_min_rank: Dict[Tuple[str, int, int], float] = {}

        # Pre-compute ranks for all containers in the initial state
        for entry in initial_state.get("containers", []):
            cid = entry["container_id"]
            pos = entry["position"]
            key: Tuple[str, int, int] = (pos["block"], pos["bay"], pos["row"])

            rank = self._rank_score(
                vessel_id=entry.get("vessel_id", ""),
                port_of_discharge=entry.get("port_of_discharge", ""),
                weight_class=entry.get("weight_class", "MEDIUM"),
                departure_time=entry.get("departure_time", ""),
            )

            self._container_rank[cid] = rank
            self._stack_containers.setdefault(key, set()).add(cid)

        # Compute stack_min_rank from the populated stack_containers map
        for key, cids in self._stack_containers.items():
            if cids:
                self._stack_min_rank[key] = min(
                    self._container_rank.get(c, float("inf")) for c in cids
                )
            else:
                self._stack_min_rank[key] = float("inf")

    # ------------------------------------------------------------------
    # Vessel schedule loader
    # ------------------------------------------------------------------

    def _load_vessel_schedule(self) -> None:
        """Parse vessel_schedule.json and populate self._vessel_ports."""
        # Support running from repo root or from anywhere by trying both
        # the relative path and a path derived from this file's location.
        candidates = [
            SCHEDULE_PATH,
            os.path.join(os.path.dirname(__file__), "..", SCHEDULE_PATH),
        ]
        schedule_data: Optional[dict] = None
        for path in candidates:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    schedule_data = json.load(fh)
                break
            except FileNotFoundError:
                continue

        if schedule_data is None:
            # Graceful degradation: ranks will still be etd-only (no port/weight sub-ordering)
            return

        for vessel in schedule_data.get("vessels", []):
            vid = vessel.get("vessel_id", "")
            ports = vessel.get("ports", [])
            if vid:
                self._vessel_ports[vid] = ports

    # ------------------------------------------------------------------
    # ETD helper (cached)
    # ------------------------------------------------------------------

    def _etd(self, s: str) -> float:
        """Parse an ISO-8601 datetime string to a Unix timestamp (cached).

        Returns float('inf') for empty / unparseable strings so that
        containers without an ETD sort to the bottom (retrieved last).

        Args:
            s: ISO-8601 datetime string, e.g. "2025-01-05T16:25:49".

        Returns:
            Unix timestamp as a float, or float('inf').
        """
        if not s:
            return float("inf")
        if s not in self._etd_cache:
            try:
                self._etd_cache[s] = datetime.fromisoformat(s).timestamp()
            except (ValueError, TypeError):
                self._etd_cache[s] = float("inf")
        return self._etd_cache[s]

    # ------------------------------------------------------------------
    # Rank score
    # ------------------------------------------------------------------

    def _rank_score(
        self,
        vessel_id: str,
        port_of_discharge: str,
        weight_class: str,
        departure_time: str,
    ) -> float:
        """Compute the retrieval-order rank for a container.

        Lower rank = retrieved sooner = should be on top of its stack.

        For truck vessels (VSL019, VSL020) there is no LOAD event grouping,
        so the rank is simply the ETD timestamp.

        For ship vessels the LOAD event processes ports in schedule order
        and within each port processes HEAVY before MEDIUM before LIGHT,
        so:
            rank = etd + port_idx * PORT_STEP + weight_offset * WEIGHT_STEP

        Args:
            vessel_id:         Vessel identifier (e.g. "VSL001").
            port_of_discharge: Port code (e.g. "PORT_03").
            weight_class:      "HEAVY", "MEDIUM", or "LIGHT".
            departure_time:    ISO-8601 ETD string.

        Returns:
            rank_score as a float.
        """
        etd = self._etd(departure_time)

        # Truck vessels: no intra-vessel sub-ordering
        if vessel_id in TRUCK_VESSELS:
            return etd

        # Ship vessels: account for port order and weight class
        vessel_ports = self._vessel_ports.get(vessel_id, [])

        if port_of_discharge in vessel_ports:
            port_idx = vessel_ports.index(port_of_discharge)
        else:
            # Unknown port: place after all known ports so it sorts to the bottom
            port_idx = len(vessel_ports)

        weight_off = WEIGHT_OFFSET.get(weight_class, 1)

        return etd + port_idx * PORT_STEP + weight_off * WEIGHT_STEP

    # ------------------------------------------------------------------
    # Unsafe count helper
    # ------------------------------------------------------------------

    def _unsafe_rank_count(
        self,
        block: str,
        bay: int,
        row: int,
        inc_rank: float,
    ) -> int:
        """Count containers in a stack that would cause reshuffles.

        A container is "unsafe" relative to incoming rank R if its own rank
        is less than R — meaning it will be retrieved AFTER the incoming
        container but sits below it, so it would need to be reshuffled.

        Equivalently: count containers with rank_score < inc_rank.

        Args:
            block, bay, row: Stack coordinates.
            inc_rank:        Rank of the incoming container.

        Returns:
            Number of containers in the stack with rank_score < inc_rank.
        """
        key = (block, bay, row)
        cids = self._stack_containers.get(key, set())
        return sum(
            1 for c in cids
            if self._container_rank.get(c, float("inf")) < inc_rank
        )

    # ------------------------------------------------------------------
    # Placement passes
    # ------------------------------------------------------------------

    def place_container(self, yard_state: YardState, event: Event) -> Position:
        """Choose the optimal position for an incoming container.

        Four-pass strategy (each pass only runs if the previous found nothing):

        Pass 1 — SAFE non-empty stacks:
            stack_min_rank >= inc_rank AND weight ordering OK.
            Preference: shortest height; tiebreaker same vessel/port on top.

        Pass 2 — Empty stacks:
            Always safe (no conflicts). Prefer less-occupied blocks.

        Pass 3 — Minimum-damage fallback:
            score = unsafe_count + height * 0.1; pick argmin.

        Pass 4 — Pure greedy (last resort):
            Any open slot in the yard.

        After choosing a position the internal tracking dicts are updated.

        Args:
            yard_state: Current yard state (query only).
            event:      Incoming container event.

        Returns:
            Position satisfying all hard constraints.
        """
        inc_rank = self._rank_score(
            vessel_id=event.vessel_id,
            port_of_discharge=event.port_of_discharge,
            weight_class=event.weight_class,
            departure_time=event.departure_time,
        )
        inc_weight_rank = WEIGHT_RANK.get(event.weight_class, 2)
        apply_weight = event.vessel_id not in TRUCK_VESSELS

        chosen: Optional[Position] = None

        # ------------------------------------------------------------------
        # Pass 1: SAFE non-empty stacks (stack_min_rank >= inc_rank)
        # ------------------------------------------------------------------
        best_h: float = float("inf")
        best_vessel_match = False
        best_port_match = False

        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h == 0 or h >= bi.tiers:
                        continue

                    key = (block_name, bay, row)
                    stack_min = self._stack_min_rank.get(key, float("inf"))

                    # SAFE condition: every container retrieved after incoming
                    if stack_min < inc_rank:
                        continue

                    # Weight ordering check (ship vessels only):
                    # incoming container will sit on top → it must be
                    # retrieved no later than the current top container.
                    # inc_weight_rank >= top_weight_rank means incoming is
                    # heavier-or-equal, which is valid (HEAVY retrieved first).
                    if apply_weight:
                        top_cid = yard_state.get_container_at(block_name, bay, row, h)
                        if top_cid:
                            top_info = yard_state.get_container_info(top_cid)
                            if top_info:
                                top_weight_rank = WEIGHT_RANK.get(
                                    top_info.weight_class, 2
                                )
                                if inc_weight_rank < top_weight_rank:
                                    # Lighter incoming on top of heavier → skip
                                    continue

                    # Tiebreaker: vessel and port match on top container
                    top_cid = yard_state.get_container_at(block_name, bay, row, h)
                    top_info = (
                        yard_state.get_container_info(top_cid) if top_cid else None
                    )
                    vessel_match = bool(
                        top_info and top_info.vessel_id == event.vessel_id
                    )
                    port_match = bool(
                        top_info
                        and top_info.port_of_discharge == event.port_of_discharge
                    )

                    # Select: shorter height wins; break ties by group affinity
                    better = False
                    if h < best_h:
                        better = True
                    elif h == best_h:
                        if vessel_match and not best_vessel_match:
                            better = True
                        elif vessel_match == best_vessel_match:
                            if port_match and not best_port_match:
                                better = True

                    if better:
                        best_h = h
                        best_vessel_match = vessel_match
                        best_port_match = port_match
                        chosen = Position(block_name, bay, row, h + 1)

        if chosen:
            self._record_placement(chosen, event.container_id, inc_rank)
            return chosen

        # ------------------------------------------------------------------
        # Pass 2: Empty stacks (trivially safe, height == 0)
        # Prefer blocks with lower occupancy ratio to spread containers evenly.
        # ------------------------------------------------------------------
        best_occ: float = float("inf")

        for block_name, bi in yard_state.blocks.items():
            occ, cap = yard_state.get_block_occupancy(block_name)
            occ_ratio = occ / cap if cap > 0 else 0.0

            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    if yard_state.get_stack_height(block_name, bay, row) == 0:
                        if occ_ratio < best_occ:
                            best_occ = occ_ratio
                            chosen = Position(block_name, bay, row, 1)

        if chosen:
            self._record_placement(chosen, event.container_id, inc_rank)
            return chosen

        # ------------------------------------------------------------------
        # Pass 3: Minimum-damage fallback (forced violation)
        # score = unsafe_count + height * 0.1
        # Minimises the number of future reshuffles caused by this placement.
        # ------------------------------------------------------------------
        best_score: float = float("inf")

        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h == 0 or h >= bi.tiers:
                        continue

                    unsafe = self._unsafe_rank_count(block_name, bay, row, inc_rank)
                    score = unsafe + h * 0.1

                    if score < best_score:
                        best_score = score
                        chosen = Position(block_name, bay, row, h + 1)

        if chosen:
            self._record_placement(chosen, event.container_id, inc_rank)
            return chosen

        # ------------------------------------------------------------------
        # Pass 4: Pure greedy — absolute last resort (yard nearly full)
        # ------------------------------------------------------------------
        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h < bi.tiers:
                        pos = Position(block_name, bay, row, h + 1)
                        self._record_placement(pos, event.container_id, inc_rank)
                        return pos

        # Fallback sentinel (should never be reached in a valid yard)
        fallback = Position(list(yard_state.blocks.keys())[0], 1, 1, 999)
        self._record_placement(fallback, event.container_id, inc_rank)
        return fallback

    # ------------------------------------------------------------------
    # Tracking helpers
    # ------------------------------------------------------------------

    def _record_placement(
        self, pos: Position, container_id: str, rank: float
    ) -> None:
        """Update internal tracking after placing a container.

        Args:
            pos:          The chosen placement position.
            container_id: The container being placed.
            rank:         Its pre-computed rank_score.
        """
        key = (pos.block, pos.bay, pos.row)
        self._container_rank[container_id] = rank
        self._stack_containers.setdefault(key, set()).add(container_id)
        old_min = self._stack_min_rank.get(key, float("inf"))
        self._stack_min_rank[key] = min(old_min, rank)

    def on_container_retrieved(
        self,
        container_id: str,
        position: Position,
        reshuffles: int,
    ) -> None:
        """Update tracking after a container is retrieved from the yard.

        As containers leave, stack_min_rank may rise, making more stacks
        eligible as SAFE targets for future placements.

        Args:
            container_id: The container that was retrieved.
            position:     Where it was stored in the yard.
            reshuffles:   Number of reshuffles that were required (informational).
        """
        key = (position.block, position.bay, position.row)
        stack = self._stack_containers.get(key, set())
        stack.discard(container_id)

        # Recompute min_rank from the remaining containers
        if not stack:
            self._stack_min_rank[key] = float("inf")
        else:
            self._stack_min_rank[key] = min(
                self._container_rank.get(c, float("inf")) for c in stack
            )

        # Remove from rank lookup — container is no longer in the yard
        self._container_rank.pop(container_id, None)
