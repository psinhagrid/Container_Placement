"""Bounded Beam Search Placement Strategy.

Academically validated approach (arXiv:2206.12303, 2504.09046):
  For each placement decision, evaluate K candidate positions by simulating
  actual future reshuffles using snapshot()/restore(). Pick the candidate
  that causes the fewest reshuffles in the upcoming retrieval window.

Uses every available API method:
  - vessel_schedule.json: find which vessels load soon → which containers retrieved next
  - get_containers_by_vessel(): get all containers for upcoming vessels accurately
  - snapshot()/restore(): simulate yard state after our placement (explicit lookahead API)
  - get_containers_above(): count exact reshuffles for each upcoming retrieval
  - on_event(): track current simulation time as events progress
  - get_block_occupancy(): spread containers across blocks

Algorithm:
  1. Find all candidate positions at MINIMUM height (never sacrifice height balance)
  2. Get upcoming retrievals: vessels loading within LOOKAHEAD_WINDOW hours
     → use get_containers_by_vessel() for each upcoming vessel
     → sort by retrieval rank (port order + weight class from vessel schedule)
  3. For each candidate (up to MAX_CANDIDATES):
       a. snapshot()
       b. place container at candidate position
       c. Count total get_containers_above() for each upcoming retrieval container
       d. restore()
  4. Return candidate with minimum simulated reshuffles
     (ties broken by existing rank/unsafe precomputation)
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from src.models import Container, Event, Position
from src.placement_interface import PlacementStrategy
from src.yard_state import YardState

WEIGHT_RANK   = {"HEAVY": 3, "MEDIUM": 2, "LIGHT": 1}
WEIGHT_OFFSET = {"HEAVY": 0, "MEDIUM": 1, "LIGHT": 2}
TRUCK_VESSELS = {"VSL019", "VSL020"}
SCHEDULE_PATH = "data/vessel_schedule.json"

MAX_CANDIDATES      = 5    # beam width: candidates to simulate
LOOKAHEAD_HOURS     = 48   # hours ahead to look for upcoming vessel loads
MAX_RETRIEVE_SIM    = 25   # max upcoming containers to simulate reshuffles for
PORT_STEP           = 200.0
WEIGHT_STEP         = 60.0
ONE_HOUR            = 3_600.0


class BeamSearchStrategy(PlacementStrategy):

    # ── Initialization ─────────────────────────────────────────────────────────

    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        self._etd_cache:    Dict[str, float] = {}
        self._current_time: float = 0.0

        # Vessel schedule: vessel_id → list of rotation dicts with parsed timestamps
        self._vessel_schedule: Dict[str, List[dict]] = {}
        self._vessel_ports:    Dict[str, List[str]]  = {}
        self._load_schedule: List[dict] = []  # all (load_start_ts, load_end_ts, vessel_id) sorted

        self._load_vessel_schedule()

        # Stack tracking for rank precomputation
        self._container_rank:  Dict[str, float]       = {}
        self._stack_containers: Dict[tuple, Set[str]] = {}
        self._stack_min_rank:   Dict[tuple, float]    = {}

        for c in initial_state.get("containers", []):
            cid = c["container_id"]
            p   = c["position"]
            key = (p["block"], p["bay"], p["row"])
            rank = self._rank_score(
                c.get("vessel_id", ""),
                c.get("port_of_discharge", ""),
                c.get("weight_class", "MEDIUM"),
                c.get("departure_time", ""),
            )
            self._container_rank[cid] = rank
            self._stack_containers.setdefault(key, set()).add(cid)
            self._stack_min_rank[key] = min(self._stack_min_rank.get(key, float("inf")), rank)

        print(
            f"[BeamSearch] Ready: {len(self._vessel_schedule)} vessels, "
            f"{len(self._load_schedule)} load windows, "
            f"{len(self._container_rank)} initial containers"
        )

    def _load_vessel_schedule(self) -> None:
        if not os.path.exists(SCHEDULE_PATH):
            return
        with open(SCHEDULE_PATH) as f:
            schedule = json.load(f)

        for vessel in schedule.get("vessels", []):
            vid   = vessel["vessel_id"]
            ports = vessel.get("ports", [])
            self._vessel_ports[vid] = ports
            rotations = []
            for rot in vessel.get("rotations", []):
                ls = self._parse_ts(rot.get("load_start", ""))
                le = self._parse_ts(rot.get("load_end", ""))
                if ls > 0 and le > 0:
                    rotations.append({"load_start_ts": ls, "load_end_ts": le,
                                       "etd": rot.get("etd", "")})
                    self._load_schedule.append({"vessel_id": vid, "load_start_ts": ls,
                                                 "load_end_ts": le})
            self._vessel_schedule[vid] = rotations

        self._load_schedule.sort(key=lambda x: x["load_start_ts"])

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _parse_ts(self, s: str) -> float:
        try:
            return datetime.fromisoformat(s).timestamp()
        except Exception:
            return 0.0

    def _etd(self, s: str) -> float:
        if not s:
            return float("inf")
        if s not in self._etd_cache:
            self._etd_cache[s] = self._parse_ts(s) or float("inf")
        return self._etd_cache[s]

    def _rank_score(self, vessel_id: str, port: str, weight: str, departure_time: str) -> float:
        etd = self._etd(departure_time)
        if etd == float("inf") or vessel_id in TRUCK_VESSELS:
            return etd
        ports    = self._vessel_ports.get(vessel_id, [])
        port_idx = ports.index(port) if port in ports else len(ports)
        w_off    = WEIGHT_OFFSET.get(weight, 1)
        return etd + port_idx * PORT_STEP + w_off * WEIGHT_STEP

    # ── Track simulation time ──────────────────────────────────────────────────

    def on_event(self, event: Event) -> None:
        ts = self._parse_ts(event.timestamp)
        if ts > 0:
            self._current_time = ts

    # ── Stack tracking updates ─────────────────────────────────────────────────

    def on_container_retrieved(self, container_id: str, position: Position,
                                reshuffles: int) -> None:
        key = (position.block, position.bay, position.row)
        stack = self._stack_containers.get(key, set())
        stack.discard(container_id)
        self._stack_min_rank[key] = (
            min(self._container_rank.get(c, float("inf")) for c in stack)
            if stack else float("inf")
        )
        self._container_rank.pop(container_id, None)

    # ── Upcoming retrieval prediction ──────────────────────────────────────────

    def _get_upcoming_retrievals(self, yard_state: YardState) -> List[str]:
        """Return container IDs that will be retrieved soon, in retrieval order.

        Uses vessel_schedule.json to find vessels loading within LOOKAHEAD_HOURS.
        Uses get_containers_by_vessel() to get current containers for those vessels.
        Sorts by rank_score so we simulate in the correct retrieval order.
        """
        window_end = self._current_time + LOOKAHEAD_HOURS * ONE_HOUR
        ranked: List[Tuple[float, str]] = []

        seen_vessels: Set[str] = set()
        for entry in self._load_schedule:
            ls = entry["load_start_ts"]
            le = entry["load_end_ts"]
            vid = entry["vessel_id"]
            if vid in seen_vessels:
                continue
            # Vessel loads within our window
            if ls <= window_end and le >= self._current_time - ONE_HOUR:
                seen_vessels.add(vid)
                for cid in yard_state.get_containers_by_vessel(vid):
                    info = yard_state.get_container_info(cid)
                    if info:
                        rank = self._rank_score(
                            info.vessel_id, info.port_of_discharge,
                            info.weight_class, info.departure_time,
                        )
                        ranked.append((rank, cid))

        ranked.sort(key=lambda x: x[0])
        return [cid for _, cid in ranked[:MAX_RETRIEVE_SIM]]

    # ── Beam search core ───────────────────────────────────────────────────────

    def _get_candidates(self, yard_state: YardState) -> List[Position]:
        """All positions at global minimum height — never sacrifice height balance."""
        min_h = float("inf")
        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h < bi.tiers:
                        min_h = min(min_h, h)
                        if min_h == 0:
                            break
                if min_h == 0:
                    break
            if min_h == 0:
                break

        if min_h == float("inf"):
            return []

        # Cache block occupancies once
        block_occ = {}
        for bn, bi in yard_state.blocks.items():
            occ, cap = yard_state.get_block_occupancy(bn)
            block_occ[bn] = occ / cap if cap > 0 else 0.0

        # Collect all min-height positions, sorted by block occupancy
        candidates = []
        for block_name, bi in yard_state.blocks.items():
            occ_ratio = block_occ[block_name]
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h != min_h or h >= bi.tiers:
                        continue
                    candidates.append((occ_ratio, block_name, bay, row))

        # Sort by block occupancy (prefer less-occupied blocks for spread)
        candidates.sort(key=lambda x: x[0])
        return [Position(bn, bay, row, min_h + 1)
                for _, bn, bay, row in candidates[:MAX_CANDIDATES * 3]]

    def _simulate_reshuffles(
        self,
        yard_state: YardState,
        candidate: Position,
        container: Container,
        upcoming: List[str],
    ) -> int:
        """snapshot → place → count actual reshuffles for upcoming → restore."""
        snap = yard_state.snapshot()
        yard_state.place_container(container, candidate)

        total = 0
        for cid in upcoming:
            above = yard_state.get_containers_above(cid)
            total += len(above)

        yard_state.restore(snap)
        return total

    # ── Main placement ─────────────────────────────────────────────────────────

    def place_container(self, yard_state: YardState, event: Event) -> Position:
        inc_rank = self._rank_score(
            event.vessel_id, event.port_of_discharge,
            event.weight_class, event.departure_time,
        )
        container = event.to_container()

        # ── Step 1: get candidate positions at minimum height ──────────────────
        all_candidates = self._get_candidates(yard_state)
        if not all_candidates:
            return self._fallback(yard_state)

        # ── Step 2: find upcoming retrievals via vessel schedule ───────────────
        upcoming = self._get_upcoming_retrievals(yard_state)

        # ── Step 3: pre-filter candidates using rank safety (fast, no snapshot) ─
        # Prefer safe candidates (stack_min_rank >= inc_rank) first
        safe, unsafe = [], []
        for pos in all_candidates:
            key = (pos.block, pos.bay, pos.row)
            if self._stack_min_rank.get(key, float("inf")) >= inc_rank - ONE_HOUR:
                safe.append(pos)
            else:
                unsafe.append(pos)

        # Take up to MAX_CANDIDATES, safe first
        candidates = (safe + unsafe)[:MAX_CANDIDATES]

        # ── Step 4: beam search — simulate actual reshuffles ───────────────────
        if upcoming:
            best_pos      = candidates[0]
            best_reshuffles = float("inf")

            for pos in candidates:
                sim = self._simulate_reshuffles(yard_state, pos, container, upcoming)
                if sim < best_reshuffles:
                    best_reshuffles = sim
                    best_pos = pos
        else:
            # No upcoming retrievals found — fall back to safe-stack preference
            best_pos = safe[0] if safe else candidates[0]

        # ── Step 5: update stack tracking ─────────────────────────────────────
        key = (best_pos.block, best_pos.bay, best_pos.row)
        self._container_rank[event.container_id] = inc_rank
        self._stack_containers.setdefault(key, set()).add(event.container_id)
        self._stack_min_rank[key] = min(
            self._stack_min_rank.get(key, float("inf")), inc_rank
        )

        return best_pos

    def _fallback(self, yard_state: YardState) -> Position:
        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h < bi.tiers:
                        return Position(block_name, bay, row, h + 1)
        return Position(list(yard_state.blocks.keys())[0], 1, 1, 999)
