"""Greedy strategy with enriched training data collection.

Placement: identical to baseline_greedy (always pick shortest stack).
Collection: records 21 features at placement time, labels at retrieval time.

New features added (based on domain analysis):
  port_order        — exact position in vessel's loading sequence (0=first)
  weight_offset     — 0=HEAVY(loaded first), 2=LIGHT(loaded last)
  intra_vessel_rank — port_order*3 + weight_offset (0-17, full loading position)
  unsafe_rank_count — containers in stack with intra_rank < ours (precise conflict count)
  free_slots        — 5-height (fewer = more pile-on risk)
  rank_gap_to_top   — our intra_rank vs top container's (interaction)
  unsafe_x_height   — unsafe_count * height (interaction: damage amplified by height)
  min_height_pct    — yard context: % of stacks at minimum height
"""

import atexit
import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Set

from src.models import Event, Position
from src.placement_interface import PlacementStrategy
from src.yard_state import YardState

WEIGHT_RANK   = {"HEAVY": 3, "MEDIUM": 2, "LIGHT": 1}
WEIGHT_OFFSET = {"HEAVY": 0, "MEDIUM": 1, "LIGHT": 2}
TRUCK_VESSELS = {"VSL019", "VSL020"}
TRAINING_DATA_PATH = "data/train/placement_features.csv"
SCHEDULE_PATH = "data/vessel_schedule.json"
ONE_HOUR = 3_600.0

FEATURE_COLS = [
    "stack_height", "top_etd_gap_days", "same_vessel", "same_port",
    "weight_ok", "weight_rank_inc", "weight_rank_top",
    "block_occ", "days_until_dep", "unsafe_count",
    "intra_vessel_rank",
    "unsafe_rank_count", "rank_gap_to_top",
    "unsafe_x_height", "min_height_pct",
    "hours_until_load", "same_group_in_stack", "initial_below_count",
    "reshuffles",
]


class GreedyCollector(PlacementStrategy):

    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        self._etd_cache: Dict[str, float] = {}
        self._placement_features: Dict[str, dict] = {}
        self._training_rows: List[dict] = []

        # Load vessel schedule for port order features
        self._vessel_ports: Dict[str, List[str]] = {}
        if os.path.exists(SCHEDULE_PATH):
            with open(SCHEDULE_PATH) as f:
                sched = json.load(f)
            for v in sched.get("vessels", []):
                self._vessel_ports[v["vessel_id"]] = v.get("ports", [])

        # Build vessel rotation load windows
        self._vessel_rotations: dict = {}
        if os.path.exists(SCHEDULE_PATH):
            with open(SCHEDULE_PATH) as f:
                sched2 = json.load(f)
            for v in sched2.get("vessels", []):
                vid = v["vessel_id"]
                rots = []
                for rot in v.get("rotations", []):
                    try:
                        ls = datetime.fromisoformat(rot.get("load_start", "")).timestamp()
                        etd_ts = datetime.fromisoformat(rot.get("etd", "")).timestamp()
                        if ls > 0:
                            rots.append({"etd_ts": etd_ts, "load_start_ts": ls})
                    except Exception:
                        pass
                self._vessel_rotations[vid] = rots

        # Track initial container IDs
        self._initial_cids: set = {c["container_id"] for c in initial_state.get("containers", [])}
        self._current_time: float = 0.0

        # Precompute stack tracking
        self._container_etd: Dict[str, float] = {}
        self._container_intra_rank: Dict[str, int] = {}
        self._stack_containers: Dict[tuple, Set[str]] = {}

        for c in initial_state.get("containers", []):
            cid = c["container_id"]
            p   = c["position"]
            key = (p["block"], p["bay"], p["row"])
            etd = self._etd(c.get("departure_time", ""))
            ir  = self._intra_rank(c.get("vessel_id", ""),
                                   c.get("port_of_discharge", ""),
                                   c.get("weight_class", "MEDIUM"))
            self._container_etd[cid] = etd
            self._container_intra_rank[cid] = ir
            self._stack_containers.setdefault(key, set()).add(cid)

        atexit.register(self._save_training_data)

    def on_event(self, event) -> None:
        try:
            ts = datetime.fromisoformat(event.timestamp).timestamp()
            if ts > 0:
                self._current_time = ts
        except Exception:
            pass

    def _etd(self, s: str) -> float:
        if not s:
            return float("inf")
        if s not in self._etd_cache:
            try:
                self._etd_cache[s] = datetime.fromisoformat(s).timestamp()
            except Exception:
                self._etd_cache[s] = float("inf")
        return self._etd_cache[s]

    def _intra_rank(self, vessel_id: str, port: str, weight: str) -> int:
        """0-17: position within vessel's loading sequence."""
        if vessel_id in TRUCK_VESSELS:
            return 0
        ports    = self._vessel_ports.get(vessel_id, [])
        port_idx = ports.index(port) if port in ports else len(ports)
        w_off    = WEIGHT_OFFSET.get(weight, 1)
        return port_idx * 3 + w_off

    def _hours_until_load(self, vessel_id: str, etd_ts: float) -> float:
        if vessel_id in TRUCK_VESSELS or self._current_time == 0:
            return 240.0
        best = 240.0
        for rot in self._vessel_rotations.get(vessel_id, []):
            if abs(rot["etd_ts"] - etd_ts) < 3600:
                hours = max(0.0, (rot["load_start_ts"] - self._current_time) / 3600)
                best = min(best, hours)
        return round(best, 2)

    def _same_group_in_stack(self, block: str, bay: int, row: int,
                              inc_etd: float, inc_ir: int) -> int:
        key = (block, bay, row)
        return sum(
            1 for cid in self._stack_containers.get(key, set())
            if abs(self._container_etd.get(cid, float("inf")) - inc_etd) < 3600.0
            and self._container_intra_rank.get(cid, -1) == inc_ir
        )

    def _initial_below_count(self, block: str, bay: int, row: int, inc_etd: float) -> int:
        key = (block, bay, row)
        return sum(
            1 for cid in self._stack_containers.get(key, set())
            if cid in self._initial_cids
            and self._container_etd.get(cid, float("inf")) < inc_etd - 3600.0
        )

    def place_container(self, yard_state: YardState, event: Event) -> Position:
        inc_etd      = self._etd(event.departure_time)
        inc_rank     = WEIGHT_RANK.get(event.weight_class, 2)
        placement_ts = self._etd(event.timestamp)
        is_truck     = int(event.vessel_id in TRUCK_VESSELS)
        inc_port_ord = 0 if is_truck else (
            self._vessel_ports.get(event.vessel_id, []).index(event.port_of_discharge)
            if event.port_of_discharge in self._vessel_ports.get(event.vessel_id, [])
            else len(self._vessel_ports.get(event.vessel_id, []))
        )
        inc_w_off    = WEIGHT_OFFSET.get(event.weight_class, 1)
        inc_ir       = inc_port_ord * 3 + inc_w_off

        # Find globally shortest stack (greedy)
        best_pos:    Optional[Position] = None
        best_height  = float("inf")
        chosen_h     = 0
        chosen_top_etd: float = float("inf")
        chosen_top_rank = 0
        chosen_top_ir   = 0
        chosen_vessel   = False
        chosen_port     = False
        chosen_weight   = True
        chosen_block: Optional[str] = None

        # Count min-height stacks for yard context
        min_h_global = float("inf")
        for bn, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(bn, bay, row)
                    if h < bi.tiers:
                        min_h_global = min(min_h_global, h)

        min_h_count = 0
        total_open  = 0
        for bn, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(bn, bay, row)
                    if h < bi.tiers:
                        total_open += 1
                        if h == min_h_global:
                            min_h_count += 1

        for block_name, bi in yard_state.blocks.items():
            for bay in range(1, bi.bays + 1):
                for row in range(1, bi.rows + 1):
                    h = yard_state.get_stack_height(block_name, bay, row)
                    if h < bi.tiers and h < best_height:
                        best_height     = h
                        best_pos        = Position(block_name, bay, row, h + 1)
                        chosen_h        = h
                        chosen_block    = block_name
                        chosen_top_etd  = float("inf")
                        chosen_top_rank = 0
                        chosen_top_ir   = 0
                        chosen_vessel   = False
                        chosen_port     = False
                        chosen_weight   = True

                        if h > 0:
                            top_cid  = yard_state.get_container_at(block_name, bay, row, h)
                            top_info = yard_state.get_container_info(top_cid) if top_cid else None
                            if top_info:
                                chosen_top_etd  = self._etd(top_info.departure_time)
                                chosen_top_rank = WEIGHT_RANK.get(top_info.weight_class, 2)
                                chosen_top_ir   = self._container_intra_rank.get(top_cid, 0)
                                chosen_vessel   = top_info.vessel_id == event.vessel_id
                                chosen_port     = top_info.port_of_discharge == event.port_of_discharge
                                if not is_truck:
                                    chosen_weight = inc_rank >= chosen_top_rank

                        if best_height == 0:
                            break
                if best_height == 0:
                    break
            if best_height == 0:
                break

        if best_pos is not None and chosen_block is not None:
            occ, cap  = yard_state.get_block_occupancy(chosen_block)
            block_occ = occ / cap if cap > 0 else 0.0

            top_etd_gap = (
                (chosen_top_etd - inc_etd) / 86_400
                if chosen_top_etd != float("inf") and inc_etd != float("inf")
                else 0.0
            )
            days_until = (
                max(0.0, (inc_etd - placement_ts) / 86_400)
                if inc_etd != float("inf") and placement_ts != float("inf")
                else 0.0
            )

            key = (chosen_block, best_pos.bay, best_pos.row)

            # Stack conflict features
            unsafe_count      = sum(
                1 for cid in self._stack_containers.get(key, set())
                if self._container_etd.get(cid, float("inf")) < inc_etd - ONE_HOUR
            )
            unsafe_rank_count = sum(
                1 for cid in self._stack_containers.get(key, set())
                if self._container_intra_rank.get(cid, 0) < inc_ir
                and self._container_etd.get(cid, float("inf")) < inc_etd + ONE_HOUR
            )
            rank_gap_to_top   = round(inc_ir - chosen_top_ir, 4)
            min_height_pct    = round(min_h_count / max(total_open, 1), 4)

            self._placement_features[event.container_id] = {
                "stack_height":       chosen_h,
                "top_etd_gap_days":   round(top_etd_gap, 4),
                "same_vessel":        int(chosen_vessel),
                "same_port":          int(chosen_port),
                "weight_ok":          int(chosen_weight),
                "weight_rank_inc":    inc_rank,
                "weight_rank_top":    chosen_top_rank,
                "block_occ":          round(block_occ, 4),
                "days_until_dep":     round(days_until, 4),
                "unsafe_count":       unsafe_count,
                "intra_vessel_rank":  inc_ir,
                "unsafe_rank_count":  unsafe_rank_count,
                "rank_gap_to_top":    rank_gap_to_top,
                "unsafe_x_height":    unsafe_count * chosen_h,
                "min_height_pct":     min_height_pct,
                "hours_until_load":    self._hours_until_load(event.vessel_id, inc_etd),
                "same_group_in_stack": self._same_group_in_stack(chosen_block, best_pos.bay, best_pos.row, inc_etd, inc_ir),
                "initial_below_count": self._initial_below_count(chosen_block, best_pos.bay, best_pos.row, inc_etd),
            }

            # Update tracking
            self._container_etd[event.container_id]         = inc_etd
            self._container_intra_rank[event.container_id]  = inc_ir
            self._stack_containers.setdefault(key, set()).add(event.container_id)

            return best_pos

        return Position(list(yard_state.blocks.keys())[0], 1, 1, 999)

    def on_container_retrieved(self, container_id: str, position: Position,
                                reshuffles: int) -> None:
        feats = self._placement_features.pop(container_id, None)
        if feats is not None:
            row = dict(feats)
            row["reshuffles"] = reshuffles
            self._training_rows.append(row)

        key = (position.block, position.bay, position.row)
        self._stack_containers.get(key, set()).discard(container_id)
        self._container_etd.pop(container_id, None)
        self._container_intra_rank.pop(container_id, None)

    def _save(self) -> None:
        self._save_training_data()

    def _save_training_data(self) -> None:
        if not self._training_rows:
            return
        os.makedirs(os.path.dirname(TRAINING_DATA_PATH), exist_ok=True)
        with open(TRAINING_DATA_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FEATURE_COLS)
            writer.writeheader()
            writer.writerows(self._training_rows)
        print(f"\n[XGBoost data] {len(self._training_rows)} rows → {TRAINING_DATA_PATH}")
