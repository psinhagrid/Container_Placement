"""Single source of truth for all feature definitions.

To add or remove a feature: edit ONLY this file.
Every collector, strategy, trainer, and loop imports from here.
No more 1.1157 bugs from stale hardcoded feature lists.
"""

FEATURES = [
    # Core
    "stack_height",
    "top_etd_gap_days",
    # Vessel / port / weight grouping
    "same_vessel",
    "same_port",
    "weight_ok",
    "weight_rank_inc",
    "weight_rank_top",
    # Block and time context
    "block_occ",
    "days_until_dep",
    "unsafe_count",
    # Retrieval rank (port order + weight within vessel)
    "intra_vessel_rank",
    # Stack conflict
    "unsafe_rank_count",
    "rank_gap_to_top",
    # Interaction
    "unsafe_x_height",
    # Yard context
    "min_height_pct",
    # Timing and grouping (new)
    "hours_until_load",
    "same_group_in_stack",
    "initial_below_count",
]

TARGET      = "reshuffles"
FEATURE_COLS = FEATURES + [TARGET]   # full CSV column list including label
