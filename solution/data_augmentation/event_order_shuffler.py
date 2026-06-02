"""Event Order Shuffler — creates diverse training scenarios by varying discharge order.

Shuffles which container arrives first within each vessel rotation's discharge window.
Same containers are discharged and loaded, just in a different sequence.

Why this works:
  - Different weight classes / ports arrive at different times → different stack composition
  - Model can't memorize "HEAVY for VSL001 always arrives first"
  - Must learn general timing and grouping principles (universal)

Valid because:
  - All container_ids still present in yard
  - LOAD events find containers by container_id → still works
  - vessel_id and departure_time unchanged within each rotation group
"""

import copy
from random import Random
from typing import List

TRUCK_VESSELS = {"VSL019", "VSL020"}


def shuffle_discharge_order(events, seed=None):
    """Shuffle container attributes within each vessel rotation's DISCHARGE events.

    Args:
        events: list of Event objects from the simulator
        seed: random seed for reproducibility

    Returns:
        New list of Event objects with shuffled discharge order
    """
    rng = Random(seed)

    # Group DISCHARGE event INDICES by vessel rotation
    vessel_discharge_indices = {}  # (vessel_id, departure_time) -> [indices]
    for i, e in enumerate(events):
        if e.type == 'DISCHARGE' and e.vessel_id not in TRUCK_VESSELS:
            key = (e.vessel_id, e.departure_time)
            if key not in vessel_discharge_indices:
                vessel_discharge_indices[key] = []
            vessel_discharge_indices[key].append(i)

    # Make a shallow copy of the events list (we'll replace specific events)
    events_copy = list(events)

    for key, indices in vessel_discharge_indices.items():
        if len(indices) <= 1:
            continue

        # Extract the "container identity" from each discharge event
        # (these are the attributes that vary between containers of the same vessel)
        container_attrs = [
            (events[i].container_id, events[i].weight_class, events[i].port_of_discharge)
            for i in indices
        ]

        # Shuffle the container assignment across the time slots
        rng.shuffle(container_attrs)

        # Reassign: keep timestamps/event_ids/vessel/departure_time, swap container attrs
        for i, (idx, (cid, wc, pod)) in enumerate(zip(indices, container_attrs)):
            orig = events[idx]
            new_e = copy.copy(orig)
            new_e.container_id    = cid
            new_e.weight_class    = wc
            new_e.port_of_discharge = pod
            # vessel_id, departure_time, timestamp, event_id, size all stay same
            events_copy[idx] = new_e

    return events_copy
