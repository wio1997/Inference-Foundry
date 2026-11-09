"""Map official YOCO persistent SWA stores to 910C paged writes."""

import torch


def prepare_store(plane, positions, request_slots, official_slots):
    """Keep original query order and treat the official -1 slot as no store."""
    if len(positions) != len(request_slots) or len(positions) != len(official_slots):
        raise ValueError("YOCO store metadata must cover the same query rows")
    if any(type(slot) is not int or slot < -1 for slot in official_slots):
        raise ValueError("invalid official YOCO store slot")

    rows = [index for index, slot in enumerate(official_slots) if slot != -1]
    if not rows:
        return None
    plan = plane.plan([positions[index] for index in rows],
                      [request_slots[index] for index in rows])
    return torch.tensor(rows, dtype=torch.int64, device=plane.cache.device), plan


def store(plane, values, prepared):
    """Commit only persistent stores. The caller owns the temporary attention view."""
    if prepared is not None:
        rows, plan = prepared
        plane.write(values.index_select(0, rows), plan)
