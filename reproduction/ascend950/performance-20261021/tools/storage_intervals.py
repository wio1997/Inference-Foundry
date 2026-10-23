#!/usr/bin/env python3
"""Priority-aware accounting for half-open storage allocation intervals."""

from __future__ import annotations

import argparse
import heapq
import json
import re
import sys
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = ["IntervalAccountingError", "partition_intervals"]

_HEX_ADDRESS = re.compile(r"(?:0[xX])?[0-9A-Fa-f]+\Z")
_REQUIRED_RECORD_KEYS = ("label", "category", "address", "size_bytes")


class IntervalAccountingError(ValueError):
    """Raised when records or category priorities are invalid."""


def _parse_address(value: Any, where: str) -> int:
    if isinstance(value, bool):
        raise IntervalAccountingError(f"{where}: address must be an integer or hexadecimal string")
    if isinstance(value, int):
        address = value
    elif isinstance(value, str):
        if _HEX_ADDRESS.fullmatch(value) is None:
            raise IntervalAccountingError(
                f"{where}: address string must be an unsigned hexadecimal integer, optionally 0x-prefixed"
            )
        address = int(value, 16)
    else:
        raise IntervalAccountingError(f"{where}: address must be an integer or hexadecimal string")
    if address < 0:
        raise IntervalAccountingError(f"{where}: address must not be negative")
    return address


def _parse_size(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise IntervalAccountingError(f"{where}: size_bytes must be a nonnegative integer (floats are not coerced)")
    if value < 0:
        raise IntervalAccountingError(f"{where}: size_bytes must not be negative")
    return value


def _validate_category_order(category_order: Any) -> tuple[list[str], dict[str, int]]:
    if isinstance(category_order, (str, bytes, Mapping)) or not isinstance(
        category_order, (list, tuple)
    ):
        raise IntervalAccountingError("category_order must be a list of category names")
    order: list[str] = []
    priority: dict[str, int] = {}
    for index, category in enumerate(category_order):
        if not isinstance(category, str):
            raise IntervalAccountingError(f"category_order[{index}]: category must be a string")
        if category in priority:
            raise IntervalAccountingError(f"category_order contains duplicate category {category!r}")
        priority[category] = index
        order.append(category)
    return order, priority


def _materialize_records(records: Any) -> list[Any]:
    if isinstance(records, (str, bytes, Mapping)):
        raise IntervalAccountingError("records must be an iterable of record objects")
    if not isinstance(records, Iterable):
        raise IntervalAccountingError("records must be an iterable of record objects")
    try:
        return list(records)
    except TypeError as exc:
        raise IntervalAccountingError("records must be an iterable of record objects") from exc


def _parse_records(
    records: Iterable[Any], priority: dict[str, int]
) -> tuple[list[tuple[str, str, int, int]], int]:
    parsed: list[tuple[str, str, int, int]] = []
    input_bytes = 0
    for index, record in enumerate(_materialize_records(records)):
        where = f"records[{index}]"
        if not isinstance(record, Mapping):
            raise IntervalAccountingError(f"{where}: record must be an object")
        missing = [key for key in _REQUIRED_RECORD_KEYS if key not in record]
        if missing:
            raise IntervalAccountingError(f"{where}: missing required field(s): {', '.join(missing)}")

        label = record["label"]
        category = record["category"]
        if not isinstance(label, str):
            raise IntervalAccountingError(f"{where}: label must be a string")
        if not isinstance(category, str):
            raise IntervalAccountingError(f"{where}: category must be a string")
        if category not in priority:
            raise IntervalAccountingError(f"{where}: unknown category {category!r}")

        start = _parse_address(record["address"], where)
        size = _parse_size(record["size_bytes"], where)
        input_bytes += size
        if size > 0:
            parsed.append((label, category, start, start + size))
    return parsed, input_bytes


def _union_bytes(intervals: list[tuple[str, str, int, int]]) -> int:
    ordered = sorted((start, end) for _, _, start, end in intervals)
    if not ordered:
        return 0
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        elif end > current_end:
            current_end = end
    return total + current_end - current_start


def _bytes_by_category(
    intervals: list[tuple[str, str, int, int]],
    category_order: list[str],
    priority: dict[str, int],
) -> dict[str, int]:
    unique_spans: dict[str, set[tuple[int, int]]] = {category: set() for category in category_order}
    for _, category, start, end in intervals:
        unique_spans[category].add((start, end))

    events: list[tuple[int, str, int]] = []
    for category, spans in unique_spans.items():
        for start, end in spans:
            events.append((start, category, 1))
            events.append((end, category, -1))
    events.sort(key=lambda event: event[0])

    counts = {category: 0 for category in category_order}
    active_heap: list[tuple[int, str]] = []
    accrued = {category: 0 for category in category_order}
    index = 0
    event_count = len(events)
    while index < event_count:
        coordinate = events[index][0]
        while index < event_count and events[index][0] == coordinate:
            _, category, delta = events[index]
            old_count = counts[category]
            new_count = old_count + delta
            if new_count < 0:
                raise RuntimeError("internal interval sweep lost category coverage")
            counts[category] = new_count
            if old_count == 0 and new_count > 0:
                heapq.heappush(active_heap, (priority[category], category))
            index += 1

        while active_heap and counts[active_heap[0][1]] == 0:
            heapq.heappop(active_heap)

        if index < event_count:
            next_coordinate = events[index][0]
            if active_heap and next_coordinate > coordinate:
                owner = active_heap[0][1]
                accrued[owner] += next_coordinate - coordinate
    return accrued


def _exact_alias_groups(intervals: list[tuple[str, str, int, int]]) -> list[list[str]]:
    labels_by_span: dict[tuple[int, int], list[str]] = {}
    for label, _, start, end in intervals:
        labels_by_span.setdefault((start, end), []).append(label)
    groups = [sorted(labels) for labels in labels_by_span.values() if len(labels) >= 2]
    groups.sort()
    return groups


def _partial_overlap_pairs(intervals: list[tuple[str, str, int, int]]) -> list[list[str]]:
    ordered = sorted(intervals, key=lambda item: (item[2], item[3], item[0], item[1]))
    active: list[tuple[int, int, str, str]] = []
    pairs: set[tuple[str, str]] = set()
    for label, category, start, end in ordered:
        still_active: list[tuple[int, int, str, str]] = []
        for active_end, active_start, active_label, active_category in active:
            if active_end <= start:
                continue
            still_active.append((active_end, active_start, active_label, active_category))
            if active_start != start or active_end != end:
                pairs.add(tuple(sorted((label, active_label))))
        active = still_active
        active.append((end, start, label, category))
    return [list(pair) for pair in sorted(pairs)]


def partition_intervals(records: Iterable[Mapping[str, Any]], category_order: list[str]) -> dict[str, Any]:
    """Account for nonempty half-open intervals by category priority.

    Intervals are whole-allocation spans ``[address, address + size_bytes)``.
    Exact aliases are grouped by identical nonempty address and size, duplicate
    spans within one category count once for category coverage, and
    ``partial_overlap_pairs`` contains unordered label pairs for non-identical
    intervals that intersect. Input bytes include every record, including
    aliases and duplicates. ``bytes_by_category`` covers the union once, with
    the earliest category in ``category_order`` winning each shared byte.

    Overlap priority does not prove ownership. It is a deterministic accounting
    rule, and interval union is virtual allocation accounting, not physical
    memory-bank placement.
    """
    order, priority = _validate_category_order(category_order)
    intervals, input_bytes = _parse_records(records, priority)
    union_bytes = _union_bytes(intervals)
    bytes_by_category = _bytes_by_category(intervals, order, priority)
    attributed = sum(bytes_by_category.values())
    if attributed != union_bytes:
        raise RuntimeError(
            f"internal accounting mismatch: category bytes {attributed} != union {union_bytes}"
        )
    return {
        "union_bytes": union_bytes,
        "input_bytes": input_bytes,
        "alias_bytes": input_bytes - union_bytes,
        "bytes_by_category": bytes_by_category,
        "exact_alias_groups": _exact_alias_groups(intervals),
        "partial_overlap_pairs": _partial_overlap_pairs(intervals),
    }


def _reject_json_float(value: str) -> None:
    raise IntervalAccountingError(f"floating-point JSON number {value!r} is not allowed")


def _reject_json_constant(value: str) -> None:
    raise IntervalAccountingError(f"non-finite JSON value {value!r} is not allowed")


def _read_input(path: str) -> tuple[list[Any], list[str]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(
            handle,
            parse_float=_reject_json_float,
            parse_constant=_reject_json_constant,
        )
    if not isinstance(payload, Mapping):
        raise IntervalAccountingError("input JSON must be an object")
    if "records" not in payload or "category_order" not in payload:
        raise IntervalAccountingError("input JSON must contain records and category_order")
    records = payload["records"]
    category_order = payload["category_order"]
    if not isinstance(records, list):
        raise IntervalAccountingError("input field records must be an array")
    if not isinstance(category_order, list):
        raise IntervalAccountingError("input field category_order must be an array")
    return records, category_order


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Account for storage allocation intervals by category priority."
    )
    parser.add_argument("--input", required=True, metavar="JSON", help="input JSON file")
    parser.add_argument("--out", required=True, metavar="JSON", help="output JSON file")
    args = parser.parse_args(argv)

    try:
        records, category_order = _read_input(args.input)
        result = partition_intervals(records, category_order)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
