#!/usr/bin/env python3
"""Audit rank-local HCCL trace payloads in the frozen target graph.

The trace Size(Byte) field is a reported operation data size, not physical
link transit bytes. All windows must pass exact operation-count gates.
"""
import argparse
import collections
import json
import statistics
from pathlib import Path

EXPECTED = {"allgatherAivKernel": 135, "reduce_scatterAivKernel": 87,
            "alltoallAivKernel": 43}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    windows = []
    invalid = []
    for rank in range(8):
        folders = sorted(p for p in args.profile_dir.glob("rank%d_*" % rank)
                         if p.is_dir())
        if len(folders) != 5:
            invalid.append({"rank": rank, "reason": "capture_count",
                            "actual": len(folders)})
            continue
        for capture, folder in enumerate(folders):
            trace_path = folder / "ASCEND_PROFILER_OUTPUT/trace_view.json"
            comm_path = folder / "ASCEND_PROFILER_OUTPUT/communication.json"
            events = json.loads(trace_path.read_text())
            scopes = sorted((e for e in events if e.get("name") == "extreme::target"
                             and e.get("cat") == "cpu_op"),
                            key=lambda e: float(e["ts"]))
            if len(scopes) != 2:
                invalid.append({"rank": rank, "capture": capture,
                                "reason": "target_scope_count",
                                "actual": len(scopes)})
                continue
            comm = json.loads(comm_path.read_text())
            collective = comm.get("step", {}).get("collective", {})
            transit = []
            for entry in collective.values():
                bw = entry.get("Communication Bandwidth Info", {})
                for link in bw.values():
                    if isinstance(link, dict):
                        transit.append(link.get("Transit Size(MB)", None))
            for cycle, scope in enumerate(scopes):
                start = float(scope["ts"])
                stop = start + float(scope["dur"])
                selected = [e for e in events if e.get("ph") == "X"
                            and isinstance(e.get("args"), dict)
                            and "size(Byte)" in e["args"]
                            and start <= float(e["ts"]) < stop]
                names = collections.Counter(e["name"] for e in selected)
                if names != EXPECTED:
                    invalid.append({"rank": rank, "capture": capture,
                                    "cycle": cycle, "reason": "family_count",
                                    "actual": dict(names)})
                    continue
                pairs = collections.Counter((e["name"], int(e["args"]["size(Byte)"]))
                                            for e in selected)
                size_sum = sum(size * n for (_, size), n in pairs.items())
                windows.append({
                    "rank": rank, "capture": capture, "cycle": cycle,
                    "event_count": len(selected), "reported_payload_bytes": size_sum,
                    "name_size_counts": [
                        {"name": name, "size_bytes": size, "count": count}
                        for (name, size), count in sorted(pairs.items())
                    ],
                    "transport_types": dict(collections.Counter(
                        str(e["args"].get("transport type")) for e in selected)),
                    "link_types": dict(collections.Counter(
                        str(e["args"].get("link type")) for e in selected)),
                    "communication_transit_sizes_nonzero": sum(
                        x not in (0, 0.0, None) for x in transit),
                })
    latest = [w for w in windows if w["capture"] == 4]
    signatures = {json.dumps(w["name_size_counts"], sort_keys=True) for w in latest}
    status = "valid" if (not invalid and len(windows) == 80 and
                         len(latest) == 16 and len(signatures) == 1) else "invalid"
    latest_payload = [w["reported_payload_bytes"] for w in latest]
    result = {
        "status": status, "all_valid_windows": len(windows),
        "latest_valid_windows": len(latest),
        "latest_ranks": sorted({w["rank"] for w in latest}),
        "latest_reported_payload_bytes_per_rank_cycle":
            statistics.median(latest_payload) if latest_payload else None,
        "latest_min_max_bytes": [min(latest_payload), max(latest_payload)]
            if latest_payload else [],
        "latest_name_size_counts": latest[0]["name_size_counts"] if latest else [],
        "invalid": invalid, "windows": windows,
        "meaning": "Profiler HCCL task Size(Byte) is reported operation data size; "
                   "not measured network transit, rank-unique logical bytes, or "
                   "a lower bound on wall time.",
        "limits": [
            "All recorded transport and link types are INVALID_TYPE if present.",
            "communication.json transit-size fields are zero on these captures.",
            "Cannot determine whether all-gather size is per-rank send or output "
            "without a matched source/API probe.",
            "Physical topology, algorithm steps, peer arrival, and overlap remain unknown."
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in (
        "status", "all_valid_windows", "latest_valid_windows",
        "latest_reported_payload_bytes_per_rank_cycle",
        "latest_name_size_counts", "invalid")}, indent=2))
    if status != "valid":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
