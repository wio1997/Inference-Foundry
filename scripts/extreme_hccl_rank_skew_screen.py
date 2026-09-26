#!/usr/bin/env python3
"""Diagnose first collective wait and cross-rank skew in synchronized profile windows.

This is not HCCL wire bandwidth or a product communication lower bound.
"""
import argparse
import glob
import json
import statistics
from pathlib import Path


def screen(root: Path) -> dict:
    rows = []
    invalid = []
    for rank in range(8):
        files = sorted(glob.glob(str(root / f"rank{rank}_*" / "ASCEND_PROFILER_OUTPUT" / "trace_view.json")))
        if not files:
            invalid.append([rank, "trace missing"])
            continue
        events = json.loads(Path(files[-1]).read_text())
        scopes = sorted((e for e in events if e.get("name") == "extreme::target" and e.get("cat") == "cpu_op"), key=lambda e: float(e["ts"]))
        if len(scopes) != 2:
            invalid.append([rank, "scope count", len(scopes)])
            continue
        comm = [e for e in events if isinstance(e.get("args"), dict) and "size(Byte)" in e["args"] and e.get("ph") == "X"]
        for cycle, scope in enumerate(scopes):
            start = float(scope["ts"])
            end = start + float(scope["dur"])
            h = sorted((e for e in comm if start <= float(e["ts"]) < end), key=lambda e: float(e["ts"]))
            if len(h) < 2:
                invalid.append([rank, cycle, "collective count", len(h)])
                continue
            matches = [e for e in h if e["name"] == "reduce_scatterAivKernel" and int(e["args"]["size(Byte)"]) == 98304]
            if not matches:
                invalid.append([rank, cycle, "first reduce_scatter missing"])
                continue
            first = matches[0]
            rows.append(dict(rank=rank, cycle=cycle, count=len(h), target_start_us=start,
                             first_start_us=float(first["ts"]), first_end_us=float(first["ts"]) + float(first["dur"]),
                             first_duration_ms=float(first["dur"])/1000,
                             first_name=first["name"], first_payload_bytes=first["args"]["size(Byte)"], preceding_collectives=sum(float(e["ts"]) < float(first["ts"]) for e in h),
                             rest_event_duration_sum_ms=sum(float(e["dur"]) for e in h if e is not first)/1000))
    cycles = []
    for cycle in range(2):
        x = [r for r in rows if r["cycle"] == cycle]
        if len(x) != 8:
            invalid.append([cycle, "rank count", len(x)])
            continue
        spread = lambda key: (max(r[key] for r in x) - min(r[key] for r in x))/1000
        cycles.append(dict(cycle=cycle, first_collective_names=sorted(set(r["first_name"] for r in x)),
                           first_payload_bytes=sorted(set(r["first_payload_bytes"] for r in x)),
                           preceding_collectives_by_rank=[r["preceding_collectives"] for r in sorted(x, key=lambda r:r["rank"])],
                           target_start_skew_ms=spread("target_start_us"),
                           first_start_skew_ms=spread("first_start_us"),
                           first_end_skew_ms=spread("first_end_us"),
                           first_duration_ms_by_rank=[r["first_duration_ms"] for r in sorted(x, key=lambda r:r["rank"])],
                           first_duration_median_ms=statistics.median(r["first_duration_ms"] for r in x),
                           rest_event_duration_sum_median_ms=statistics.median(r["rest_event_duration_sum_ms"] for r in x)))
    return dict(status="valid" if len(rows) == 16 and not invalid else "invalid", cycles=cycles, invalid=invalid,
                interpretation="First collective duration largely includes wait for late ranks in these synchronized profiler captures; no inference about normal product wait or attainable HCCL latency. Event-duration sums are descriptive only.")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args=ap.parse_args()
    out=screen(args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2)+"\n")
    print(json.dumps({"status":out["status"],"cycles":out["cycles"]},indent=2))

if __name__ == "__main__":
    main()
