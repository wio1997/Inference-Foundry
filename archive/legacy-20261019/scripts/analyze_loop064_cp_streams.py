#!/usr/bin/env python3
"""Summarize observed cross-stream Compressor/QLI device intervals.

This is a narrow timeline discriminator, not a per-layer time or TPS bound.
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def interval(event):
    return float(event["ts"]), float(event["ts"]) + float(event["dur"])


def overlap(left, right):
    a, b = interval(left)
    c, d = interval(right)
    return max(0.0, min(b, d) - max(a, c))


def summarize(path):
    events = json.loads(path.read_text())
    kernels = [event for event in events if event.get("ph") == "X"
               and event.get("name") in ("Compressor", "VllmQuantLightningIndexer")
               and "Physic Stream Id" in event.get("args", {})]
    compressor = [event for event in kernels if event["name"] == "Compressor"]
    qli = [event for event in kernels if event["name"] == "VllmQuantLightningIndexer"]
    comp_stream = Counter(str(event["args"]["Physic Stream Id"]) for event in compressor)
    qli_stream = Counter(str(event["args"]["Physic Stream Id"]) for event in qli)
    qli_main = qli_stream.most_common(1)[0][0] if qli_stream else None
    aux = [event for event in compressor if str(event["args"]["Physic Stream Id"]) != qli_main]
    main_device = [event for event in events if event.get("ph") == "X"
                   and "dur" in event and str(event.get("args", {}).get("Physic Stream Id")) == qli_main]
    rows = []
    for event in sorted(aux, key=lambda x: float(x["ts"])):
        nearby = min(qli, key=lambda q: abs(float(q["ts"]) - float(event["ts"]))) if qli else None
        simultaneous = [
            {"name": other["name"], "overlap_us": overlap(event, other),
             "start_offset_us": float(other["ts"]) - float(event["ts"])}
            for other in main_device if overlap(event, other) > 0
        ]
        rows.append({
            "concurrent_main_stream_ops": simultaneous,
            "concurrent_main_stream_overlap_us": sum(item["overlap_us"] for item in simultaneous),
            "compressor_start_us": float(event["ts"]),
            "compressor_duration_us": float(event["dur"]),
            "compressor_stream": str(event["args"]["Physic Stream Id"]),
            "nearest_qli_start_us": float(nearby["ts"]) if nearby else None,
            "nearest_qli_duration_us": float(nearby["dur"]) if nearby else None,
            "nearest_qli_stream": str(nearby["args"]["Physic Stream Id"]) if nearby else None,
            "nearest_qli_overlap_us": overlap(event, nearby) if nearby else None,
            "max_qli_overlap_us": max((overlap(event, q) for q in qli), default=0.0),
        })
    return {
        "trace": str(path),
        "compressor_count": len(compressor),
        "qli_count": len(qli),
        "compressor_by_stream": dict(comp_stream),
        "qli_by_stream": dict(qli_stream),
        "off_qli_stream_compressor_count": len(aux),
        "off_qli_stream_compressor_rows": rows,
        "scope": "Device interval overlap only; target-layer mapping, dependency join and net cycle effect require further evidence",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.profile_root)
    traces = sorted(root.glob("rank*_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json"))
    if not traces:
        raise SystemExit("No exported trace_view.json files")
    result = {"trace_count": len(traces), "traces": [summarize(p) for p in traces]}
    Path(args.output).write_text(json.dumps(result, indent=2) + chr(10))
    print(json.dumps({"trace_count": len(traces),
                      "off_stream_counts": [x["off_qli_stream_compressor_count"] for x in result["traces"]]}, indent=2))


if __name__ == "__main__":
    main()
