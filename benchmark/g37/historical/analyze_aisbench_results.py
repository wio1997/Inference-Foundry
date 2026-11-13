"""Validate actual AISBench 3.1 outputs before publishing performance numbers.

Grounded in the exact client image's calculators and summarizer, copied under
client_contract/. No fabricated/synthetic measurements are accepted.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def number(value):
    return float(str(value).split()[0])


def analyze(run_root, length):
    root = run_root / f"formal_{length}"
    details = list(root.rglob("*_details.jsonl"))
    assert len(details) == 1, f"Expected one raw details file: {details}"
    detail = details[0]
    rows = [json.loads(line) for line in detail.read_text().splitlines() if line.strip()]
    assert len(rows) == 32, f"Expected32 requests, got{len(rows)}"
    assert all(row.get("success") is True for row in rows), "Failed request present"
    assert all(row.get("input_tokens") == length for row in rows), "Input length/usage mismatch"
    assert all(row.get("output_tokens") == 1024 for row in rows), "Output length/usage mismatch"
    common_path = detail.with_name(detail.name.removesuffix("_details.jsonl") + ".json")
    csv_path = common_path.with_suffix(".csv")
    common = json.loads(common_path.read_text())

    def metric(name):
        stage = common[name]
        assert set(stage) == {"total"}, f"Unexpected AISBench stages: {name}: {stage}"
        return number(stage["total"])

    assert metric("Total Requests") == metric("Success Requests") == 32
    assert metric("Failed Requests") == 0
    assert metric("Max Concurrency") == 16
    assert metric("Total Input Tokens") == 32 * length
    assert metric("Total Generated Tokens") == 32 * 1024
    duration = metric("Benchmark Duration") / 1000
    assert duration > 0
    output_tps = metric("Output Token Throughput")
    # The official calculator rounds duration(ms) and throughput to4 decimals.
    assert math.isclose(output_tps, 32768 / duration, rel_tol=1e-6, abs_tol=.001)
    with csv_path.open(newline="") as stream:
        statistics = {row["Performance Parameters"]: row for row in csv.DictReader(stream)}
    latency = {}
    for name in ("TTFT", "TPOT", "E2EL"):
        row = statistics[name]
        assert row["Stage"] == "total"
        for stat in ("mean", "p99"):
            aliases = {"mean", "average"} if stat == "mean" else {stat}
            keys = [key for key in row if key.lower() in aliases]
            assert len(keys) == 1, f"Missing{stat}: {row}"
            value = row[keys[0]]
            assert str(value).endswith(" ms"), f"Unexpected latency unit: {value}"
            latency[f"{name.lower()}_{stat}_ms"] = number(value)
    return {
        "status": "validated_actual_AISBench_requests_and_aggregates",
        "input_tokens": length,
        "output_tokens": 1024,
        "configured_concurrency": 16,
        "average_concurrency": metric("Concurrency"),
        "requests": 32,
        "successful_requests": 32,
        "duration_s": duration,
        "aggregate_output_tokens_per_s": output_tps,
        "aggregate_input_tokens_per_s": metric("Input Token Throughput"),
        "aggregate_total_tokens_per_s": metric("Total Token Throughput"),
        **latency,
        "evidence": {
            str(path.relative_to(run_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (detail, common_path, csv_path)
        },
        "scope": "one fixed32-request run; no repeatability or model-accuracy claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--profile", type=int, choices=(32768, 65536, 131072, 262144))
    args = parser.parse_args()
    lengths = [args.profile] if args.profile else [32768, 65536, 131072, 262144]
    results = []
    failed = False
    for length in lengths:
        try:
            row = analyze(args.run_root, length)
        except Exception as error:
            failed = True
            row = {"input_tokens": length, "status": "NOT_VALIDATED", "error": str(error)}
        results.append(row)
    suffix = str(args.profile) if args.profile else "all"
    output = args.run_root / f"verified_results_{suffix}.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
