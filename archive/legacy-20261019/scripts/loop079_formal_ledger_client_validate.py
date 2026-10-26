#!/usr/bin/env python3
"""Fail-closed warmup48/measured48 client admission for the Bound diagnostic."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path

from loop079_formal_ledger_client import body

FROZEN_DATASET_SHA256 = "4d9885088dac6b681c52c5ce12d39d192477ed95282f58c828da17e47fb2797d"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def admit_phase(directory, phase, dataset_lines):
    summary_file = json.loads((directory / "summary.json").read_text())
    summary = summary_file["summary"]
    if (summary["scope"] != "instrumented_bound_diagnostic_not_formal_tps"
            or summary["phase"] != phase or summary["n"] != 48
            or summary["success"] != 48 or summary["fail"] != 0
            or summary["concurrency"] != 12 or summary["max_tokens"] != 1024):
        raise ValueError(f"phase contract failed: {phase}")
    if len(summary_file["requests"]) != 48:
        raise ValueError(f"phase request summary incomplete: {phase}")
    if summary["wall_start_monotonic_ns"] >= summary["wall_end_monotonic_ns"]:
        raise ValueError("invalid client wall interval")
    files = sorted(directory.glob("request_*.json"))
    if len(files) != 48 or [p.name for p in files] != [f"request_{i:03d}.json" for i in range(48)]:
        raise ValueError(f"phase request files incomplete: {phase}")
    admitted = []
    intervals = []
    for i, path in enumerate(files):
        record = json.loads(path.read_text())
        row = record["row"]
        if row != summary_file["requests"][i]:
            raise ValueError(f"request summary mismatch: {phase}/{i}")
        if (row["i"] != i or row["phase"] != phase or row["http_status"] != 200
                or row["error"] is not None or row["output_tokens"] != 1024
                or row["usage"].get("completion_tokens") != 1024
                or type(row["usage"].get("prompt_tokens")) is not int
                or row["usage"]["prompt_tokens"] < 0
                or row["input_tokens"] != row["usage"]["prompt_tokens"]
                or not isinstance(row["response_id"], str) or not row["response_id"]):
            raise ValueError(f"request contract failed: {phase}/{i}")
        if (type(row["chunks"]) is not int or row["chunks"] < 1
                or not isinstance(row["ttft_ms"], (int, float)) or not math.isfinite(row["ttft_ms"]) or row["ttft_ms"] <= 0
                or not isinstance(row["tpot_ms"], (int, float)) or not math.isfinite(row["tpot_ms"]) or row["tpot_ms"] <= 0):
            raise ValueError(f"request timing/text missing: {phase}/{i}")
        if not (summary["wall_start_monotonic_ns"] <= row["start_monotonic_ns"] < row["end_monotonic_ns"] <= summary["wall_end_monotonic_ns"]):
            raise ValueError(f"request outside client wall: {phase}/{i}")
        intervals.append((row["start_monotonic_ns"], row["end_monotonic_ns"]))
        dataset_raw = dataset_lines[i].encode()
        prompt = json.loads(dataset_lines[i])["question"]
        expected_body_hash = sha(json.dumps(body(prompt, 1024), sort_keys=True, ensure_ascii=False).encode())
        if row["dataset_row_sha256"] != sha(dataset_raw) or row["request_body_sha256"] != expected_body_hash:
            raise ValueError(f"frozen dataset/body mismatch: {phase}/{i}")
        events = record["events"]
        if len(events) != row["sse_events"] or not events or events[-1].get("done") is not True:
            raise ValueError(f"SSE terminal mismatch: {phase}/{i}")
        usage_events = 0
        saw_finish = False
        text_delta_events = 0
        previous_recv_ns = row["start_monotonic_ns"]
        for seq, event in enumerate(events):
            raw = base64.b64decode(event["payload_b64"], validate=True)
            if event["seq"] != seq or event["payload_sha256"] != sha(raw) or event["response_id"] != row["response_id"]:
                raise ValueError(f"SSE payload identity mismatch: {phase}/{i}/{seq}")
            recv_ns = event["fragment_recv_monotonic_ns"]
            if type(recv_ns) is not int or not (previous_recv_ns <= recv_ns <= row["end_monotonic_ns"]):
                raise ValueError(f"SSE event outside request: {phase}/{i}/{seq}")
            previous_recv_ns = recv_ns
            if seq == len(events) - 1:
                if raw != b"[DONE]":
                    raise ValueError(f"SSE DONE bytes mismatch: {phase}/{i}")
            elif raw == b"[DONE]" or event.get("done"):
                raise ValueError(f"early SSE DONE: {phase}/{i}/{seq}")
            else:
                data = json.loads(raw)
                if not isinstance(data, dict) or data.get("error") is not None or data.get("id") != row["response_id"]:
                    raise ValueError(f"SSE semantic identity/error: {phase}/{i}/{seq}")
                choices = data.get("choices", [])
                if not isinstance(choices, list) or len(choices) > 1:
                    raise ValueError(f"SSE choice cardinality: {phase}/{i}/{seq}")
                has_delta = False
                for choice in choices:
                    if choice.get("index") != 0 or choice.get("finish_reason") == "error":
                        raise ValueError(f"SSE choice index/error: {phase}/{i}/{seq}")
                    saw_finish |= choice.get("finish_reason") is not None
                    delta = choice.get("delta") or {}
                    has_delta |= bool(delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"))
                if event.get("has_text_delta") is not has_delta:
                    raise ValueError(f"SSE text delta mismatch: {phase}/{i}/{seq}")
                text_delta_events += has_delta
                usage = data.get("usage")
                if usage is not None:
                    usage_events += 1
                    if not isinstance(usage, dict) or type(usage.get("completion_tokens")) is not int or type(usage.get("prompt_tokens")) is not int:
                        raise ValueError(f"SSE usage type: {phase}/{i}/{seq}")
                    if usage != row["usage"] or seq != len(events) - 2:
                        raise ValueError(f"SSE final usage mismatch: {phase}/{i}/{seq}")
        if usage_events != 1 or not saw_finish or text_delta_events != row["chunks"]:
            raise ValueError(f"SSE final usage/finish missing: {phase}/{i}")
        final_data = json.loads(base64.b64decode(events[-2]["payload_b64"]))
        if final_data.get("usage", {}).get("completion_tokens") != 1024:
            raise ValueError(f"missing final usage: {phase}/{i}")
        admitted.append({
            "phase": phase,
            "dataset_index": i,
            "response_id": row["response_id"],
            "dataset_row_sha256": row["dataset_row_sha256"],
            "request_body_sha256": row["request_body_sha256"],
            "sse_events": len(events),
            "start_monotonic_ns": row["start_monotonic_ns"],
            "end_monotonic_ns": row["end_monotonic_ns"],
        })
    points = sorted([(start, 1) for start, _ in intervals] + [(end, -1) for _, end in intervals])
    active = peak = 0
    for _, change in points:
        active += change
        peak = max(peak, active)
    if active != 0 or peak != 12:
        raise ValueError(f"actual client concurrency mismatch: {phase}, peak={peak}")
    return admitted, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--warmup-dir", required=True, type=Path)
    p.add_argument("--measured-dir", type=Path)
    p.add_argument("--warmup-only", action="store_true")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    dataset_raw = args.dataset.read_bytes()
    if sha(dataset_raw) != FROZEN_DATASET_SHA256:
        raise ValueError("frozen dataset SHA mismatch")
    lines = dataset_raw.decode().splitlines()[:48]
    if len(lines) != 48:
        raise ValueError("frozen dataset requires 48 rows")
    warm, warm_summary = admit_phase(args.warmup_dir, "warmup", lines)
    for field in ("kernel_boot_id", "time_namespace", "timens_offsets"):
        value = warm_summary["clock"].get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"missing warmup client clock identity: {field}")
    if args.warmup_only:
        if args.measured_dir is not None:
            raise ValueError("warmup-only cannot accept measured dir")
        result = {
            "status": "warmup48_client_admitted",
            "scope": "instrumented_bound_diagnostic_not_formal_tps",
            "request_count": 48,
            "unique_response_ids": len(set(row["response_id"] for row in warm)),
            "warmup_summary": warm_summary,
            "request_index": warm,
        }
        if result["unique_response_ids"] != 48:
            raise ValueError("duplicate warmup response IDs")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        print(json.dumps({"status": result["status"], "requests": 48}))
        return
    if args.measured_dir is None:
        raise ValueError("measured-dir required without warmup-only")
    measured, measured_summary = admit_phase(args.measured_dir, "measured", lines)
    all_ids = [row["response_id"] for row in warm + measured]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("duplicate response id across phases")
    if warm_summary["wall_end_monotonic_ns"] > measured_summary["wall_start_monotonic_ns"]:
        raise ValueError("warmup/measured chronology overlap")
    for field in ("kernel_boot_id", "time_namespace", "timens_offsets"):
        value = measured_summary["clock"].get(field)
        if not isinstance(value, str) or not value.strip() or warm_summary["clock"][field] != value:
            raise ValueError(f"client clock scope changed: {field}")
    if [row["request_body_sha256"] for row in warm] != [row["request_body_sha256"] for row in measured]:
        raise ValueError("warmup/measured request bodies differ")
    result = {
        "status": "client_two_phase_admitted",
        "scope": "instrumented_bound_diagnostic_not_formal_tps",
        "request_count": 96,
        "unique_response_ids": 96,
        "same_48_request_bodies_across_phases": True,
        "warmup_summary": warm_summary,
        "measured_summary": measured_summary,
        "request_index": warm + measured,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "requests": 96}))


if __name__ == "__main__":
    main()
