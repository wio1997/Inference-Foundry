#!/usr/bin/env python3
"""Summarize admitted Host lineage; never convert instrumented times to a Bound."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values):
    return {"n": len(values), "min": min(values), "median": statistics.median(values),
            "max": max(values), "sum": sum(values)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--server-admission", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    server = json.loads(a.server_admission.read_text())
    client_path = a.run_dir / "client_admission.json"
    client = json.loads(client_path.read_text())
    if server["status"] != "server_two_phase_admitted" or client["status"] != "client_two_phase_admitted":
        raise ValueError("Host/client lineage not admitted")
    if server["client_report_sha256"] != digest(client_path):
        raise ValueError("server/client report SHA mismatch")
    phases = {}
    needed = {"scheduler_append", "runtime_handoff", "runner_done", "output_add_request",
              "runtime_post_drain", "output_receive", "api_consume"}
    rows = []
    origins = []
    raw_by_phase_dataset = {}
    for path in sorted((a.run_dir / "ledger").glob("pid*.jsonl")):
        if ".flush." in path.name:
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("event") == "clock_origin":
                origins.append(row)
            if row.get("event") in needed:
                rows.append(row)
    if len(origins) != len(server["ledger_processes"]):
        raise ValueError("server process clock origins incomplete")
    origin_scope = {(r["boot_id"], r["time_namespace"], r["time_namespace_offsets"].strip())
                    for r in origins}
    if len(origin_scope) != 1:
        raise ValueError("server clocks have different scope")
    origin_scope = next(iter(origin_scope))
    for phase in ("warmup", "measured"):
        c = client[f"{phase}_summary"]["clock"]
        if (c["kernel_boot_id"], c["time_namespace"], c["timens_offsets"].strip()) != origin_scope:
            raise ValueError("client/server cross-process monotonic clock scope not proved")
    for phase in ("warmup", "measured"):
        index = {r["response_id"]: r for r in client["request_index"] if r["phase"] == phase}
        if len(index) != 48:
            raise ValueError("client phase not 48")
        subset = [r for r in rows if r["phase"] == phase]
        adds = {r["request_id"]: r for r in subset if r["event"] == "output_add_request"}
        bulk = {r["request_id"]: r for r in subset if r["event"] == "scheduler_append" and r["bulk"]}
        rank0_handoffs = [r for r in subset if r["event"] == "runtime_handoff" and r["rank"] == 0]
        rank0_done = {(r["cohort"]): r for r in subset if r["event"] == "runner_done" and r["rank"] == 0}
        rank0_drains = [r for r in subset if r["event"] == "runtime_post_drain" and r["rank"] == 0]
        if len(adds) != 48 or len(bulk) != 48:
            raise ValueError("internal request count")
        if len(rank0_handoffs) != 4 or len(rank0_done) != 4 or len(rank0_drains) != 4:
            raise ValueError("rank0 cohort count")
        handoff_by_id = {}
        done_by_id = {}
        for row in rank0_handoffs:
            for internal in row["req_ids"]:
                handoff_by_id[internal] = row
                done_by_id[internal] = rank0_done[row["cohort"]]
        if set(adds) != set(bulk) or set(adds) != set(handoff_by_id):
            raise ValueError("Host IDs do not join")
        g = [bulk[k]["g_before"] for k in adds]
        cycles = [rank0_done[k]["cycles"] for k in sorted(rank0_done)]
        sampled_q = sum(sum(sum(int(v) for v in cycle) for cycle in row["count_history"])
                        for row in rank0_drains)
        prefill_by_internal = {}
        for row in subset:
            if row["event"] != "output_receive" or row.get("prefill_stats") in (None, "None"):
                continue
            internal = row["request_id"]
            value = row["prefill_stats"]
            if internal in prefill_by_internal and prefill_by_internal[internal] != value:
                raise ValueError("inconsistent per-request PrefillStats repr")
            prefill_by_internal[internal] = value
        if set(prefill_by_internal) != set(adds):
            raise ValueError("PrefillStats request coverage")
        fields = ("num_prompt_tokens", "num_computed_tokens", "num_cached_tokens",
                  "num_local_cached_tokens", "num_external_cached_tokens")
        prefill_values = []
        for value in prefill_by_internal.values():
            parsed = {}
            for field in fields:
                found = re.search(r"\b" + field + r"=(\d+)\b", value)
                if found is None:
                    raise ValueError("PrefillStats repr field absent")
                parsed[field] = int(found.group(1))
            if parsed["num_prompt_tokens"] != parsed["num_computed_tokens"] + parsed["num_cached_tokens"]:
                raise ValueError("PrefillStats arithmetic mismatch")
            prefill_values.append(parsed)
        api_by_external = {}
        for row in subset:
            if row["event"] == "api_consume":
                api_by_external.setdefault(row["output_request_id"], []).append(row)
        if set(api_by_external) != set(index):
            raise ValueError("API raw request coverage")
        raw_by_phase_dataset[phase] = {}
        for external, c in index.items():
            ordered = sorted(api_by_external[external], key=lambda r: (r["pid"], r["seq"]))
            raw = [t for row in ordered for t in row["raw_ids"]]
            if len(raw) != 1024:
                raise ValueError("API raw length mismatch")
            raw_by_phase_dataset[phase][c["dataset_index"]] = raw
        arrival_ms, prep_to_handoff_ms, runtime_host_ms, drain_ms = [], [], [], []
        for internal, add in adds.items():
            external = add["external_req_id"]
            if external not in index:
                raise ValueError("external ID missing")
            c = index[external]
            hand = handoff_by_id[internal]
            done = done_by_id[internal]
            times = (c["start_monotonic_ns"], add["monotonic_ns"],
                     hand["monotonic_ns"], done["monotonic_ns"], c["end_monotonic_ns"])
            if list(times) != sorted(times):
                raise ValueError("Host/client event chronology reversed")
            arrival_ms.append((times[1] - times[0]) / 1e6)
            prep_to_handoff_ms.append((times[2] - times[1]) / 1e6)
            runtime_host_ms.append((times[3] - times[2]) / 1e6)
            drain_ms.append((times[4] - times[3]) / 1e6)
        client_sum = client[f"{phase}_summary"]
        phases[phase] = {
            "request_count": 48,
            "prebulk_g": summary(g),
            "prebulk_g_histogram": dict(sorted(Counter(g).items())),
            "ordinary_admitted_tokens": sum(g),
            "runtime_retained_tokens": 48 * 1024,
            "runtime_sampled_q_before_remaining_clip": sampled_q,
            "runtime_q_minus_retained_R": sampled_q - 48 * 1024,
            "runtime_terminal_admitted_tokens": 48 * 1024 - sum(g),
            "output_processor_prefill_stats_repr": {
                "request_count": len(prefill_values),
                "prompt_tokens_total": sum(v["num_prompt_tokens"] for v in prefill_values),
                "computed_prompt_tokens": summary([v["num_computed_tokens"] for v in prefill_values]),
                "cached_prompt_tokens": summary([v["num_cached_tokens"] for v in prefill_values]),
                "external_cached_prompt_tokens_total": sum(v["num_external_cached_tokens"] for v in prefill_values),
                "scope": "OutputProcessor PrefillStats repr, not device-ready or physical work proof",
            },
            "all48_first_runtime_token_is_in_bulk_admitted_prefix": all(
                r["admitted_raw_ids"] and r["admitted_raw_ids"][0] == r["incoming_raw_ids"][0]
                for r in bulk.values()),
            "first_runtime_token_server_full_raw_ordinal_zero_based_range": [min(g), max(g)],
            "rank0_cohort_cycles": cycles,
            "rank0_cohort_cycle_sum": sum(cycles),
            "runtime_retained_tokens_per_rank0_cycle_observed": 48 * 1024 / sum(cycles),
            "runtime_admitted_tokens_per_rank0_cycle_observed": (48 * 1024 - sum(g)) / sum(cycles),
            "client_wall_s_instrumented": client_sum["duration_s"],
            "client_tps_instrumented_diagnostic_only": client_sum["output_tps_diagnostic_only"],
            "client_start_to_output_add_ms": summary(arrival_ms),
            "output_add_to_rank0_handoff_ms": summary(prep_to_handoff_ms),
            "rank0_handoff_to_rank0_done_ms": summary(runtime_host_ms),
            "rank0_done_to_client_end_ms": summary(drain_ms),
        }
    common_prefix = []
    equal = 0
    for i in range(48):
        warm = raw_by_phase_dataset["warmup"][i]
        measured = raw_by_phase_dataset["measured"][i]
        equal += warm == measured
        common_prefix.append(next((j for j, (a_token, b_token) in enumerate(zip(warm, measured))
                                   if a_token != b_token), 1024))
    result = {
        "status": "posthoc_host_lineage_metrics_conditional_on_run543_admission",
        "scope": "instrumented 48+48 Host/client trajectory only; not formal TPS, compulsory work, device-ready or feasible Scheduling Bound",
        "source_run": "Run542",
        "source_controller_exit": 1,
        "source_controller_failure": "original validator expected enum repr CUDAGraphMode.FULL while actual str(mode) is FULL",
        "server_admission_sha256": digest(a.server_admission),
        "client_admission_sha256": digest(client_path),
        "cross_process_monotonic_clock_scope": {
            "boot_id": origin_scope[0], "time_namespace": origin_scope[1],
            "time_namespace_offsets": origin_scope[2],
            "scope": "same boot/namespace/offsets for 10 server processes and two client phases; controller clock not proved",
        },
        "phases": phases,
        "paired_warmup_measured_server_raw_outputs": {
            "identical_full_1024_sequences": equal,
            "common_prefix_tokens": summary(common_prefix),
            "scope": "Observed current trajectories only; no semantic equivalence or memoization policy conclusion",
        },
        "still_unknown": ["legal pre-window memoization", "fresh required Target F", "cache/prefill/seed/KV device-ready", "all8 mixed resource service", "critical path overlap", "controller clock scope"],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "measured_g_sum": phases["measured"]["ordinary_admitted_tokens"],
                      "measured_cycles": phases["measured"]["rank0_cohort_cycle_sum"]}))


if __name__ == "__main__":
    main()
