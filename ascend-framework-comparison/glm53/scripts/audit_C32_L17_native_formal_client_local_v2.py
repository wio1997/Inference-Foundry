"""DRAFT_PENDING_PRIMARY_REVIEW: L17 formal client audit v2 with optional D storage.

The original client report is never rewritten.  The source-qualified native
optional-D-storage helper is applied separately; raw D storage nulls remain null.
Timing is the client full clock including native API/gateway P/D queueing.
No model, SDK, SSH or resource call is made.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import glm53_C32_L17_pipeline_contract_v3 as contract

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, p):
    ordered = sorted(values)
    at = (len(ordered) - 1) * p
    low = math.floor(at)
    high = math.ceil(at)
    return ordered[low] + (ordered[high] - ordered[low]) * (at - low)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8)


def audit(run_id):
    assert run_id == contract.FORMAL_RUN
    run = ROOT / "runs" / run_id
    raw = (run / "formal-result.remote.json").read_bytes()
    report = json.loads(raw)
    framework = report["framework"]
    concurrency = report["concurrency"]
    namespace = report["namespace"]
    assert report["project_id"] == "glm53"
    assert framework == "sglang" and concurrency == 32
    assert report["run_id"] == run_id
    assert namespace == "glm53-formal-" + run_id + "-C32"
    expected = json.loads((ROOT / contract.FORMAL_MANIFEST_FILE).read_text())
    assert hashlib.sha256((json.dumps(expected["data"], indent=2) + "\n").encode()).hexdigest() == expected["sha256"] == report["input_manifest_sha256"]
    manifest = expected["data"]
    accepted = {(row["prefix_id"], row["variant"]): row for row in manifest["formal"]}
    assert report["http_connection_policy"] == {
        "upstream_force_close": True,
        "frontend_force_close": True,
        "automatic_model_request_retries": False,
        "revision": "v2",
    }
    assert report["phase"] == "COMPLETE"
    assert len(report["warmup"]) == 32 and len(report["gateway"]) == 160
    cache_review = contract.load_module(
        "l17_cache_annotation", contract.CACHE_ANNOTATION_HELPER, contract.CACHE_ANNOTATION_HELPER_SHA
    )
    review = cache_review.review(report)
    assert review["pass"]
    review.pop("runtime_or_model_changes", None)
    review["cache_review_changed_runtime_or_model"] = False
    cache_path = ROOT / contract.FORMAL_CACHE_REVIEW
    cache_raw = (json.dumps(review, indent=2) + "\n").encode()
    if cache_path.exists():
        assert cache_path.read_bytes() == cache_raw, "Preserve previous cache review"
    else:
        cache_path.write_bytes(cache_raw)
        assert review["independently_verified_prefix_requests"] == 4 * concurrency
    reviewed = {row["request_id"]: row for row in review["rows"]}
    for index, warm in enumerate(report["warmup"]):
        native = warm["native"]
        request = native["extra_details_data"]["glm53_request"]
        assert warm["pass"] and native["success"]
        assert native["input_tokens"] == 74400 and native["output_tokens"] == 1
        assert request["namespace"] == namespace
        assert request["input_sha256_le_u32"] == manifest["prefixes"][index]["ids_sha256_le_u32"]
        assert request["native_attempt_setting"] == 1
        assert warm["request_id"] == f"{namespace}-p{index:02d}-warm"
        assert warm["terminal"]["gateway"]["status"] == "D_STREAM_COMPLETE"
    gate = report["cache_gate"]
    assert gate["pass"] and gate["all_expected_keys_present"] and gate["actual_payload_samples_read"]
    assert gate["prefix_count"] == 32 and gate["SDK_init_bm"] is True and gate["SDK_dram_size"] == "0GB"
    assert gate["close_rc"] == gate["init_rc"] == gate["setup_rc"] == 0
    assert len(gate["prefixes"]) == 32
    assert {row["prefix_id"] for row in gate["prefixes"]} == set(range(32))
    payload_reads = 0
    for prefix in gate["prefixes"]:
        prefix_id = prefix["prefix_id"]
        assert prefix["prefix_ids_sha256_le_u32"] == manifest["prefixes"][prefix_id]["ids_sha256_le_u32"]
        assert prefix["keys"] == 1743
        assert len(prefix["samples"]) == 6
        assert {row["key_index"] for row in prefix["samples"]} == {0, 1, 2, 1740, 1741, 1742}
        for sample in prefix["samples"]:
            assert sample["read1"]["bytes"] == sample["read2"]["bytes"] == sample["expected_bytes"]
            assert sample["read1"]["sha256"] == sample["read2"]["sha256"]
            assert len(sample["read1"]["sha256"]) == 64
            payload_reads += 2
    formal = report["formal"]
    rows = formal["rows"]
    events = formal["client_inflight_events"]
    assert len(rows) == 128 and len(events) == 256 and formal["client_concurrency_limit"] == 32
    assert {(row["prefix_id"], row["variant"]) for row in rows} == set(accepted)
    assert len({row["request_id"] for row in rows}) == len(rows)
    assert formal["first_dispatch_perf_counter_ns"] == min(row["outer_dispatch_perf_counter_ns"] for row in rows)
    assert formal["last_terminal_perf_counter_ns"] == max(row["outer_terminal_perf_counter_ns"] for row in rows)
    seconds = (formal["last_terminal_perf_counter_ns"] - formal["first_dispatch_perf_counter_ns"]) / 1e9
    assert seconds > 0
    ttft = []
    tpot = []
    e2e = []
    output = 0
    logical_input = 0
    noncached_input = None
    inflight = 0
    peak = 0
    refills = []
    for event in events:
        inflight += 1 if event["event"] == "dispatch" else -1
        assert event["event"] in ["dispatch", "terminal"] and 0 <= inflight <= 32
        assert event["client_inflight"] == inflight
        peak = max(peak, inflight)
    assert inflight == 0
    for row in rows:
        native = row["result"]["native"]
        result = row["result"]
        prefix_id = row["prefix_id"]
        variant = row["variant"]
        meta = accepted[(prefix_id, variant)]
        rid = f"{namespace}-p{prefix_id:02d}-v{variant}"
        assert row["request_id"] == native["uuid"] == rid
        assert result["accepted_request_id"] == meta["request_id"]
        assert result["input_ids_sha256_le_u32"] == meta["ids_sha256_le_u32"]
        assert result["error"] is None
        extra = native["extra_details_data"]
        usage = extra["glm53_service_usage"]
        request = extra["glm53_request"]
        assert native["success"] and native["input_tokens"] == usage["prompt_tokens"] == request["input_tokens"] == 80000
        assert native["output_tokens"] == usage["completion_tokens"] == request["output_target"] == 600
        assert extra["glm53_finish_reason"] == "length" and request["native_attempt_setting"] == 1
        assert request["namespace"] == namespace and request["input_sha256_le_u32"] == meta["ids_sha256_le_u32"]
        assert result["count_pass"] and result["logical_success"] and result["transport_success"]
        assert reviewed[rid]["logical_prefix_verified"]
        gateway = result["terminal"]["gateway"]
        assert gateway["logical_request_id"] == rid and gateway["status"] == "D_STREAM_COMPLETE"
        assert gateway["prefill_http_status"] == gateway["decode_http_status"] == 200
        valid = native["extra_perf_data"]["glm53_valid_output_chunk_times"]
        assert valid and all(math.isfinite(value) for value in valid)
        assert all(a <= b for a, b in zip(valid, valid[1:]))
        first = valid[0] - row["outer_dispatch_perf_counter_ns"] / 1e9
        per = (valid[-1] - valid[0]) / 599
        full = (row["outer_terminal_perf_counter_ns"] - row["outer_dispatch_perf_counter_ns"]) / 1e9
        assert first >= 0 and per >= 0 and full >= first
        assert close(first, result["ttft_seconds"]) and close(per, result["tpot_seconds"])
        ttft.append(first)
        tpot.append(per)
        e2e.append(full)
        output += native["output_tokens"]
        logical_input += native["input_tokens"]
    for prefix_id in range(32):
        group = sorted((row for row in rows if row["prefix_id"] == prefix_id), key=lambda row: row["variant"])
        assert [row["variant"] for row in group] == [1, 2, 3, 4]
        for a, b in zip(group, group[1:]):
            assert b["outer_dispatch_perf_counter_ns"] >= a["outer_terminal_perf_counter_ns"]
            refills.append((b["outer_dispatch_perf_counter_ns"] - a["outer_terminal_perf_counter_ns"]) / 1e9)
    warm_terminal = max(row["terminal"]["gateway"]["gateway_terminal_monotonic_ns"] for row in report["warmup"])
    assert formal["first_dispatch_perf_counter_ns"] >= warm_terminal
    summary = report["summary"]
    assert close(summary["full_clock_seconds"], seconds)
    assert summary["formal_requests"] == 128
    assert summary["transport_successes"] == summary["strict_count_lifecycle_successes"] == 128
    assert summary["cache_verified_strict_successes"] == review["original_cache_passes"]
    assert summary["strict_count_lifecycle_failures"] == 0
    assert summary["successful_output_tokens"] == output == 76800
    assert summary["successful_logical_input_tokens"] == logical_input == 10240000
    assert close(summary["successful_output_tokens_per_second"], output / seconds)
    assert close(summary["successful_logical_input_tokens_per_second"], logical_input / seconds)
    quantiles = {}
    for name, values in [("ttft_seconds", ttft), ("tpot_seconds", tpot), ("e2e_seconds", e2e)]:
        quantiles[name] = {label: percentile(values, p) for label, p in [("P50", 0.5), ("P90", 0.9), ("P99", 0.99)]}
        native_quantiles = summary["quantiles"][name]
        assert native_quantiles["measured_requests"] == len(rows) and native_quantiles["missing_requests"] == 0
        assert all(close(native_quantiles[key], value) for key, value in quantiles[name].items())
    sla = (
        quantiles["ttft_seconds"]["P50"] < 4
        and quantiles["ttft_seconds"]["P90"] < 12
        and quantiles["ttft_seconds"]["P99"] < 30
        and quantiles["tpot_seconds"]["P50"] < 0.018
        and quantiles["tpot_seconds"]["P90"] < 0.040
    )
    assert summary["aggregate_SLA"]["pass"] == sla
    assert summary["request_count_and_clock_complete"]
    assert summary["all_formal_prefix_hits_verified"] == (review["original_cache_passes"] == 128)
    assert summary["goodput_tokens_per_second"] is None
    assert report["backend_HTTP_attempts"] == 320 and len(report["gateway"]) == 160
    assert not report["pending_terminal_ids"]
    assert report["D_metrics"]["sample_count"] == report["D_metrics"]["complete_sample_count"] > 0
    steady = summary["native_AISBench_steady"]
    assert steady["status"] in ["CALCULATED", "N/A"]
    if steady["status"] == "CALCULATED":
        assert steady["sample_count"] == len(steady["sample_request_ids"])
        assert set(steady["sample_request_ids"]) <= {row["request_id"] for row in rows}
    manifest_file_sha = digest(ROOT / contract.FORMAL_MANIFEST_FILE)
    result = {
        "project_id": "glm53",
        "framework": framework,
        "concurrency": concurrency,
        "run_id": run_id,
        "namespace": namespace,
        "client_audit_pass": True,
        "measurement_valid": "PENDING_NATIVE_POSTRUN_AUDIT",
        "formal_requests": len(rows),
        "warmup_logical_requests": 32,
        "backend_HTTP_attempts": 320,
        "successful_backend_inferences": 320,
        "all_successful_backend_output_tokens": 76992,
        "formal_successful_output_tokens": output,
        "full_clock_seconds": seconds,
        "output_tokens_per_second": output / seconds,
        "logical_input_tokens_per_second": logical_input / seconds,
        "noncached_logical_P_input_tokens": noncached_input,
        "noncached_scope": "No SGLang cached-token field was filled; no noncached count is invented.",
        "quantiles": quantiles,
        "aggregate_SLA_pass": sla,
        "native_AISBench_steady": steady,
        "actual_client_peak_inflight": peak,
        "refill_gap_seconds": {
            "min": min(refills),
            "max": max(refills),
            "median": percentile(refills, 0.5),
        },
        "gate_payload_reads": payload_reads,
        "all_prefix_metadata_and_repeatable_payload_samples_passed": True,
        "natural_cache_policy_preserved_by_frozen_sources": True,
        "cache_policy": "do_not_clear_L1_L2_after_warm",
        "profiling": False,
        "retries": 0,
        "raw_formal_report_sha256": hashlib.sha256(raw).hexdigest(),
        "D_metric_archive_sha256": report["D_metrics"]["raw_archive_sha256"],
        "native_postrun_audit_required": True,
        "original_client_baseline_accepted": report["baseline_accepted"],
        "original_client_cache_verified_requests": review["original_cache_passes"],
        "independently_verified_prefix_requests": 128,
        "cache_review_revision": review["review_revision"],
        "D_storage_null_requests": review["D_storage_null_requests"],
        "raw_report_and_fields_unchanged": True,
        "cache_annotation_review": contract.FORMAL_CACHE_REVIEW,
        "cache_annotation_review_sha256": digest(cache_path),
        "source_evidence_sha256": {
            contract.FORMAL_CACHE_REVIEW: digest(cache_path),
            contract.CACHE_ANNOTATION_HELPER: contract.CACHE_ANNOTATION_HELPER_SHA,
            contract.FORMAL_MANIFEST_FILE: manifest_file_sha,
            contract.NATIVE_CONFIG: contract.NATIVE_CONFIG_SHA,
        },
    }
    target = ROOT / contract.FORMAL_CLIENT_AUDIT
    target.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    print(json.dumps(audit(parser.parse_args().run_id), indent=2))
