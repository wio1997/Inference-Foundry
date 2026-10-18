#!/usr/bin/env python3
"""Pair native P/D tail records; acceptance means only sampled row agreement."""
import argparse
import json
from pathlib import Path


def compare(p_records, d_records, requests, ranks=range(16)):
    if (not p_records or not d_records or [r.get("input_tokens") for r in requests] != [256, 257]
        or any(r.get("functional_pass") is not True or "P_response" not in r for r in requests)):
        return {"verdict": "incomplete", "reason": "empty records or incomplete request set", "device_pass": False}
    mismatches, incomplete, matched, missing_mtp = [], [], 0, []
    def coverage(records, capture, rank, n, side):
        registries = [r for r in records if r.get("kind") == "cache_registry" and r.get("tp_rank") == rank]
        if len(registries) != 1:
            incomplete.append([n, side, rank, "cache registry missing/duplicate"])
            return
        expected = {(layer["name"], c, pos) for layer in registries[0]["layers"]
                    for c in range(len(layer["views"])) for pos in (n - 2, n - 1)}
        observed = {(r["layer"], r["component"], r["position"]) for r in capture.get("rows", [])}
        if not expected or expected != observed:
            incomplete.append([n, side, rank, "sample row/component coverage differs from registry"])
    for request in requests:
        n = request["input_tokens"]
        remote_id = request["P_response"]["kv_transfer_params"]["remote_request_id"]
        prompt_sha = request["prompt_sha256"]
        p_index = {}
        for rank in ranks:
            selected = [r for r in p_records if r.get("kind") == "P_tail" and r.get("req_id") == remote_id and r.get("tp_rank") == rank]
            if len(selected) != 1 or selected[0].get("status") != "observed":
                incomplete.append([n, "P", rank, "missing/refused/duplicate tail"])
                continue
            p = selected[0]
            if p.get("n") != n or p.get("prompt_sha256") != prompt_sha or p.get("boundary") != "native_get_output_returned":
                incomplete.append([n, "P", rank, "identity/boundary mismatch"])
                continue
            identities = [r for r in p_records if r.get("kind") == "runner_identity" and r.get("tp_rank") == rank]
            if (len(identities) != 1 or identities[0].get("runnable_hook_bound") is not True
                or p.get("runnable_hook_bound") is not True):
                incomplete.append([n, "P", rank, "runnable hook not established"])
                continue
            if p.get("mtp_calls") == 0 and p.get("scheduled_K") == 1:
                missing_mtp.append([n, rank])
            if not p.get("event_ready_after_native_wait") or p.get("mtp_stream") != p.get("constructor_stream"):
                incomplete.append([n, "P", rank, "completion dependency not established"])
                continue
            if p.get("refused_layers"):
                incomplete.append([n, "P", rank, "unsupported/missing writer metadata"])
            coverage(p_records, p, rank, n, "P")
            if not p.get("mtp_layer_names") or not set(p["mtp_layer_names"]).issubset({r["layer"] for r in p.get("rows", []) if r.get("family") == "MTP"}):
                incomplete.append([n, "P", rank, "MTP cache rows missing"])
            for row in p.get("rows", []):
                key = (rank, row["layer"], row["component"], row["position"])
                if (key in p_index or not row.get("writer_metadata_observed") or
                    row.get("writer_slot") != row["block"] * 128 + row["offset"] or
                    row["offset"] != row["position"] % 128):
                    incomplete.append([n, "P", rank, "duplicate/unproven writer row"])
                p_index[key] = row
        for rank in ranks:
            selected = [r for r in d_records if r.get("kind") == "D_tail" and r.get("remote_req_id") == remote_id and r.get("tp_rank") == rank]
            if len(selected) != 1 or selected[0].get("status") != "observed":
                incomplete.append([n, "D", rank, "missing/refused/duplicate tail"])
                continue
            d = selected[0]
            if d.get("n") != n or d.get("boundary") != "native_handle_requests_returned_before_finished_enqueue":
                incomplete.append([n, "D", rank, "identity/boundary mismatch"])
                continue
            if d.get("refused_layers") or not d.get("rows"):
                incomplete.append([n, "D", rank, "incomplete rows"])
            coverage(d_records, d, rank, n, "D")
            for row in d.get("rows", []):
                key = (row["remote_tp"], row["layer"], row["component"], row["position"])
                source = p_index.get(key)
                if source is None:
                    incomplete.append([n, "D", rank, "no selected P writer reference", key])
                    continue
                fields = ("dtype", "row_shape", "bytes", "offset", "sha256")
                if source["block"] != row["remote_block"] or any(source[f] != row[f] for f in fields):
                    mismatches.append({"n": n, "D_rank": rank, "P_key": key, "reason": "source mapping or sampled bytes differ"})
                else:
                    matched += 1
    verdict = "sampled_rows_match" if matched and not (mismatches or incomplete or missing_mtp) else "incomplete"
    if mismatches or missing_mtp:
        verdict = "direct_advance_not_supported_by_observation"
    return {"verdict": verdict, "matched_sampled_rows": matched, "mismatches": mismatches,
            "missing_P_mtp_call": missing_mtp, "incomplete": incomplete,
            "device_pass": False, "full_prefix_proved": False,
            "scope": "Finite row/call diagnostic only; no optimization, kernel-completion trace, full-prefix or end-to-end correctness acceptance"}


def read_records(root):
    return [json.loads(line) for path in sorted(root.glob("*.jsonl"))
            for line in path.read_text().splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p-dir", type=Path, required=True)
    p.add_argument("--d-dir", type=Path, required=True)
    p.add_argument("--requests", type=Path, required=True)
    a = p.parse_args()
    report = compare(read_records(a.p_dir), read_records(a.d_dir), json.loads(a.requests.read_text())["rows"])
    print(json.dumps(report, indent=2))
    return 0 if report["verdict"] == "sampled_rows_match" else 2


if __name__ == "__main__":
    raise SystemExit(main())
