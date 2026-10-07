"""Independently reconcile saved Run267 baseline/NOT_APPLICABLE evidence.

Never import the failed controller or infer a candidate performance comparison.
"""
from pathlib import Path
import hashlib
import json
import math
import re

ROOT = Path(__file__).resolve().parents[2] / "runs/GLM-RUN-0267"


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    assert hashlib.sha256((ROOT / "controller_spec.json").read_bytes()).hexdigest() == "202a2a2548a67b3531bec7156e5e334c1dba671b1ddc1eda8e0f1b3433269bcc"
    assert read("state.json")["status"] == "failed"
    assert read("earlymtpcompare.phase.json")["exit_code"] == 1
    assert "NOT_APPLICABLE" in read("failure.json")["error"]
    assert not read("failure.json")["results"]
    job = "jobs/APPLICABILITY-CLOSURE-20261007/"
    assert hashlib.sha256((ROOT / job / "controller_spec.json").read_bytes()).hexdigest() == "9a6f62594004cd35cfbeb41a8ebe2e106a9da8515e426bb6ecb87bf881800d7d"
    closure = read(job + "result.json")
    assert read(job + "state.json")["status"] == "completed" and closure["passed"]
    assert closure["model_requests"] == 0 and not closure["D_reload"] and not closure["source_mutation"]
    assert closure["original_state_preserved"] and closure["same_workers"] and closure["async_all16"]
    for identity_name, prefix in (("download_identity.json", ""), ("candidate_PD_download_identity.json", ""), (job + "download_identity.json", job)):
        for name, row in read(identity_name).items():
            assert hashlib.sha256((ROOT / prefix / name).read_bytes()).hexdigest() == row["sha256"], name

    labels = sorted(p.name.removesuffix("_request.json") for p in ROOT.glob("*_request.json"))
    assert labels == ["correctness_mode0_complete", "correctness_mode0_short", "retained_mode_warm"]
    gold = read("functional_plan.json")["complete_golden"]
    native = (ROOT / "epochs/candidate/native_167.log").read_text(errors="replace")
    requests = {}
    for label in labels:
        body = read(label + "_request.json")
        result = read(label + "_result.json")
        events = read(label + "_events.json")
        parsed = [None if line[6:] == "[DONE]" else json.loads(line[6:]) for line in (ROOT / (label + "_D.sse")).read_text().splitlines() if line.startswith("data: ")]
        assert parsed == [event["value"] for event in events] and parsed[-1] is None
        assert not any(event and "error" in event for event in parsed)
        choices = [choice for event in parsed if event for choice in event.get("choices", [])]
        ids = [token for choice in choices for token in choice.get("token_ids", choice.get("delta", {}).get("token_ids", [])) or []]
        complete = body["max_tokens"] == 96
        expected = gold["token_ids"] if complete else [785, 1196, 374, 10156, 264, 3405, 304, 8452]
        assert ids == result["token_ids"] == expected
        assert result["completion_tokens"] == len(expected)
        assert result["finish_reason"] == ("stop" if complete else "length")
        assert result["prompt_tokens"] == (gold["prompt_tokens"] if complete else 2334)
        if complete:
            assert result["final_content"] == gold["final_content"] and result["semantic_accepted"]
        usage = [event["usage"] for event in parsed if event and event.get("usage")][-1]
        assert usage["prompt_tokens"] == result["prompt_tokens"] and usage["completion_tokens"] == len(ids)
        arrivals = [event["arrived_ns"] for event in events if event["value"] and any(choice.get("token_ids") or choice.get("delta", {}).get("token_ids") for choice in event["value"].get("choices", []))]
        assert math.isclose(result["TPOT_ms"], (arrivals[-1] - arrivals[0]) / 1e6 / (len(ids) - 1), abs_tol=1e-8)
        assert math.isclose(result["PD_wall_s"], result["P_wall_s"] + result["D_wall_s"], abs_tol=1e-8)
        assert any("hits_total" in key and value == result["prompt_tokens"] for key, value in result["external_KV_delta"].items())
        work = read(label + "_workload.json")
        delta = {key: work["after"]["counters"][key] - value for key, value in work["before"]["counters"].items()}
        assert delta == work["delta"] and all(value >= 0 and float(value).is_integer() for value in delta.values())
        normalized = {key.split("{", 1)[0]: int(value) for key, value in delta.items()}
        signature = {"num_drafts": normalized["vllm:spec_decode_num_drafts_total"], "num_draft_tokens": normalized["vllm:spec_decode_num_draft_tokens_total"], "num_accepted_tokens": normalized["vllm:spec_decode_num_accepted_tokens_total"], "accepted_position0": normalized["vllm:spec_decode_num_accepted_tokens_per_pos_total"]}
        signature["invalid_draft_tokens"] = signature["num_drafts"] - signature["num_draft_tokens"]
        assert signature == work["signature"] and all(work["checks"].values())
        request_id = read(label + "_P.raw")["kv_transfer_params"]["remote_request_id"]
        lines = [line for line in native.splitlines() if "KV cache transfer for request " + request_id + " took " in line]
        assert len(lines) == 16 and {int(re.search(r"local_device_id (\d+)", line).group(1)) for line in lines} == set(range(16))
        requests[label] = dict(prompt_tokens=result["prompt_tokens"], output_tokens=len(ids), finish_reason=result["finish_reason"], MTP_signature=signature, all16_KV_transfer=True)

    witness = read("retained_witness.json")
    roles = witness["same_worker_identity"]
    ranks = set(range(16))
    pids = set(roles["worker_namespace_pids"].values())
    for key in ("H10_rows", "H9_rows", "H5_rows"):
        rows = witness[key]
        assert len(rows) == 16 and {row["rank"] for row in rows} == ranks and {row["pid"] for row in rows} == pids
    assert witness["H6_mode"] == witness["H5_event_mode"] == 1 and witness["H10_mode"] == witness["H9_mode"] == 0
    assert witness["H6"]["all_rank_witness"] and witness["H6"]["mode"] == 1
    assert witness["H6"]["library_sha256"] == read("functional_plan.json")["native_candidate"]["sha256"]
    assert len(witness["H6"]["rows"]) == 16 and all(row["mode"] == 1 and row["dispatch_cached_true"] and row["combine_cached_true"] for row in witness["H6"]["rows"])
    assert all(row["mode"] == 1 and row["returned_none"] and not row["configured_overlap"] for row in witness["H5_rows"])
    assert all(row["mode"] == 0 and not row["eligible"] and not row["early"] and row["async_scheduling"] and row["num_spec"] == row["scheduled_K"] == 1 and row["num_tokens"] == 2 for row in witness["H10_rows"])
    assert all(row["mode"] == 0 and not row["skipped_host_sync"] and row["runtime_mode"] == "FULL" and row["capture_sizes"] == [2] and row["MTP_enforce_eager"] for row in witness["H9_rows"])
    assert not list((ROOT / "witnesses").glob("mode1*")) and (ROOT / "early_draft_mode.bin").read_bytes() == b"\x00"
    guards = read("guards_after.json")
    assert all(row["health"] == 200 and row["idle"] and len(row["device_owners"]) == 16 for row in guards.values())
    assert roles == guards["167"]
    stack = read("retained_stack.json")
    assert stack["H6"] and stack["H5"] and not any(stack[key] for key in ("H10", "H9", "H8", "H4"))
    assert stack["normal_requests"] == 3 and not stack["comparison_completed"] and stack["pure_KV_API_repairs"]
    output = dict(verdict="NOT_APPLICABLE_ASYNC_ALREADY_ACTIVE", candidate_ever_enabled=False, candidate_NPU_correctness=False, matched_AB_completed=False, performance_gain=None, baseline_correctness=True, requests=requests, original_failed_state_preserved=True, separate_zero_request_closure=True, async_all16=True, same_workers_at_closure=True, retained_research_stack=["H6", "H5"], Current=None, formal_SLA=False, full_API=False)
    (ROOT / "applicability_reduced.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "requests"}))


if __name__ == "__main__":
    main()
