#!/usr/bin/env python3
"""Read-only handoff/output-ledger audit; configured zero is not external zero."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evidence/20260927_loop078_bound/run403"
RUNNER = Path("/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py")
CACHED = Path("/data/wio/vllm_ascend_26/framework/vllm/vllm/v1/worker/gpu_input_batch.py")
REQUEST = Path("/data/wio/vllm_ascend_26/framework/vllm/vllm/v1/request.py")
SCHED = Path("/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/patch/platform/patch_kv_delivery_preemption.py")
SERVING = ROOT / "runtime/fixed_serving.py"


def source(path, needles):
    raw = path.read_bytes()
    txt = raw.decode()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "anchors": {needle: txt.count(needle) for needle in needles}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    src = {
        "runner": source(RUNNER, ["initial_output_counts=[0] * _cfg.batch_size", "handoff_before_model_forward", "_extreme_req_ids = tuple(self.input_batch.req_ids[:num_reqs])", "extreme_bulk_output=True"]),
        "cached_request": source(CACHED, ["output_token_ids: list[int]", "return self.num_prompt_tokens + len(self.output_token_ids)"]),
        "scheduler_request": source(REQUEST, ["self._output_token_ids: list[int] = []", "return len(self._output_token_ids)"]),
        "scheduler_bulk": source(SCHED, ["num_output_tokens_before = len(request._output_token_ids)", "if model_runner_output.extreme_bulk_output:"]),
        "serving": source(SERVING, ["self.initial_output_counts = initial", "self.remaining = remaining"]),
    }
    assert all(all(v >= 1 for v in row["anchors"].values()) for row in src.values())
    capture_tar = BASE / "capture.tar.gz"
    with tarfile.open(capture_tar) as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".json")]
        captures = [json.load(tar.extractfile(m)) for m in members]
    assert len(captures) == 40
    pairs = {(x["cohort"], x["rank"]) for x in captures}
    assert pairs == {(c,r) for c in range(1,6) for r in range(8)}
    assert all(x["initial_output_counts"] == [0]*12 and x["remaining"] == [1024]*12 for x in captures)
    assert all(x["slot_identity"] == "cohort-local slot; external request IDs unavailable" for x in captures)
    runtime = [json.loads((BASE / "runtime" / ("rank%d_cohort%d.json" % (r,c))).read_text()) for c,r in sorted(pairs)]
    assert all(x["handoff_before_model_forward"] is True and x["target_graph_mode"] == "FULL" for x in runtime)
    assert all(x["generated_output_counts"] == [1024]*12 for x in runtime)
    req_ids = set()
    for c in range(1,6):
        cohort = [x for x in runtime if x["cohort"] == c]
        assert all(x["req_ids"] == cohort[0]["req_ids"] for x in cohort)
        req_ids.update(cohort[0]["req_ids"])
    assert len(req_ids) == 60
    gate = json.loads((ROOT / "evidence/20260927_loop078_bound/run404/gates.json").read_text())
    assert gate["status"] == "clean_frozen_diagnostic_gate" and gate["server_post_count"] == 60
    assert gate["cohorts"] == 5 and gate["ranks"] == 8 and gate["capture_files"] == gate["runtime_reports"] == 40
    assert gate["unique_runtime_request_ids"] == 60 and gate["source_sha_restored"] is True
    result = {
        "status": "configured_internal_zero_confirmed_external_publication_unclosed",
        "source": src,
        "run403_capture_tar_sha256": hashlib.sha256(capture_tar.read_bytes()).hexdigest(),
        "rank_cohort_pairs": len(pairs), "unique_runtime_req_ids": len(req_ids),
        "configured_internal_initial_output_counts_zero": True,
        "observed_run403_internal_initial_counts_zero": True,
        "handoff_before_current_model_forward": True,
        "external_pre_handoff_p_i_verified": False,
        "missing_direct_join": "Record ModelRunner CachedRequestState.output_token_ids length and Scheduler Request._output_token_ids length at same req_id handoff, plus API publication/SSE ID ledger before bulk return. Run403 route capture has cohort-local slots only; client result lacks direct ID join.",
        "conditional_DS7_cardinality_per_cohort_cycles": 128,
        "conditional_DS7_cardinality_five_cohorts_cycles": 640,
        "cardinality_scope": "Five Run403 diagnostic cohorts only if each had zero already-published external tokens and ordinary DSpark7 max8 new tokens per verification. Not Run239 four-cohort floor or a Product latency bound.",
        "run404_gate_path": "evidence/20260927_loop078_bound/run404/gates.json",
        "run404_gate_verified": True,
        "source_anchor_scope": "String occurrence and current mounted-file SHA only; active branch/control flow and historical Run403 loaded objects are not proved by this audit.",
        "finite_bound_promotion": False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({"status": result["status"], "rank_cohort_pairs": len(pairs), "external_p_i_verified": False}))


if __name__ == "__main__":
    main()
