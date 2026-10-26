#!/usr/bin/env python3
"""Read-only audit of device-side capture feasibility and Draft tail dependency."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROPOSER = Path("/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    serving = ROOT / "runtime/fixed_serving.py"
    runtime = ROOT / "runtime/extreme_decode.py"
    src = {"fixed_serving": serving.read_text(), "runtime": runtime.read_text(),
           "proposer": PROPOSER.read_text()}
    assert "count_history[index].copy_(self.runtime.state.num_sampled)" in src["fixed_serving"]
    assert "counts_cpu = count_history[:cycles].cpu()" in src["fixed_serving"]
    assert src["fixed_serving"].index("counts_cpu =") > src["fixed_serving"].index("# The last speculative preparation")
    assert "markov_emb = self.model.markov_embed(draft_token_ids[:, idx])" in src["proposer"]
    assert "logits_bias = self.model.markov_bias(markov_emb)" in src["proposer"]
    assert "draft_token_ids[:, idx + 1].copy_(logits[:, idx].argmax(dim=-1))" in src["proposer"]
    dspark = json.loads((ROOT / "evidence/20260927_loop076_bound/run372/analysis.json").read_text())
    windows = dspark["windows"]
    tails = [task for w in windows for task in w["tail_tasks"]]
    assert len(windows) == 16 and tails
    assert all(task["stream_id"] == "47" for task in tails)
    assert all(task["model_id"] == "4294967295" for task in tails)
    start = max(task["start_us_after_host_exit"] for task in tails)
    finish = max(task["start_us_after_host_exit"] + task["duration_us"] for task in tails)
    output = {
        "status": "read_only_loop077_capture_design_audit",
        "source_sha256": {key: digest(path) for key, path in
                          (("fixed_serving", serving), ("runtime", runtime), ("proposer", PROPOSER))},
        "existing_device_count_history": {
            "allocated_on_runtime_device": True,
            "staged_each_cycle_before_buffer_reuse": True,
            "copied_to_host_only_after_cohort": True,
            "proposal": "Export all cycle x slot counts from existing counts_cpu with cohort identity; derive useful clipped u_k from prior generated count and real initial_output_counts, keeping raw staged counts separate.",
        },
        "target_route_capture": {
            "run121_validated_ref_count": 86,
            "run121_live_ordinals": [43, 85],
            "proposal": "On selected cycles only, stack current device group_list refs[43:86] after target.execute to owned device storage; no synchronize/CPU/JSON in hot path. Export at cohort end; assert 43 rows x32 and cross-rank 576 routes/layer. Preallocate if the stack allocation affects cycle timing.",
            "risk": "Stack/copy is an extra current-stream operation; graph execution or borrowed side stream may require explicit producer event. Compare count and cycle distribution with uninstrumented A/A, and do not use captured cycles as unperturbed cost.",
        },
        "draft_serial_chain": {
            "source": str(PROPOSER),
            "edge": "draft[idx] -> markov_embed -> markov_bias -> add(raw_logits[idx]) -> argmax -> draft[idx+1]",
            "tail_tasks": len(tails),
            "latest_tail_start_us_after_host_scope_exit": start,
            "latest_tail_finish_us_after_host_scope_exit": finish,
            "stream_ids": sorted({x["stream_id"] for x in tails}),
            "model_ids": sorted({x["model_id"] for x in tails}),
            "bound_use": "Dependency edge only; Host scope exit is not Draft device completion or a removable 5ms Product gap.",
        },
        "next_gate": "Run low-impact selected-cycle route+counts acquisition without Level1 profiler or hot-path D2H, then separately measure all-rank device Draft-tail/next-Target join and resource contention with unperturbed legal cohort; do not claim finite bound before full compulsory-work and DAG closure.",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": output["status"], "tail_tasks": len(tails),
                      "max_tail_finish_us": finish}))


if __name__ == "__main__":
    main()
