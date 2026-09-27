#!/usr/bin/env python3
"""Source-pinned two-cycle dependency skeleton; no timing bound is inferred."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "runtime": ROOT / "runtime/extreme_decode.py",
    "proposer": ROOT / "bootstrap/vllm_dspark_handoff.py",
    "serving": ROOT / "runtime/fixed_serving.py",
    "graph": Path("/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/compilation/acl_graph.py"),
    "state": ROOT / "runtime/fixed_decode.py",
}
ANCHORS = {
    "runtime": [
        "self._state_machine.prepare_target_inputs()",
        "target_output = self.target.execute(self.state)",
        "acceptance_output = self.acceptance.execute(",
        "self._state_machine.advance_state(acceptance_output)",
        "next_draft = self.proposer.execute(",
        "self.state.draft_tokens.copy_(next_draft)",
    ],
    "proposer": [
        "self._host_copy_stream.wait_stream(current)",
        "self._host_count_copy.copy_(counts, non_blocking=True)",
        "self._host_copy_event.record()",
        "self._host_copy_event.synchronize()",
        "self._commit_host_mirrors()",
        "self._launch_host_count_copy(state.num_sampled)",
        "next_draft = self.proposer._propose(",
    ],
    "serving": [
        "result = self.runtime.step()",
        "token_history[index].copy_(result.acceptance.sampled_token_ids)",
        "count_history[index].copy_(self.runtime.state.num_sampled)",
        "progress = self._committed_progress()",
        "self._park_completed(progress)",
    ],
    "state": ["state.num_sampled.copy_("],
    "graph": [
        "need_sync = self.runtime_mode == CUDAGraphMode.FULL and not is_draft_eagle",
        "if not self.enable_enpu and need_sync:",
        "torch.npu.current_stream().synchronize()",
        "entry.aclgraph.replay()",
    ],
}

def read_sources():
    result = {}
    for name, path in SOURCES.items():
        raw = path.read_bytes()
        source = raw.decode()
        missing = [anchor for anchor in ANCHORS[name] if anchor not in source]
        if missing:
            raise RuntimeError(f"{name} source contract changed: {missing}")
        result[name] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                        "anchors": len(ANCHORS[name])}
    return result

def graph():
    nodes = [
        ("c_target_prepare", "current_stream", "Target input and metadata preparation"),
        ("c_graph_pre_replay_sync", "host_wait", "FULL Graph wrapper conditional current-stream synchronize"),
        ("c_target_graph_logits", "graph_and_current_stream", "Target FULL Graph plus logits binding"),
        ("c_acceptance", "current_stream", "Accept and update sampled counts"),
        ("c_state_advance", "current_stream", "Advance state and accepted output"),
        ("c_proposer_host_commit_prev", "host_wait", "Commit previous count-copy event before Draft consumer"),
        ("c_count_copy_launch", "side_stream", "Current count copy enqueued after wait_stream"),
        ("c_count_copy_complete", "side_stream", "Current count copy event has completed"),
        ("c_draft_prepare_model", "current_stream", "Draft prepare, body, seven Markov feedback steps"),
        ("c_draft_commit", "current_stream", "Commit draft tokens and increment cycle"),
        ("c_serving_stage", "current_stream", "Stage accepted tokens/counts before buffer reuse"),
        ("c_serving_progress_park", "host_and_current_stream", "Host mirror progress and optional park"),
        ("c1_target_prepare", "current_stream", "Next Target state and metadata preparation"),
        ("c1_graph_pre_replay_sync", "host_wait", "Next FULL Graph wrapper conditional current-stream synchronize"),
        ("c1_target_graph_logits", "graph_and_current_stream", "Next Target FULL Graph plus logits"),
        ("c1_acceptance", "current_stream", "Next acceptance and sampled counts"),
        ("c1_state_advance", "current_stream", "Next state advance"),
        ("c1_proposer_host_commit_c", "host_wait", "Wait for c count-copy event and update Draft Host mirror"),
        ("c1_draft_prepare_model", "current_stream", "Next Draft consumer"),
    ]
    edges = [
        ("c_target_prepare","c_graph_pre_replay_sync","source_order"),
        ("c_graph_pre_replay_sync","c_target_graph_logits","host_sync"),
        ("c_target_graph_logits","c_acceptance","data"),
        ("c_acceptance","c_state_advance","data"),
        ("c_state_advance","c_proposer_host_commit_prev","source_order"),
        ("c_proposer_host_commit_prev","c_count_copy_launch","host_order"),
        ("c_state_advance","c_count_copy_launch","side_stream_wait_on_current"),
        ("c_count_copy_launch","c_count_copy_complete","side_stream_service"),
        ("c_count_copy_launch","c_draft_prepare_model","host_order_only_side_copy_async"),
        ("c_state_advance","c_draft_prepare_model","data"),
        ("c_draft_prepare_model","c_draft_commit","data"),
        ("c_draft_commit","c_serving_stage","source_order"),
        ("c_serving_stage","c_serving_progress_park","source_order"),
        ("c_serving_progress_park","c1_target_prepare","source_order"),
        ("c1_target_prepare","c1_graph_pre_replay_sync","source_order"),
        ("c1_graph_pre_replay_sync","c1_target_graph_logits","host_sync"),
        ("c1_target_graph_logits","c1_acceptance","data"),
        ("c1_acceptance","c1_state_advance","data"),
        ("c1_state_advance","c1_proposer_host_commit_c","source_order"),
        ("c_count_copy_complete","c1_proposer_host_commit_c","side_stream_event_completion"),
        ("c1_proposer_host_commit_c","c1_draft_prepare_model","host_mirror_data"),
        ("c1_state_advance","c1_draft_prepare_model","data"),
    ]
    names = {name for name,_,_ in nodes}
    assert all(a in names and b in names for a,b,_ in edges)
    indegree = {name:0 for name in names}
    followers = {name:[] for name in names}
    for a,b,_ in edges:
        followers[a].append(b); indegree[b] += 1
    ready = [name for name,n in indegree.items() if n == 0]
    seen = 0
    while ready:
        node = ready.pop()
        seen += 1
        for nxt in followers[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
    assert seen == len(nodes), "dependency cycle"
    return {"nodes":[{"id":a,"resource":b,"meaning":c,"duration_floor_ms":None}
                     for a,b,c in nodes],
            "edges":[{"from":a,"to":b,"kind":c,"measured_delay_ms":None}
                     for a,b,c in edges],
            "conditional_edges":[{"from":"c_count_copy_complete",
                                  "to":"c_serving_progress_park",
                                  "condition":"park_completed_slots invokes _commit_host_mirrors for nonempty newly completed slots",
                                  "kind":"host_wait_on_current_copy_completion"}],
            "unproven_hazards":[{"reader":"c_count_copy_launch_to_complete",
                                 "writer":"c1_state_advance",
                                 "tensor":"state.num_sampled reused NPU buffer",
                                 "required_safe_order":"copy_complete(c) before next overwrite",
                                 "current_explicit_wait":"not found before c1 advance_state; normally large Target interval, but no source-level proof for arbitrary reschedule"}]}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    sources = read_sources()
    ledger = graph()
    result = {
        "status":"source_pinned_partial_two_cycle_dag",
        "edge_semantics":"Mixed host enqueue/source order and device completion dependencies. Do not run a finish-to-start longest-path solver on all edges. Only a source skeleton; split enqueue/wait-return/completion endpoints before numeric scheduling.",
        "conditional_context":{"graph_pre_replay_sync":"replay branch && runtime_mode FULL && !(is_draft_model && use_eagle) && !enable_enpu",
                               "next_target_metadata_schedule":"off/serial/overlap instantiation not established by this source skeleton; overlap adds private-stream edges",
                               "parking":"conditional edges apply only when new slots park"},

        "sources":sources,
        **ledger,
        "observed_run395_current_stream_median_ms":{
            "begin64_to_begin65":56.54833984375,
            "pre_target_to_target":49.46484375,
            "state_advance_to_proposer":6.39697265625,
            "source":"Run395 clean sparse events; instrumented observed intervals, not node floors"},
        "gates_before_finite_bound":[
            "Graph internal/HCCL producer and completion events plus all8 collective arrival",
            "Host copy side-stream record/complete and next Draft consumer timestamp",
        "Copy-complete before state.num_sampled buffer reuse, including parking-conditional synchronous wait",
            "Per-rank near-in-time clock calibration and stream identity",
            "Sparse marker A/A overhead; source schedule mode and enable_enpu runtime value",
            "Complete per-node compulsory work/traffic and mixed attainable capacities",
            "Legal prefill/seed/arrival/output drain and held-out E2E calibration",
        ],
        "scope":"Source-order/data skeleton only. No duration lower bound or all8 makespan. The current-stream stage medians are not added as disjoint costs."
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"status":result["status"],"nodes":len(ledger["nodes"]),"edges":len(ledger["edges"])}))

if __name__ == "__main__":
    main()
