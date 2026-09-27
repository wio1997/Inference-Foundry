#!/usr/bin/env python3
"""Source-pinned conditional TP8 candidate-row map, not a runtime certificate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path("/data/wio/Inference_Foundry")
ASC = Path("/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend")
VLLM = Path("/data/wio/vllm_ascend_26/framework/vllm/vllm")
SOURCES = {
    ROOT / "runtime/fixed_decode.py": ("target_logits_indices=torch.arange(total", "target_query_start_loc=query_start"),
    ROOT / "runtime/target_adapter.py": ("sample_hidden = hidden_states[state.target_logits_indices]",),
    ROOT / "runtime/fixed_serving.py": ("def _park_completed", "self.runtime.state.active_mask[idx] = False"),
    ROOT / "runtime/extreme_decode.py": ("target_output = self.target.execute(self.state)",),
    ASC / "ops/register_custom_ops.py": ("tensor_model_parallel_all_gather(x, 0)", "x = x[:-pad_size]", "tensor_model_parallel_reduce_scatter(x, 0)"),
    ASC / "ops/fused_moe/prepare_finalize.py": ("maybe_all_gather_and_maybe_unpad(hidden_states, True, True)", "get_pcp_group().all_gather", "maybe_pad_and_reduce(hidden_states, True)"),
    ASC / "ops/fused_moe/token_dispatcher.py": ("npu_moe_init_routing", "npu_moe_token_unpermute"),
    ASC / "ops/vocab_parallel_embedding.py": ("dist.reduce_scatter_tensor(self._embed_rs_out_buf",),
    ASC / "ops/linear_op.py": ("class SequenceRowParallelOp", "matmul_and_reduce(input_parallel"),
    ASC / "models/deepseek_v4.py": ("hidden_states = sequence_parallel_chunk(hidden_states)", "tensor_model_parallel_all_gather(final_hidden_states, 0)"),
    ASC / "attention/context_parallel/dsa_cp.py": (".permute(1, 0, 2, 3)", "dist.all_to_all_single(recv, send", "local_attn_output, o_proj_full_handles"),
    ASC / "quantization/methods/w4a8.py": ("topk_weights, topk_ids = select_experts", "w1 = [layer.w13_weight]", "w2 = [layer.w2_weight]"),
    ASC / "compilation/acl_graph.py": ("entry.aclgraph.replay()",),
    ASC / "worker/model_runner_v1.py": ("target_logits_indices=torch.arange(", "target_graph_requested", "ExtremeHandoffInputs("),
    VLLM / "model_executor/models/utils.py": ("def sequence_parallel_chunk_impl(", "start = tp_rank * chunk"),
    VLLM / "distributed/parallel_state.py": ("def all_gather(self, input_", "def reduce_scatter(self, input_"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_identity(labels):
    assert labels == list(range(96)), "candidate-row map no longer slot-major"


def conditional_map(rank_order=tuple(range(8)), dsa_rank_order=None,
                    pad=0, logits_indices=None, dispatch_inverse=True,
                    layer_interleave=None):
    assert sorted(rank_order) == list(range(8))
    if dsa_rank_order is None:
        dsa_rank_order = rank_order
    assert sorted(dsa_rank_order) == list(range(8))
    # Nonzero padding is deliberately not represented by this idealized model.
    assert pad == 0, "actual padding/unpadding branch needs a runtime map"
    local = [list(range(12 * rank, 12 * (rank + 1))) for rank in range(8)]
    # Primitive model only: same group-relative order at RS split and AG join.
    gathered = [row for source_rank in rank_order for row in local[source_rank]]
    permutation = list(reversed(range(96)))
    sorted_rows = [gathered[i] for i in permutation]
    inverse = [permutation.index(i) for i in range(96)]
    restored = [sorted_rows[i] for i in inverse] if dispatch_inverse else sorted_rows
    # DSA primitive: query rows are local chunks; head-owner receiver joins
    # source chunks in group-relative order, conditional on the native ABI.
    restored_by_head_owner = [
        [row for source_rank in dsa_rank_order for row in local[source_rank]]
        for _ in range(8)
    ]
    if layer_interleave is not None:
        restored_by_head_owner = [
            [rows[i] for i in layer_interleave] for rows in restored_by_head_owner
        ]
    if logits_indices is None:
        logits_indices = list(range(96))
    sampled = [restored[index] for index in logits_indices]
    return dict(moe_prepare_rows=gathered, dispatcher_restored_rows=restored,
                dsa_restored_rows=restored_by_head_owner,
                target_logits_rows=sampled)


def self_test():
    normal = conditional_map()
    require_identity(normal["moe_prepare_rows"])
    require_identity(normal["dispatcher_restored_rows"])
    require_identity(normal["target_logits_rows"])
    for rows in normal["dsa_restored_rows"]:
        require_identity(rows)
    for kwargs in (dict(rank_order=(1, 0, 2, 3, 4, 5, 6, 7)),
                   dict(dsa_rank_order=(1, 0, 2, 3, 4, 5, 6, 7)),
                   dict(dispatch_inverse=False),
                   dict(layer_interleave=list(reversed(range(96)))),
                   dict(logits_indices=list(reversed(range(96))))):
        bad = conditional_map(**kwargs)
        try:
            require_identity(bad["moe_prepare_rows"])
            require_identity(bad["dispatcher_restored_rows"])
            for rows in bad["dsa_restored_rows"]:
                require_identity(rows)
            require_identity(bad["target_logits_rows"])
        except AssertionError:
            pass
        else:
            raise AssertionError(f"bad row map accepted: {kwargs}")
    try: conditional_map(pad=1)
    except AssertionError: pass
    else: raise AssertionError("unsupported nonzero padding accepted")
    print("Run573 conditional row-map negative gate PASS")


def main(out):
    self_test()
    source = {}
    for path, anchors in SOURCES.items():
        content = path.read_text()
        for anchor in anchors:
            assert anchor in content, (str(path), anchor)
        source[str(path)] = dict(sha256=digest(path), anchors=list(anchors))
    # The six borrowed Run403 sources must still be the source those route
    # snapshots executed under; additional row-order sources were not pinned
    # by that diagnostic and cannot retroactively certify its array indices.
    manifest = ROOT / "evidence/20260927_loop078_bound/run403/source_before.sha256"
    for line in manifest.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        assert digest(Path(name)) == expected
    result = dict(schema=2, status="source_index_and_idealized_primitives_not_live_ready",
        contract="DeepSeek V4 Flash W4A8 8x910B3 DP1TP8 DSpark7 fixed W0",
        source=source, run403_source_manifest_sha256=digest(manifest),
        conditional_result="Idealized primitive identity examples under contiguous12-row RS, matching group-relative AG/A2A order, zero pad and assumed native query-order; no per-layer Runtime composition or actual router-row certificate.",
        modelled_primitives=["contiguous 96-to-8x12 split and same-order concatenate",
                             "synthetic dispatcher permutation and exact inverse",
                             "synthetic DSA local query chunks and head-owner concatenate",
                             "synthetic target logits index selection"],
        unmodelled_nodes=["production ModelRunner->ExtremeHandoffInputs->FixedDecodeState.bind entry values",
                          "actual FlashComm1/sequence-parallel branch and nonzero padding/unpadding",
                          "DSA native sparse-attention output query order, A2A rank order and wo_b RS",
                          "43-layer query-row map composition and native MoE dispatcher semantics",
                          "physical Target/Draft expert weight format and group ownership"],
        runtime_fields_required=[
            "same cohort/rank/replay generation and startup FULL Graph key",
            "request-to-slot, active mask, Target input ids/positions/query boundaries and logits indices",
            "actual FlashComm1/SP/PCP/DP/EP branch flags, pad lengths and group global-rank ordering",
            "per-layer MoE prepare input/output query rows, router input pointer and exact pre-dispatch row map",
            "native sparse-attention output query-order contract and DSA A2A/wo_b RS send/recv row map",
            "Target/Draft expert weight shape/dtype/packed format/scale/owner",
            "full route/group/acceptance/parking source and output joins"],
        prohibited=["promote old Run403 retained-set array-index attribution to semantic token identity",
                    "treat clairvoyant retained-prefix work reduction as fixed-W0 target",
                    "infer compulsory HBM from unique packed-set footprint",
                    "claim numeric Resource/Scheduling/Product Bound"],
        formal_current_tps=571.681, finite_resource_floor_s=None,
        finite_scheduling_floor_s=None, finite_product_tps_ceiling=None)
    out.mkdir(parents=True, exist_ok=False)
    (out / "preflight.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(status=result["status"],source_files=len(source),runtime_fields=len(result["runtime_fields_required"]))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        assert args.out_dir is not None
        main(args.out_dir)
