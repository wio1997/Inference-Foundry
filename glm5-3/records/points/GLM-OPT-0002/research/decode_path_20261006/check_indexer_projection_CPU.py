"""Run the actual SFA AST on CPU tensors with CP/cache/NPU call doubles.

This checks forward-local projection reuse and fallback wiring. Native RoPE,
indexer, cache writes and collectives still require actual model correctness.
"""
import ast
from pathlib import Path
import functools
import hashlib
import json
from types import SimpleNamespace as NS
import torch

HERE = Path(__file__).resolve().parent
FILES = [HERE / "vllm-ascend__vllm_ascend__attention__sfa_v1.py", HERE / "indexer_candidate/sfa_v1.py"]
torch.set_num_threads(1)
assert not torch.npu.is_initialized() if hasattr(torch, "npu") else True


class Preprocess:
    NATIVE, PROLOG_V3, MLAPO = range(3)


class Attention:
    DecodeOnly, SpecDecoding, Prefill = range(3)


def methods(path):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendSFAImpl")
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in
             {"indexer_select_pre_process", "indexer_select_post_process", "forward"}]
    ns = dict(torch=torch, HAS_TRITON=True, PreprocessType=Preprocess,
              AscendAttentionState=Attention, MLAPO_MAX_SUPPORTED_TOKENS=4096,
              rope_forward_triton_siso=lambda tensor, *args, **kwargs: tensor,
              wait_for_kv_layer_from_connector=lambda *args: None,
              notify_kv_cache_written=lambda *args: None,
              record_attention_compute_start=lambda: None,
              maybe_save_kv_layer_to_connector=lambda *args: None)
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), ns)
    return ns


def run(ns, x, wk, qb, mode, has_indexer, skip_topk, state):
    calls = []
    s = NS(head_dim=128, n_head=64, qk_rope_head_dim=4, is_rope_neox_style=False,
           enable_sparse_li_c8=False, preprocess_type=mode, has_indexer=has_indexer,
           skip_topk=skip_topk, q_lora_rank=8, kv_lora_rank=4, use_index_cache=False,
           use_torch_npu_lightning_indexer=True, layer_name="test",
           vllm_config=NS(parallel_config=NS(prefill_context_parallel_size=1)))
    saved = {}

    def projection(t):
        calls.append(t.clone())
        return t @ wk, None

    s.wk_weights_proj = projection
    s.k_norm = lambda t: torch.nn.functional.layer_norm(t, (128,))
    s.wq_b = lambda t: (t @ qb, None)
    s.fused_qkv_a_proj = lambda t: (torch.cat([t[:, :8], t[:, :8]], -1), None)
    s.q_a_layernorm = lambda t: t.clone()
    s._compose_sfa_kv_cache = lambda value: value
    s._get_sfa_kv_slot_mapping = lambda metadata: metadata.slot_mapping
    s._get_parallel_forward_context = lambda *args: NS(actual_seq_lengths_query=torch.tensor([len(x)]),
        actual_seq_lengths_key=torch.tensor([len(x)]), gather_full_o_proj=False, topk_num_tokens=len(x),
        kv_slot_mapping=torch.arange(len(x)))
    s._prepare_native_hidden_states = lambda t, metadata: t
    s.exec_kv = lambda *args: (None, None)
    s._prepare_kv_for_parallel = lambda kpe, knope, scale, kli, kscale, full: (kli, kscale, None, [])
    s._q_proj_and_k_up_proj = lambda qc: (qc, qc)
    s.rope_single = lambda q, *args: q
    s._record_query_gather_context = lambda *args: None
    s._store_parallel_kv = lambda kpe, knope, scale, kli, *args: (kpe, knope, kli)
    s._write_indexer_cache = lambda kli, *args: saved.update(k=kli.clone())
    s._get_indexcache_topk_indices = lambda n: torch.zeros((n, 1, 2), dtype=torch.int32)
    s._execute_sparse_flash_attention_process = lambda *args: torch.ones((len(x), 1, 4), dtype=x.dtype)
    s._v_up_proj = lambda value: value
    s._finalize_o_proj = lambda value, out, full: value

    # Deliberately replace fused-path hidden_states, to ensure the new local kw
    # remains None there and post_process uses the updated input.
    def fused(**kwargs):
        value = kwargs["hidden_states"] * 2
        return value, value[:, :4], value[:, :4], value[:, :8]

    s._sfa_preprocess_prolog_v3 = s._sfa_preprocess_mlapo = fused

    def indexer(impl, q, qscale, original_shape, weights, *args):
        saved.update(q=q.clone(), weights=weights.clone())
        return torch.zeros((len(q), 1, 2), dtype=torch.int32)

    ns["DeviceOperator"] = NS(indexer_select_post_process=indexer)
    for name in ("indexer_select_pre_process", "indexer_select_post_process"):
        setattr(s, name, functools.partial(ns[name], s))
    meta = NS(cos=torch.ones((len(x), 4)), sin=torch.ones((len(x), 4)), slot_mapping=torch.arange(len(x)),
              num_input_tokens=len(x), attn_state=state)
    kv_cache = (torch.zeros((1, 1, 4)), torch.zeros((1, 1, 4)), torch.zeros((1, 1, 128)))
    before = x.clone()
    out = ns["forward"](s, "test", x, kv_cache, meta, output=torch.empty((len(x), 4)))
    assert torch.equal(x, before), "input mutated"
    return saved, calls, out


torch.manual_seed(17)
namespaces = [methods(path) for path in FILES]
cases = []
for dtype in (torch.bfloat16, torch.float16):
    wk = torch.randn(32, 192).to(dtype)
    qb = torch.randn(8, 8192).to(dtype)
    for tokens in (1, 2, 8):
        for mode, has_indexer, skip, state in [
            (Preprocess.NATIVE, True, False, Attention.DecodeOnly),
            (Preprocess.NATIVE, True, False, Attention.SpecDecoding),
            (Preprocess.NATIVE, True, False, Attention.Prefill),
            (Preprocess.NATIVE, True, True, Attention.DecodeOnly),
            (Preprocess.NATIVE, False, True, Attention.DecodeOnly),
            (Preprocess.PROLOG_V3, True, False, Attention.DecodeOnly),
            (Preprocess.MLAPO, True, False, Attention.SpecDecoding),
        ]:
            x = torch.randn(tokens, 32).to(dtype)
            a, b = [run(ns, x, wk, qb, mode, has_indexer, skip, state) for ns in namespaces]
            assert a[0].keys() == b[0].keys()
            assert all(torch.equal(a[0][name], b[0][name]) for name in a[0])
            assert torch.equal(a[2], b[2])
            removed = int(mode == Preprocess.NATIVE and has_indexer and not skip)
            assert len(a[1]) - len(b[1]) == removed
            if removed:
                assert torch.equal(a[1][0], a[1][1]) and torch.equal(a[1][0], b[1][0])
            cases.append(dict(dtype=str(dtype), tokens=tokens, mode=mode, has_indexer=has_indexer,
                              skip_topk=skip, state=state, original_projection_calls=len(a[1]),
                              candidate_projection_calls=len(b[1]), K_Q_weights_bitwise_equal=True))

result = dict(CPU_only=True, NPU_initialized=False, cases=cases, cases_passed=len(cases),
              source_identity={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in FILES},
              limits=["Actual helper and forward AST, CPU BF16/FP16 projection and layer norm; NPU RoPE/indexer/cache/collective calls doubled.",
                      "No cached tensor across calls/layers. Each case supplies fresh x and dynamic batch length.",
                      "Native all-rank byte checks and golden PD still required before matched performance comparison."])
print(json.dumps(result))
