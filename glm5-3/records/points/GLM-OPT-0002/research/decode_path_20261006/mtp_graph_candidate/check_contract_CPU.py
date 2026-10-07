"""Execute candidate AST predicates/contracts with CPU lifecycle doubles.

No torch/model import, HTTP or NPU. Device numerical/stream correctness remains
for the bounded real capture/replay diagnostic.
"""
import ast
from copy import deepcopy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace as NS

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "llm_base_proposer.py"
TREE = ast.parse(SOURCE.read_text())


class Tensor:
    def __init__(self, ptr, shape, data=None, dtype="int32", device="npu:0"):
        self.ptr, self.shape = ptr, shape
        self.data, self.dtype, self.device = list(data or []), dtype, device

    def data_ptr(self):
        return self.ptr

    def stride(self):
        if len(self.shape) == 1:
            return (1,)
        return (self.shape[1], 1)

    def numel(self):
        total = 1
        for size in self.shape:
            total *= size
        return total

    def __getitem__(self, index):
        assert isinstance(index, slice) and index.start is None
        return Tensor(self.ptr, (min(index.stop, self.shape[0]), *self.shape[1:]), self.data, self.dtype, self.device)

    def clone(self):
        return Tensor(self.ptr + 100000, self.shape, self.data, self.dtype, self.device)


class AscendSFADCPMetadata(NS):
    pass


MODES = NS(FULL="FULL", NONE="NONE")
STATES = NS(DecodeOnly="DecodeOnly", SpecDecoding="SpecDecoding")


def expression(predicate):
    return compile(ast.Expression(predicate), str(SOURCE), "eval")


def main():
    pc = NS(tensor_parallel_size=16, decode_context_parallel_size=16,
            prefill_context_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1)
    config = NS(model_config=NS(hf_config=NS(model_type="glm_moe_dsa")), parallel_config=pc,
                compilation_config=NS(cudagraph_mode=NS(has_full_cudagraphs=lambda: True)))
    proposer = NS(vllm_config=config, speculative_config=NS(draft_tensor_parallel_size=16),
                  use_cuda_graph=True, method="mtp", num_speculative_tokens=1, parallel_drafting=False,
                  _enable_probabilistic_draft_probs=False, use_compress=False, uses_mrope=False,
                  uses_xdrope_dim=0, supports_mm_inputs=False,
                  runner=NS(dynamic_eplb=False, input_batch=NS(generators={})))
    static_node = next(n.value for n in ast.walk(TREE) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Attribute) and t.attr == "_glm_k1_graph" for t in n.targets)
                       and isinstance(n.value, ast.BoolOp))
    env = dict(self=proposer, pc=pc, vllm_config=config, os=NS(environ={"VLLM_ASCEND_GLM_K1_MTP_GRAPH": "1"}),
               enable_sp=lambda *args: False, lmhead_tp_enable=lambda: False)
    static = expression(static_node)
    assert eval(static, env)
    static_rejections = 0
    for owner, field, changed in [(proposer, "method", "eagle"), (proposer, "num_speculative_tokens", 2),
                                  (proposer, "parallel_drafting", True), (proposer, "_enable_probabilistic_draft_probs", True),
                                  (proposer, "use_compress", True), (proposer, "uses_mrope", True),
                                  (proposer, "uses_xdrope_dim", 4), (proposer, "supports_mm_inputs", True),
                                  (pc, "tensor_parallel_size", 8), (pc, "decode_context_parallel_size", 8),
                                  (pc, "pipeline_parallel_size", 2), (pc, "data_parallel_size", 2),
                                  (proposer.runner, "dynamic_eplb", True), (proposer, "use_cuda_graph", False)]:
        previous = getattr(owner, field)
        setattr(owner, field, changed)
        assert not eval(static, env)
        setattr(owner, field, previous)
        static_rejections += 1
    env["os"].environ.clear()
    assert not eval(static, env)
    static_rejections += 1

    runtime_node = next(n.value for n in ast.walk(TREE) if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "use_call_graph" for t in n.targets)
                        and isinstance(n.value, ast.BoolOp))
    runtime = expression(runtime_node)
    combinations = 0
    for bits in itertools.product((False, True), repeat=10):
        uniform, lora, prefill, decode, shape, indices, greedy, history, generators, grammar = bits
        proposer.runner.input_batch.generators = {1: "fixture"} if generators else {}
        args = dict(self=proposer, use_call_graph=True, uniform_decode=uniform, has_lora=lora,
                    num_prefill_reqs=int(prefill), num_decode_reqs=int(decode), batch_size=1,
                    num_tokens=2 if shape else 1, token_indices_to_sample=Tensor(10, (1 if indices else 2,)),
                    sampling_metadata=NS(all_greedy=greedy, output_token_ids=[1] if history else []),
                    scheduler_output=NS(has_structured_output_requests=grammar))
        expected = uniform and not lora and not prefill and decode and shape and indices and greedy and not history and not generators and not grammar
        assert bool(eval(runtime, args)) == expected
        combinations += 1

    proposer.num_speculative_tokens = 1
    proposer.input_ids = Tensor(100, (512,))
    proposer.hidden_states = Tensor(200, (512, 8))
    positions = Tensor(300, (512,))
    proposer._get_positions = lambda count: positions[:count]
    proposer._glm_k1_layers = {"draft.attn": NS(kv_cache=[Tensor(400, (256, 128))],
                                               impl=NS(topk_indices_buffer=Tensor(500, (512, 32))))}
    impl = proposer._glm_k1_layers["draft.attn"].impl
    impl.indexer_cache = [Tensor(700, (256, 128)), Tensor(800, (256, 1))]
    impl._compose_sfa_kv_cache = lambda main: (*main, *impl.indexer_cache)
    context = NS(cudagraph_runtime_mode=MODES.FULL, batch_descriptor=NS(uniform=True, num_tokens=2))
    # Use a hashable descriptor, as vLLM BatchDescriptor is frozen.
    class Descriptor:
        uniform = True
        num_tokens = 2
    context.batch_descriptor = Descriptor()
    globals_dict = dict(torch=NS(Tensor=Tensor), CUDAGraphMode=MODES, AscendAttentionState=STATES,
                        get_forward_context=lambda: context)
    for name in ("_glm_k1_graph_contract", "_run_glm_k1_graph"):
        node = deepcopy(next(n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef) and n.name == name))
        node.decorator_list = []
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), globals_dict)
        setattr(proposer, name, lambda *a, _f=globals_dict[name], **k: _f(proposer, *a, **k))
    fields = ("cum_query_lens", "seq_lens", "slot_mapping", "pcp_slot_mapping", "block_table",
              "sin", "cos", "attn_mask", "group_len", "group_key_idx", "group_key_cache_idx")
    meta = AscendSFADCPMetadata(num_input_tokens=2, num_prefills=0, num_decodes=1, num_decode_tokens=2,
                               attn_state=STATES.SpecDecoding, head_dim=128, block_size=128,
                               num_actual_tokens=0, dcp_context=NS(kv_gather_block_ids=None, kv_gather_block_table=None))
    for i, field in enumerate(fields):
        setattr(meta, field, Tensor(1000 + i * 100, (2,)))
    for i, field in enumerate(("slot_mapping", "block_table", "seq_lens")):
        setattr(meta.dcp_context, field, Tensor(3000 + i * 100, (2,)))
    kwargs = dict(num_input_tokens=2, batch_size=1, token_indices_to_sample=Tensor(600, (1,)),
                  num_tokens=2, inputs_embeds=None, multi_steps_attn_metadata=[{"draft.attn": meta}])
    signature = proposer._glm_k1_graph_contract(kwargs)
    assert signature is not None
    fixed_output = Tensor(5000, (1, 1), [101])
    modes = []
    def acl(**kwargs):
        modes.append(context.cudagraph_runtime_mode)
        return fixed_output
    proposer._glm_k1_acl = acl
    proposer._glm_k1_contracts = {}
    first = proposer._run_glm_k1_graph(**kwargs)
    for field in fields:
        getattr(meta, field).data = [128, 129]  # values may change across blocks
    meta.num_actual_tokens = 2
    meta.attn_state = STATES.DecodeOnly
    assert proposer._glm_k1_graph_contract(kwargs) == signature
    fixed_output.data[0] = 202
    second = proposer._run_glm_k1_graph(**kwargs)
    assert first.data == [101] and second.data == [202] and first is not fixed_output
    pointer_fallbacks = 0
    for owner, field in [(meta, f) for f in fields] + [(meta.dcp_context, f) for f in ("slot_mapping", "block_table", "seq_lens")]:
        tensor = getattr(owner, field)
        tensor.ptr += 1
        assert proposer._run_glm_k1_graph(**kwargs) is fixed_output
        assert modes[-1] == MODES.NONE and context.cudagraph_runtime_mode == MODES.FULL
        tensor.ptr -= 1
        pointer_fallbacks += 1
    for tensor in [*proposer._glm_k1_layers["draft.attn"].kv_cache, impl.topk_indices_buffer, *impl.indexer_cache, kwargs["token_indices_to_sample"]]:
        tensor.ptr += 1
        assert proposer._run_glm_k1_graph(**kwargs) is fixed_output and modes[-1] == MODES.NONE
        tensor.ptr -= 1
        pointer_fallbacks += 1
    meta.num_prefills = 1
    assert proposer._run_glm_k1_graph(**kwargs) is fixed_output and modes[-1] == MODES.NONE
    meta.num_prefills = 0
    kwargs["token_indices_to_sample"].shape = (2,)
    assert proposer._run_glm_k1_graph(**kwargs) is fixed_output and modes[-1] == MODES.NONE
    result = dict(passed=True, CPU_only=True, NPU_or_model_import=False, HTTP_requests=0,
                  static_rejections=static_rejections, runtime_boolean_cases=combinations,
                  pointer_mismatch_eager_fallbacks=pointer_fallbacks,
                  dynamic_values_and_decode_states_match=True, output_owned_after_next_replay=True,
                  limitations="AST execution with tensor/ACL/stream doubles; actual capture, all-rank lifetimes and numerical correctness unverified.")
    (HERE / "CPU_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
