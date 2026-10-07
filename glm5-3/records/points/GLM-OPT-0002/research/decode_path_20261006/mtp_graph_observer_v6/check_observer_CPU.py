"""Exercise real observer, production run and context AST on CPU.

No torch/model/device import or network. Filesystem/mmap are real; model,
metadata tensors and graph entry are doubles. This is driver integration only.
"""
import ast
import __future__
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "llm_base_proposer.py"
TREE = ast.parse(SOURCE.read_text())
COPIES = 0


class Tensor:
    def __init__(self, values, ptr=500):
        self.values, self.ptr = values, ptr
        self.shape = (1, 1)

    def to(self, device):
        global COPIES
        assert device == "cpu"
        COPIES += 1
        return self

    def tolist(self):
        return self.values

    def data_ptr(self):
        return self.ptr

    def __getitem__(self, index):
        return self

    def clone(self):
        return Tensor(self.values, self.ptr + 100)


class AscendSFADCPMetadata(NS):
    pass


class NoopOffloader:
    pass


class Descriptor:
    uniform = True
    num_tokens = 2


class ACLDouble:
    def __init__(self, descriptor, entry):
        self.concrete_aclgraph_entries = {descriptor: entry}
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        return Tensor([[101]], 600)


class Mode(str):
    def is_valid_runtime_mode(self):
        return self in ("NONE", "FULL")


def actual_context_types():
    base = HERE.parent / "stream_ordered_replay/original/forward_context.py"
    extra = HERE.parent / "stream_ordered_replay/original/ascend_forward_context.py"
    node = next(n for n in ast.parse(base.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "ForwardContext")
    env = dict(__name__=__name__, dataclass=dataclass, field=field, CUDAGraphMode=NS(NONE=Mode("NONE")))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(base), "exec", flags=__future__.annotations.compiler_flag), env)
    proxy = next(n for n in ast.parse(extra.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "_ExtraForwardContextProxy")
    exec(compile(ast.Module(body=[proxy], type_ignores=[]), str(extra), "exec", flags=__future__.annotations.compiler_flag), env)
    return env["ForwardContext"], env["_ExtraForwardContextProxy"], env


BASE_CONTEXT, EXTRA_PROXY, CONTEXT_ENV = actual_context_types()


def environment(root, rank):
    descriptor = Descriptor()
    context = BASE_CONTEXT(no_compile_layers={}, attn_metadata={}, slot_mapping={}, batch_descriptor=descriptor, cudagraph_runtime_mode=Mode("NONE"))
    CONTEXT_ENV["get_forward_context"] = lambda: context
    CONTEXT_ENV["envs_vllm"] = NS(VLLM_USE_V2_MODEL_RUNNER=False)
    proxy = EXTRA_PROXY()
    proxy.capturing = False
    try:
        proxy.num_actual_tokens
    except AttributeError:
        pass
    else:
        raise AssertionError("Actual proxy must reject the former fabricated field")
    assert not hasattr(context, "num_actual_tokens")
    signature = (2, 1, ("fixture persistent inputs",))
    output = Tensor([[101]], 600)
    entry = NS(aclgraph=object(), output=output)
    pc = NS(tensor_parallel_size=16, decode_context_parallel_size=16)
    positions = Tensor([10, 11])
    proposer = NS(_glm_k1_graph=True, use_cuda_graph=True, method="mtp", num_speculative_tokens=1,
                  speculative_config=NS(enforce_eager=False), use_compress=False,
                  _enable_probabilistic_draft_probs=False, parallel_drafting=False,
                  vllm_config=NS(parallel_config=pc, model_config=NS(hf_config=NS(model_type="glm_moe_dsa"))),
                  draft_attn_groups=[], attn_layer_names=["draft.attn"],
                  runner=NS(use_async_scheduling=True, input_batch=NS(sampling_metadata=NS(all_greedy=True))),
                  _glm_k1_contracts={descriptor: signature},
                  _glm_k1_acl=ACLDouble(descriptor, entry),
                  _glm_k1_graph_contract=lambda kwargs: signature,
                  _get_positions=lambda n: positions)
    env = dict(__file__=str(SOURCE), os=NS(environ={"GLM_MTP_GRAPH_DIAGNOSTIC_ROOT": str(root)} if root else {}, getpid=lambda: 1000+rank),
               get_offloader=lambda: NoopOffloader(), get_forward_context=lambda: context, get_tp_group=lambda: NS(rank_in_group=rank),
               _EXTRA_CTX=proxy,
               _H11_MAP=None, _H11_LAST=None, _H11_TRANSITION=0, _H11_ROWS=[],
               _H11_ORIGINAL_LOAD=lambda self, model: None,
               CUDAGraphMode=NS(FULL=Mode("FULL"), NONE=Mode("NONE")),
               enable_sp=lambda config: False, lmhead_tp_enable=lambda: False)
    production = ast.parse((HERE.parent / "mtp_graph_candidate_v5/llm_base_proposer.py").read_text())
    run = next(n for n in ast.walk(production) if isinstance(n, ast.FunctionDef) and n.name == "_run_glm_k1_graph")
    exec(compile(ast.Module(body=[run], type_ignores=[]), "actual_production_run_AST", "exec"), env)
    env["_H11_PRODUCTION_RUN"] = env["_run_glm_k1_graph"]
    names = ("_h11_mode", "_h11_observed_load", "_h11_observed_run")
    nodes = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == 3
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), env)
    metadata = AscendSFADCPMetadata(num_decodes=1, num_prefills=0, num_decode_tokens=2,
        seq_lens=Tensor([12]), slot_mapping=Tensor([10,11]),
        dcp_context=NS(slot_mapping=Tensor([10,11]), seq_lens=Tensor([12])))
    kwargs = dict(num_input_tokens=2, batch_size=1, multi_steps_attn_metadata=[{"draft.attn": metadata}])
    kwargs["token_indices_to_sample"] = Tensor([1])
    kwargs["sampling_metadata"] = proposer.runner.input_batch.sampling_metadata
    kwargs["is_prefill"] = False
    return env, proposer, context, metadata, positions, kwargs


def main():
    global COPIES
    before = COPIES
    rank_cases = 0
    dummy_cases = 0
    production_tree = ast.parse((HERE.parent / "mtp_graph_candidate_v5/llm_base_proposer.py").read_text())
    dummy = next(n for n in ast.walk(production_tree) if isinstance(n, ast.FunctionDef) and n.name == "dummy_run")
    call = next(n for n in ast.walk(dummy) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_runnable")
    dummy_keys = {k.arg for k in call.keywords}
    propose = next(n for n in ast.walk(production_tree) if isinstance(n, ast.FunctionDef) and n.name == "_propose")
    inputs = next(n.value for n in ast.walk(propose) if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "model_inputs")
    runtime_keys = {k.value for k in inputs.keys}
    assert "sampling_metadata" not in dummy_keys and "sampling_metadata" in runtime_keys
    with TemporaryDirectory(prefix="glm-mtp-observer-") as temporary:
        for rank in range(16):
            root = Path(temporary) / str(rank)
            root.mkdir()
            mode = root / "mtp_graph_mode.bin"
            mode.write_bytes(b"\x00")
            assert not (root / "witnesses").exists()
            env, proposer, context, metadata, positions, kwargs = environment(root, rank)
            env["_h11_observed_load"](proposer, object())
            cap = json.loads((root / "witnesses" / ("h11_capability_rank%d.json" % rank)).read_text())
            assert cap["offloader"] == "NoopOffloader"
            assert cap["eligible"] and cap["rank"] == rank and cap["pid"] == 1000+rank
            originals = {}
            for transition, value in enumerate((0, 1, 0), 1):
                with mode.open("r+b") as f:
                    f.write(bytes([value]))
                context.cudagraph_runtime_mode = "FULL" if value else "NONE"
                for step in range(2):
                    positions.values = [10+step, 11+step]
                    metadata.seq_lens.values = [12+step]
                    env["_h11_observed_run"](proposer, **kwargs)
                name = "h11_mode%d_transition%d_rank%d.json" % (value, transition, rank)
                path = root / "witnesses" / name
                row = json.loads(path.read_text())
                assert len(row["rows"]) == 2 and row["mode"] == value
                assert all(v["actual_replay"] == bool(value) for v in row["rows"])
                if value:
                    assert all(v["owned_output"] and v["detailed_snapshot"] for v in row["rows"])
                    assert row["rows"][0]["positions"] != row["rows"][1]["positions"]
                    assert row["rows"][0]["seq"] != row["rows"][1]["seq"]
                else:
                    assert all(v["fallback_reason"] == "runtime_none" and not v["detailed_snapshot"] and "draft_ids" not in v for v in row["rows"])
                original = path.read_bytes()
                copies = COPIES
                env["_h11_observed_run"](proposer, **kwargs)
                assert COPIES == copies and path.read_bytes() == original
                originals[path] = original
                assert all(p.read_bytes() == raw for p, raw in originals.items())
            env["_H11_MAP"].close()
            rank_cases += 1

        # Actual dummy/profile/capture kwargs omit sampling_metadata.
        for mode_value in (0, 1):
            for graph_mode in ("NONE", "FULL"):
                root = Path(temporary) / ("dummy%d%s" % (mode_value, graph_mode))
                root.mkdir()
                (root / "mtp_graph_mode.bin").write_bytes(bytes([mode_value]))
                env, proposer, context, metadata, positions, kwargs = environment(root, 0)
                context.cudagraph_runtime_mode = Mode(graph_mode)
                kwargs = {k: v for k, v in kwargs.items() if k in dummy_keys}
                copies = COPIES
                env["_h11_observed_run"](proposer, **kwargs)
                assert COPIES == copies and not (root / "witnesses").exists()
                assert env["_H11_LAST"] is None
                env["_H11_MAP"].close()
                dummy_cases += 1

        # Real request prefill is excluded from observation too.
        root = Path(temporary) / "runtime_prefill"
        root.mkdir()
        (root / "mtp_graph_mode.bin").write_bytes(b"\x01")
        env, proposer, context, metadata, positions, kwargs = environment(root, 0)
        kwargs["is_prefill"] = True
        copies = COPIES
        env["_h11_observed_run"](proposer, **kwargs)
        assert COPIES == copies and not (root / "witnesses").exists()
        env["_H11_MAP"].close()

        # Independently exercise runtime recording without calling load_model.
        root = Path(temporary) / "runtime_only"
        root.mkdir()
        (root / "mtp_graph_mode.bin").write_bytes(b"\x01")
        env, proposer, context, metadata, positions, kwargs = environment(root, 0)
        context.cudagraph_runtime_mode = "FULL"
        for _ in range(2):
            env["_h11_observed_run"](proposer, **kwargs)
        assert (root / "witnesses/h11_mode1_transition1_rank0.json").is_file()
        env["_H11_MAP"].close()

        env, proposer, context, metadata, positions, kwargs = environment(None, 0)
        copies = COPIES
        env["_h11_observed_run"](proposer, **kwargs)
        assert COPIES == copies and env["_H11_MAP"] is None

        fallback_cases = ("empty", "none_first", "non_sfa", "no_dcp", "signature_mismatch",
                          "descriptor_nonuniform", "descriptor_size", "none_descriptor", "none_contract_forbidden")
        for name in fallback_cases:
            root = Path(temporary) / name
            root.mkdir()
            (root / "mtp_graph_mode.bin").write_bytes(b"\x01")
            env, proposer, context, metadata, positions, kwargs = environment(root, 0)
            context.cudagraph_runtime_mode = Mode("FULL")
            if name in ("empty", "none_first", "non_sfa", "no_dcp"):
                values = {"empty": {}, "none_first": {"indexer": None, "draft": metadata},
                          "non_sfa": {"draft": object()},
                          "no_dcp": {"draft": AscendSFADCPMetadata(dcp_context=None)}}[name]
                kwargs["multi_steps_attn_metadata"] = [values]
                proposer._glm_k1_graph_contract = lambda kwargs: None
            elif name == "signature_mismatch":
                proposer._glm_k1_graph_contract = lambda kwargs: ("changed address",)
            elif name == "descriptor_nonuniform":
                context.batch_descriptor.uniform = False
            elif name == "descriptor_size":
                context.batch_descriptor.num_tokens = 4
            else:
                context.cudagraph_runtime_mode = Mode("NONE")
                if name == "none_descriptor":
                    context.batch_descriptor = None

                def forbidden_contract(kwargs):
                    raise AssertionError("NONE must never compute a graph contract")

                proposer._glm_k1_graph_contract = forbidden_contract
            copies = COPIES
            original_mode = context.cudagraph_runtime_mode
            for _ in range(2):
                result = env["_h11_observed_run"](proposer, **kwargs)
                assert result.tolist() == [[101]] and context.cudagraph_runtime_mode == original_mode
            assert proposer._glm_k1_acl.calls == 2 and COPIES == copies
            record = json.loads((root / "witnesses/h11_mode1_transition1_rank0.json").read_text())
            assert all(not row["actual_replay"] and not row["detailed_snapshot"] and row["fallback_reason"] and "draft_ids" not in row for row in record["rows"])
            env["_H11_MAP"].close()

    row = dict(passed=True, CPU_only=True, HTTP_or_model_requests=0,
               source_sha256=__import__("hashlib").sha256(SOURCE.read_bytes()).hexdigest(),
               rank_lifecycle_cases=rank_cases, transitions_per_rank=3,
               actual_ForwardContext_and_extra_proxy_AST=True, old_bad_attribute_rejected=True,
               actual_production_run_AST=True, eager_fallback_cases=len(fallback_cases),
               fallback_no_tensor_D2H=True, NONE_never_computes_contract=True,
               actual_dummy_vs_propose_kwargs_contract=True, dummy_profile_capture_cases=dummy_cases,
               prefill_observation_excluded=True,
               missing_directory_load_and_runtime=True, selector_mmap_is_real=True,
               heavy_CPU_copies_stop_after_two_rows=True, prior_witnesses_immutable=True,
               observed_cpu_copies=COPIES-before,
               limitations="Actual context dataclass/strict extra proxy and observer AST/filesystem/mmap; model/tensor/graph are doubles, no NPU replay or numerical claim.")
    (HERE / "CPU_result.json").write_text(json.dumps(row, indent=2)+"\n")
    print(json.dumps(row))


if __name__ == "__main__":
    main()
