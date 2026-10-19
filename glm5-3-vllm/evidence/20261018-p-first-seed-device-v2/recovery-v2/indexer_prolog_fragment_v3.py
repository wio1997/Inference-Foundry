"""Synthetic native producer/join/cache-write fragment; not a model benchmark."""
import json
import statistics
import time
from types import SimpleNamespace

import scipy.linalg
import torch
import torch_npu

from indexer_prolog_fragment_native_methods_v1 import (
    AscendW8A8DynamicLinearMethod,
    Indexer,
    Prolog,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.utils import enable_custom_op

torch.npu.set_device(0)
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = True
torch.manual_seed(191019)
assert enable_custom_op()
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
init_device_properties_triton()
torch.set_grad_enabled(False)
main = torch.npu.Stream()
aux = torch.npu.Stream()
torch.npu.set_stream(main)


def rand(shape, scale=1.0):
    return torch.randn(shape, dtype=torch.bfloat16, device="npu") * scale


def nz(shape):
    return torch_npu.npu_format_cast(
        torch.randint(-3, 4, shape, dtype=torch.int8, device="npu"), 29
    )


p = Prolog()
p.weight_dq = nz((6144, 2048))
p.weight_uq_qr = nz((2048, 1024))
p.weight_dkv_kr = nz((6144, 576))
assert all(torch_npu.get_npu_format(w) == 29 for w in [p.weight_dq, p.weight_uq_qr, p.weight_dkv_kr])
p.W_UK_T = rand((4, 192, 512), 0.02)
p.q_a_layernorm = SimpleNamespace(
    weight=torch.ones(2048, dtype=torch.bfloat16, device="npu"), variance_epsilon=1e-5
)
p.kv_a_layernorm = SimpleNamespace(
    weight=torch.ones(512, dtype=torch.bfloat16, device="npu"), variance_epsilon=1e-5
)
p._quant_type = AscendW8A8DynamicLinearMethod
p.enable_sparse_sfa_c8 = True
p.has_indexer = True
p.local_num_heads = 4
p.kv_lora_rank = 512
p.q_lora_rank = 2048
p.qk_rope_head_dim = 64
p.sfa_qsfa_tile_size = 128
p.sfa_qsfa_k_nope_clip_alpha = torch.ones(1, device="npu")
p.sfa_qsfa_kr_cache_dummy = torch.empty(0, dtype=torch.bfloat16, device="npu")
p.dequant_scale_w_dq = torch.full((1, 2048), 0.005, device="npu")
p.dequant_scale_w_uq_qr = torch.full((1, 1024), 0.005, device="npu")
p.dequant_scale_w_dkv_kr = torch.full((1, 576), 0.005, device="npu")

idx = Indexer()
idx.head_dim = 128
idx.qk_rope_head_dim = 64
idx.is_rope_neox_style = False
idx.enable_sparse_li_c4 = False
idx.c8_k_cache_dtype = torch.int8
idx.c8_k_scale_cache_dtype = torch.float16
idx.q_hadamard = torch.tensor(
    scipy.linalg.hadamard(128), dtype=torch.bfloat16, device="npu"
) / (128**0.5)
wk = rand((160, 6144), 0.02)
gamma = torch.ones(128, dtype=torch.float32, device="npu")
beta = torch.zeros(128, dtype=torch.float32, device="npu")
idx.wk_weights_proj = lambda x: (torch.nn.functional.linear(x, wk), None)
idx.k_norm = lambda x: torch.nn.functional.layer_norm(
    x.float(), [128], gamma, beta, 1e-6
).to(x.dtype)


def flatten(result):
    if isinstance(result, torch.Tensor):
        return [result]
    if isinstance(result, (tuple, list)):
        return [t for value in result for t in flatten(value)]
    assert result is None
    return []


def compare(a, b):
    aa, bb = flatten(a), flatten(b)
    assert len(aa) == len(bb)
    return [bool(torch.equal(u, v)) for u, v in zip(aa, bb)]


rows = []
for n in [6, 24]:
    for c8 in [False, True]:
        # Same original layer precision in both schedules. This is not an
        # attempt to quantize originally BF16 Indexer layers.
        idx.enable_sparse_li_c8 = c8
        idx.enable_sparse_li_quant = c8
        x = rand((n, 6144))
        positions = torch.arange(n, device="npu", dtype=torch.float32).view(-1, 1)
        freq = torch.arange(64, device="npu", dtype=torch.float32).view(1, -1) / 100
        cos = torch.cos(positions * freq).bfloat16().view(n, 1, 1, 64)
        sin = torch.sin(positions * freq).bfloat16().view(n, 1, 1, 64)
        slots = torch.arange(n, device="npu", dtype=torch.int64)
        slots32 = slots.to(torch.int32).view(-1, 1)
        cache = {}
        for mode in ["native", "parallel"]:
            cache[mode] = (
                torch.zeros((1085, 128, 1, 656), dtype=torch.int8, device="npu"),
                torch.zeros((138880, 128), dtype=torch.int8 if c8 else torch.bfloat16, device="npu"),
                torch.zeros((138880, 1), dtype=torch.float16, device="npu"),
            )

        def one(mode):
            kv, kc, sc = cache[mode]
            if mode == "parallel":
                aux.wait_stream(main)
                with torch.npu.stream(aux):
                    k, scale, weights = idx.forward_k(x, cos, sin)
            q = p._sfa_preprocess_prolog_v3(x, (kv,), cos, sin, slots)
            if mode == "native":
                k, scale, weights = idx.forward_k(x, cos, sin)
            else:
                main.wait_stream(aux)
            # Cache writes remain on the original stream, after the join.
            DeviceOperator.scatter_cache(kc, slots32, k.view(n, 128))
            if scale is not None:
                DeviceOperator.scatter_cache(sc, slots32, scale.view(n, 1))
            return q[1:], (k, scale, weights), (kv, kc, sc)

        eager_a = one("native")
        eager_b = one("parallel")
        torch.npu.synchronize()
        eager_equal = compare(eager_a, eager_b)
        assert all(eager_equal), {"N": n, "C8": c8, "eager_equal": eager_equal}
        graphs, outputs = {}, {}
        for mode in ["native", "parallel"]:
            for _ in range(3):
                one(mode)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            pool = torch.npu.graph_pool_handle()
            with torch.npu.graph(graph, pool=pool, stream=main):
                outputs[mode] = one(mode)
            graphs[mode] = (graph, pool)
        checks = []
        for generation in range(3):
            x.add_(0.125)
            reference = one("native")
            torch.npu.synchronize()
            for mode in ["native", "parallel", "parallel", "native"]:
                graphs[mode][0].replay()
                torch.npu.synchronize()
                equal = compare(reference, outputs[mode])
                checks.append({"generation": generation, "mode": mode, "equal": equal})
                assert all(equal), {"N": n, "C8": c8, "check": checks[-1]}
        timings = []
        for cycle in range(3):
            for mode in ["native", "parallel", "parallel", "native"]:
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record(main)
                for _ in range(40):
                    graphs[mode][0].replay()
                end.record(main)
                end.synchronize()
                timings.append({"cycle": cycle, "mode": mode, "us_per_replay": start.elapsed_time(end) * 1000 / 40})
        medians = {mode: statistics.median(z["us_per_replay"] for z in timings if z["mode"] == mode) for mode in ["native", "parallel"]}
        row = {"N": n, "Indexer_C8": c8, "eager_equal": eager_equal, "replay_checks": checks, "timings": timings, "median_us": medians, "parallel_over_native": medians["parallel"] / medians["native"]}
        rows.append(row)
        print(json.dumps({"progress": {k: v for k, v in row.items() if k not in ["replay_checks", "timings"]}}), flush=True)
        del outputs, graphs, cache, eager_a, eager_b, reference
        torch.npu.synchronize()
        torch.npu.empty_cache()

print("INDEXER_PROLOG_FRAGMENT_BEGIN\n" + json.dumps({"completed": True, "all_bitwise": True, "rows": rows, "peak_allocated_bytes": torch.npu.max_memory_allocated(), "scope": "Synthetic real dimensions and original exact-image method bodies, same BF16/C8 per-layer variants, main C8 KV. Separate ACL graph pools with changing inputs; original cache writes after join. Single NPU, no TP/CP/PD or full attention/Indexer top-k; not a full-model quality or service result."}) + "\nINDEXER_PROLOG_FRAGMENT_END", flush=True)
