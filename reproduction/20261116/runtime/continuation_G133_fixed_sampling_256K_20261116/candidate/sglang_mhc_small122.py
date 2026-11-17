"""Qualified small-row HC coefficients; cold work outside NPU graph capture."""
import torch
import triton
import triton.language as tl
from sglang_mhc_projection119 import _hc_mix_stats_bf16x3_kernel, weight_parts_for_layer, _PARTS
@triton.jit
def _hc_mix_reduce_sinkhorn_kernel(
    part_mix_ptr,
    part_sq_ptr,
    scale_ptr,
    base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    m,
    inv_k,
    rms_eps,
    MIX: tl.constexpr,
    HC: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    ITERS: tl.constexpr,
    EPS: tl.constexpr,
    part_mix_residual_ptr=None,
):
    """One CTA per row keeps the sinkhorn reductions two-dimensional."""
    row = tl.program_id(0)
    if row >= m:
        return
    j = tl.arange(0, HC)
    jj = j[:, None]
    kk = j[None, :]

    a_pre = tl.zeros([HC], dtype=tl.float32)
    a_post = tl.zeros([HC], dtype=tl.float32)
    a_comb = tl.zeros([HC, HC], dtype=tl.float32)
    sq = tl.zeros([], dtype=tl.float32)
    for s in tl.static_range(NUM_SLICES):
        off = (s * m + row) * MIX
        v_pre = tl.load(part_mix_ptr + off + j)
        v_post = tl.load(part_mix_ptr + off + HC + j)
        v_comb = tl.load(part_mix_ptr + off + 2 * HC + jj * HC + kk)
        if part_mix_residual_ptr is not None:
            v_pre += tl.load(part_mix_residual_ptr + off + j)
            v_post += tl.load(part_mix_residual_ptr + off + HC + j)
            v_comb += tl.load(part_mix_residual_ptr + off + 2 * HC + jj * HC + kk)
        a_pre += v_pre
        a_post += v_post
        a_comb += v_comb
        sq += tl.load(part_sq_ptr + s * m + row)
    rsqrt = 1.0 / tl.sqrt(sq * inv_k + rms_eps)

    s0 = tl.load(scale_ptr + 0)
    s1 = tl.load(scale_ptr + 1)
    s2 = tl.load(scale_ptr + 2)

    pre = tl.sigmoid(a_pre * rsqrt * s0 + tl.load(base_ptr + j)) + EPS
    tl.store(pre_ptr + row * HC + j, pre)
    post = 2.0 * tl.sigmoid(a_post * rsqrt * s1 + tl.load(base_ptr + HC + j))
    tl.store(post_ptr + row * HC + j, post)

    comb = a_comb * rsqrt * s2 + tl.load(base_ptr + 2 * HC + jj * HC + kk)
    comb = tl.exp(comb - tl.max(comb, axis=1)[:, None])
    comb = comb / tl.sum(comb, axis=1)[:, None] + EPS
    comb = comb / (tl.sum(comb, axis=0)[None, :] + EPS)
    for _ in tl.static_range(ITERS - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + EPS)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + EPS)
    tl.store(comb_ptr + row * HC * HC + jj * HC + kk, comb)

"""Fixed caller qualified by G121; model performance remains unverified."""
import torch
import triton

FOOTPRINTS = (5, 6, 10, 12, 15, 18, 20, 24)

def fused_pre(x, weight_parts, scale, base, rms_eps=1e-20, hc_eps=1e-6):
    assert x.device.type == 'npu' and x.dtype == torch.bfloat16 and x.is_contiguous()
    assert x.ndim == 2 and x.shape[0] in FOOTPRINTS and x.shape[1] == 20480
    assert len(weight_parts) == 3
    assert all(w.shape == (24, 20480) and w.dtype == torch.bfloat16 and w.device == x.device and w.is_contiguous() for w in weight_parts)
    assert scale.numel() == 3 and base.numel() == 24
    assert scale.dtype == base.dtype == torch.float32 and scale.device == base.device == x.device
    assert scale.is_contiguous() and base.is_contiguous()
    m, k = x.shape
    partial = torch.empty((16, m, 24), dtype=torch.float32, device=x.device)
    square = torch.empty((16, m), dtype=torch.float32, device=x.device)
    pre = torch.empty((m, 4), dtype=torch.float32, device=x.device)
    post = torch.empty_like(pre)
    comb = torch.empty((m, 4, 4), dtype=torch.float32, device=x.device)
    # Fixed32-row tile for actual5..24-row footprints; no geometry grid.
    _hc_mix_stats_bf16x3_kernel[(triton.cdiv(m, 32), 16)](
        x, *weight_parts, partial, square, m, K=k, K_PER_SLICE=k//16,
        MIX_COLS=24, MIX_PAD=32, BLOCK_K=64, BLOCK_M=32,
        num_warps=4, num_stages=3)
    _hc_mix_reduce_sinkhorn_kernel[(m,)](
        partial, square, scale, base, pre, post, comb, m, 1.0/k, rms_eps,
        MIX=24, HC=4, NUM_SLICES=16, ITERS=20, EPS=hc_eps,
        num_warps=4)
    return pre, post, comb


import datetime,hashlib,json,os
from pathlib import Path
_DIRECTORY=Path('/task/sglang_penalty133/smallHC_hits')
_QUALIFIED={}
_CAPTURED={}

def _record():
    _DIRECTORY.mkdir(exist_ok=True)
    rank=torch.distributed.get_rank()
    value={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rank':rank,'pid':os.getpid(),
           'qualified_parameter_count':len(_QUALIFIED),'captured_parameter_count':len(_CAPTURED),
           'qualified':list(_QUALIFIED.values()),'captured':list(_CAPTURED.values()),
           'cached_parameter_count':len(_PARTS),
           'persistent_shared_weightparts_bytes':sum(v.numel()*v.element_size() for parts in _PARTS.values() for v in parts),
           'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'matmul_allow_hf32':torch.npu.matmul.allow_hf32,
           'qualification_scope':'Startup outside-capture operand plus deterministic nonzero probe, actual loaded HC parameters; not AISBench output quality',
           'host_checks_outside_capture':True,'unseen_after_freeze':'fail_closed'}
    path=_DIRECTORY/f'rank{rank}.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def run_for_layer(hc,x,cfg,reference):
    key=hc.fn.data_ptr()
    first=key not in _QUALIFIED
    if first:
        assert not torch.npu.is_current_stream_capturing(), 'Cold HC qualification inside capture'
        assert not (_DIRECTORY/'FROZEN.json').exists(), 'Unknown HC parameter after formal freeze'
        assert torch.npu.matmul.allow_hf32 is False, 'Component/serving precision mismatch'
    parts=weight_parts_for_layer(hc.fn)
    outputs=fused_pre(x,parts,hc.scale,hc.base,cfg.rms_eps,cfg.eps)
    if first:
        expected=reference(x)
        actual_pass=all(torch.allclose(a,b,atol=2e-5,rtol=2e-5) for a,b in zip(outputs,expected))
        actual_errors=[float((a-b).abs().max()) for a,b in zip(outputs,expected)]
        probe=(torch.arange(5*20480,device=x.device,dtype=torch.float32).remainder_(31).sub_(15).div_(8)).bfloat16().reshape(5,20480)
        observed=fused_pre(probe,parts,hc.scale,hc.base,cfg.rms_eps,cfg.eps)
        expected=reference(probe)
        probe_pass=all(torch.allclose(a,b,atol=2e-5,rtol=2e-5) for a,b in zip(observed,expected))
        probe_errors=[float((a-b).abs().max()) for a,b in zip(observed,expected)]
        _QUALIFIED[key]={'parameter_ptr':key,'startup_rows':x.shape[0],
                         'startup_public_passed':actual_pass,'startup_max_abs':actual_errors,
                         'nonzero_probe_public_passed':probe_pass,'nonzero_probe_max_abs':probe_errors,
                         'atol_rtol':[2e-5,2e-5],'outside_capture':True}
        _record()
        assert actual_pass and probe_pass, _QUALIFIED[key]
        if len(_QUALIFIED)==86:print('G127_SMALL_HC_ACTUAL86_PUBLIC_PASS',flush=True)
    if key not in _CAPTURED and torch.npu.is_current_stream_capturing():
        _CAPTURED[key]={'parameter_ptr':key,'rows':x.shape[0]}
        _record()
        if len(_CAPTURED)==86:print('G127_SMALL_HC_ACTUAL86_GRAPH_CAPTURED',flush=True)
    return outputs
