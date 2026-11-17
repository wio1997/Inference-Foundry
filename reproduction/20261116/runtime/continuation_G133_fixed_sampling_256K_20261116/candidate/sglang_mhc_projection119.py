"""Unmodified functions extracted from pinned SGLang GPU mHC implementation."""
import torch
import triton
import triton.language as tl

@triton.jit
def _hc_mix_stats_reduce_kernel(
    part_mix_ptr,
    part_sq_ptr,
    mixes_ptr,
    M,
    inv_k,
    eps,
    MIX: tl.constexpr,
    MIX_PAD: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Sum the NUM_SLICES partials in slice order and apply the rms scaling."""
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, MIX_PAD)
    mask_m = offs_m < M
    mask_n = offs_n < MIX
    acc = tl.zeros([BLOCK_M, MIX_PAD], dtype=tl.float32)
    sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    for s in tl.static_range(NUM_SLICES):
        acc += tl.load(
            part_mix_ptr + (s * M + offs_m[:, None]) * MIX + offs_n[None, :],
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )
        sq += tl.load(part_sq_ptr + s * M + offs_m, mask=mask_m, other=0.0)
    rsqrt = 1.0 / tl.sqrt(sq * inv_k + eps)
    tl.store(
        mixes_ptr + offs_m[:, None] * MIX + offs_n[None, :],
        acc * rsqrt[:, None],
        mask=mask_m[:, None] & mask_n[None, :],
    )


def split_bf16_hc_weight(weight: torch.Tensor):
    assert weight.dtype == torch.float32 and weight.is_contiguous()
    high = weight.bfloat16()
    residual = weight - high.float()
    middle = residual.bfloat16()
    low = (residual - middle.float()).bfloat16()
    return high, middle, low


@triton.jit
def _hc_mix_stats_bf16x3_kernel(
    X,
    W_HI,
    W_MID,
    W_LO,
    MIX,
    SQ,
    M,
    K: tl.constexpr,
    K_PER_SLICE: tl.constexpr,
    MIX_COLS: tl.constexpr,
    MIX_PAD: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # M stays runtime-valued so variable prefill lengths reuse the same binary.
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, MIX_PAD)
    start = tl.program_id(1) * K_PER_SLICE
    ks = start + tl.arange(0, BLOCK_K)
    hi = tl.zeros((BLOCK_M, MIX_PAD), tl.float32)
    mid = tl.zeros((BLOCK_M, MIX_PAD), tl.float32)
    lo = tl.zeros((BLOCK_M, MIX_PAD), tl.float32)
    sq = tl.zeros((BLOCK_M,), tl.float32)
    for block in range(K_PER_SLICE // BLOCK_K):
        k = ks + block * BLOCK_K
        x = tl.load(
            X + rows[:, None].to(tl.int64) * K + k[None, :],
            rows[:, None] < M,
            0,
        )
        offsets = cols[None, :] * K + k[:, None]
        w_hi = tl.load(W_HI + offsets, cols[None, :] < MIX_COLS, 0)
        w_mid = tl.load(W_MID + offsets, cols[None, :] < MIX_COLS, 0)
        w_lo = tl.load(W_LO + offsets, cols[None, :] < MIX_COLS, 0)
        hi = tl.dot(x, w_hi, hi)
        mid = tl.dot(x, w_mid, mid)
        lo = tl.dot(x, w_lo, lo)
        xf = x.to(tl.float32)
        sq += tl.sum(xf * xf, 1)
    offsets = (tl.program_id(1) * M + rows[:, None]) * MIX_COLS + cols[None, :]
    tl.store(
        MIX + offsets, (hi + mid) + lo, (rows[:, None] < M) & (cols[None, :] < MIX_COLS)
    )
    tl.store(SQ + tl.program_id(1) * M + rows, sq, rows < M)


"""NPU caller for existing compensated projection; original FP32 stats contract."""
import torch
import triton


def project(x, weight_parts, eps):
    assert x.device.type=='npu' and x.dtype==torch.bfloat16 and x.is_contiguous()
    assert x.ndim==2 and x.shape[1]==20480 and len(weight_parts)==3
    assert all(w.shape==(24,20480) and w.dtype==torch.bfloat16 and w.is_contiguous() and w.device==x.device for w in weight_parts)
    m,k=x.shape
    mixes=torch.empty((m,24),device=x.device,dtype=torch.float32)
    if m==0:return mixes
    # Existing GPU implementation constants, preserved for one eligibility run.
    slices,block_m,block_k=16,128,64
    partial=torch.empty((slices,m,24),device=x.device,dtype=torch.float32)
    sq=torch.empty((slices,m),device=x.device,dtype=torch.float32)
    _hc_mix_stats_bf16x3_kernel[(triton.cdiv(m,block_m),slices)](
        x,*weight_parts,partial,sq,m,K=k,K_PER_SLICE=k//slices,
        MIX_COLS=24,MIX_PAD=32,BLOCK_K=block_k,BLOCK_M=block_m,
        num_warps=4,num_stages=3,
    )
    reduce_m=32
    _hc_mix_stats_reduce_kernel[(triton.cdiv(m,reduce_m),)](
        partial,sq,mixes,m,1.0/k,eps,MIX=24,MIX_PAD=32,
        NUM_SLICES=slices,BLOCK_M=reduce_m,num_warps=4,
    )
    return mixes


# Weight tensors remain FP32 in the model. Extra3BF16parts are cached once per
# original parameter and retain all three terms; runtime public coefficients
# are checked against the original FP32 projection before formal scoring.
import datetime,hashlib,json,os
from pathlib import Path
_PARTS={}
_VERIFIED={}
_DIRECTORY=Path('/task/sglang_penalty133/projection_hits')

def weight_parts_for_layer(fn):
    key=fn.data_ptr()
    if key not in _PARTS:
        if (_DIRECTORY/'FROZEN.json').exists():
            raise RuntimeError('Unseen projection parameter after formal freeze')
        assert not torch.npu.is_current_stream_capturing(), 'Cold HC parts inside capture'
        _PARTS[key]=split_bf16_hc_weight(fn)
    return _PARTS[key]

def project_for_layer(x,fn,eps):
    return project(x,weight_parts_for_layer(fn),eps)

def verify_once(hc,x,outputs,reference):
    key=hc.fn.data_ptr()
    if key in _VERIFIED:return
    if (_DIRECTORY/'FROZEN.json').exists():
        raise RuntimeError('Unverified projection parameter after formal freeze')
    expected=reference()
    errors=[float((a-b).abs().max()) for a,b in zip(outputs,expected)]
    passed=all(torch.allclose(a,b,atol=2e-5,rtol=2e-5) for a,b in zip(outputs,expected))
    rank=torch.distributed.get_rank();_DIRECTORY.mkdir(exist_ok=True)
    item={'parameter_ptr':key,'rows':x.shape[0],'actual_input_public_coefficients_passed':passed,'max_abs':errors,'atol_rtol':[2e-5,2e-5]}
    _VERIFIED[key]=item
    value={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rank':rank,'pid':os.getpid(),'verified_parameter_count':len(_VERIFIED),'cached_parameter_count':len(_PARTS),'persistent_weightparts_bytes':sum(v.numel()*v.element_size() for parts in _PARTS.values() for v in parts),'all_checks_strict_passed':all(v['actual_input_public_coefficients_passed'] for v in _VERIFIED.values()),'entries':list(_VERIFIED.values()),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'known_parameter_shadow_disabled_after_record':True,'unseen_parameter_after_FROZEN':'fail_closed_without_inference_shadow'}
    path=_DIRECTORY/f'rank{rank}.json';temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
    assert passed,errors
    if len(_VERIFIED)==80:print('G119_PROJECTION_ACTUAL80_PUBLIC_OUTPUTS_PASS '+json.dumps({'rank':rank,'pid':os.getpid(),'parameters':80}),flush=True)
