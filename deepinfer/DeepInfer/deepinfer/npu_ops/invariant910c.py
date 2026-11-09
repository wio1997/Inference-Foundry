"""Batch-independent NPU arithmetic for the original invariant-mode contract.

The grid may change with the batch, but a row's reduction tree never does.
These are numerical leaf implementations, not request padding or scheduling.
"""
import torch
import triton
import triton.language as tl


def project(x, weight):
    """Select the numerical leaf without changing the default execution path."""
    from deepinfer import envs
    if x.device.type == 'npu' and envs.DEEPINFER_BATCH_INVARIANT:
        return linear(x, weight)
    return torch.nn.functional.linear(x, weight)


@triton.jit
def _rms_norm(X, WEIGHT, FLOAT, BF16, WIDTH: tl.constexpr, EPS: tl.constexpr,
              BK: tl.constexpr = 1024):
    row = tl.program_id(0)
    columns = tl.arange(0, BK)
    squares = tl.full((), 0, tl.float32)
    for start in range(tl.cdiv(WIDTH, BK)):
        index = start*BK+columns
        value = tl.load(X+row*WIDTH+index,index<WIDTH,0).to(tl.float32)
        squares += tl.sum(value*value,0)
    rstd = tl.rsqrt(squares/WIDTH+EPS)
    for start in range(tl.cdiv(WIDTH, BK)):
        index = start*BK+columns
        value = tl.load(X+row*WIDTH+index,index<WIDTH,0).to(tl.float32)
        weight = tl.load(WEIGHT+index,index<WIDTH,0).to(tl.float32)
        normalized = value*rstd*weight
        tl.store(FLOAT+row*WIDTH+index,normalized,index<WIDTH)
        tl.store(BF16+row*WIDTH+index,normalized,index<WIDTH)


def rms_norm(x, weight, eps):
    """Original FP32 RMS equation and mixed-output ABI, fixed within a row."""
    width = x.shape[-1]
    if width < 1 or weight.shape != (width,):
        raise ValueError('RMS norm requires a matching one-dimensional weight')
    flat = x.reshape(-1,width).contiguous()
    full = torch.empty_like(x,dtype=torch.float32,memory_format=torch.contiguous_format)
    reduced = torch.empty_like(x,dtype=torch.bfloat16,memory_format=torch.contiguous_format)
    if flat.shape[0]:
        _rms_norm[(flat.shape[0],)](flat,weight.contiguous(),full,reduced,width,eps,
                                   enable_fp_fusion=False)
    return reduced,full


@triton.jit
def _linear_fp32(X, W, Y, M, N: tl.constexpr, K: tl.constexpr,
                 XS: tl.constexpr, WS: tl.constexpr, WK: tl.constexpr,
                 BN: tl.constexpr = 8, BK: tl.constexpr = 256):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.full((BN,), 0, tl.float32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + ks
        x = tl.load(X + row * XS + k, (row < M) & (k < K), 0).to(tl.float32)
        w = tl.load(W + cols[:, None] * WS + k[None, :] * WK,
                    (cols[:, None] < N) & (k[None, :] < K), 0).to(tl.float32)
        acc += tl.sum(w * x[None, :], 1)
    tl.store(Y + row * N + cols, acc, (row < M) & (cols < N))


def linear(x, weight):
    """Original linear formula with an M-independent accumulation algorithm."""
    if x.shape[-1] != weight.shape[1] or weight.ndim != 2:
        raise ValueError('linear expects input [...,K] and weight [N,K]')
    shape = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    output = x.new_empty((*shape, weight.shape[0]))
    if not flat.shape[0]:
        return output
    if x.dtype == weight.dtype == torch.bfloat16:
        # Reuse the already integrated original-head GEMM; its K loop and
        # accumulator tile are independent of M, including a masked last tile.
        from .bf16_logits910c import bf16_gemm_nt
        bf16_gemm_nt(flat, weight, output.reshape(flat.shape[0], -1))
    elif x.dtype == weight.dtype == torch.float32:
        _linear_fp32[(flat.shape[0], triton.cdiv(weight.shape[0], 8))](
            flat, weight, output, flat.shape[0], weight.shape[0], flat.shape[1],
            flat.stride(0), *weight.stride(), enable_fp_fusion=False)
    else:
        raise TypeError('invariant linear requires matching BF16 or FP32 operands')
    return output


@triton.jit
def _qk(Q, K, V, L, H: tl.constexpr, D: tl.constexpr, S: tl.constexpr,
        Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr,
        K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
        SCALE: tl.constexpr, BD: tl.constexpr, BS: tl.constexpr = 16):
    row, head = tl.program_id(0), tl.program_id(1)
    keys = tl.program_id(2) * BS + tl.arange(0, BS)
    dims = tl.arange(0, BD)
    q = tl.load(Q + row * Q0 + head * Q1 + dims * Q2, dims < D, 0).to(tl.float32)
    kv = tl.load(K + row * K0 + keys[:, None] * K1 + dims[None, :] * K2,
                 (keys[:, None] < S) & (dims[None, :] < D), 0).to(tl.float32)
    valid = tl.load(V + row * S + keys, keys < S, 0)
    logits = tl.sum(kv * q[None, :], 1) * SCALE
    logits = tl.where(valid, logits, -float('inf'))
    tl.store(L + (row * H + head) * S + keys, logits, keys < S)


@triton.jit
def _softmax(L, SINK, P, LSE, H: tl.constexpr, S: tl.constexpr,
             HAS_SINK: tl.constexpr, BS: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    keys = tl.arange(0, BS)
    logits = tl.load(L + (row * H + head) * S + keys, keys < S, -float('inf'))
    maximum = tl.max(logits, 0)
    nonempty = maximum != -float('inf')
    safe_maximum = tl.where(nonempty, maximum, 0.)
    normalizer = tl.sum(tl.exp(logits - safe_maximum), 0)
    lse = tl.where(nonempty, safe_maximum + tl.log(normalizer), float('inf'))
    if HAS_SINK:
        sink = tl.load(SINK + head)
        m = tl.maximum(maximum, sink)
        denominator = tl.sum(tl.exp(logits - m), 0) + tl.exp(sink - m)
        probability = tl.exp(logits - m) / denominator
    else:
        probability = tl.where(nonempty, tl.exp(logits - safe_maximum) / normalizer, 0.)
    tl.store(P + (row * H + head) * S + keys, probability, keys < S)
    tl.store(LSE + row * H + head, lse)


@triton.jit
def _pv(P, K, O, H: tl.constexpr, D: tl.constexpr, DV: tl.constexpr,
        S: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
        BS: tl.constexpr, BD: tl.constexpr = 8):
    row, head = tl.program_id(0), tl.program_id(1)
    dims = tl.program_id(2) * BD + tl.arange(0, BD)
    keys = tl.arange(0, BS)
    p = tl.load(P + (row * H + head) * S + keys, keys < S, 0)
    kv = tl.load(K + row * K0 + keys[:, None] * K1 + dims[None, :] * K2,
                 (keys[:, None] < S) & (dims[None, :] < DV), 0).to(tl.float32)
    result = tl.sum(p[:, None] * kv, 0)
    tl.store(O + (row * H + head) * DV + dims, result, dims < DV)


def attend(query, gathered, valid, scale, sink, dim_v):
    """FlashMLA reference equation, including sink and empty-row semantics.

    A program owns one query/head. It neither splits keys by batch occupancy
    nor changes reduction order when the same token moves to another row.
    """
    rows, heads, dim = query.shape
    keys = gathered.shape[1]
    output = query.new_empty((rows, heads, dim_v))
    lse = torch.empty((rows, heads), device=query.device, dtype=torch.float32)
    if not rows:
        return output, lse
    if not keys:
        output.zero_(); lse.fill_(float('inf'))
        return output, lse
    logits = torch.empty((rows, heads, keys), device=query.device, dtype=torch.float32)
    probabilities = torch.empty_like(logits)
    valid = valid.contiguous()
    _qk[(rows, heads, triton.cdiv(keys, 16))](
        query, gathered, valid, logits, heads, dim, keys,
        *query.stride(), *gathered.stride(), scale, triton.next_power_of_2(dim),
        enable_fp_fusion=False)
    _softmax[(rows, heads)](logits, logits if sink is None else sink,
        probabilities, lse, heads, keys, sink is not None, triton.next_power_of_2(keys),
        enable_fp_fusion=False)
    _pv[(rows, heads, triton.cdiv(dim_v, 8))](
        probabilities, gathered, output, heads, dim, dim_v, keys,
        *gathered.stride(), triton.next_power_of_2(keys), enable_fp_fusion=False)
    return output, lse


@triton.jit
def _engram_gate(HIDDEN, ORIGINAL, KEY, VALUE, QW, KW, MASK, OUT,
                 C: tl.constexpr, H: tl.constexpr, EPS: tl.constexpr,
                 BK: tl.constexpr = 1024):
    row, channel = tl.program_id(0), tl.program_id(1)
    offsets = tl.arange(0, BK)
    oo = tl.full((),0,tl.float32)
    kk = tl.full((),0,tl.float32)
    product = tl.full((),0,tl.float32)
    for part in range(tl.cdiv(H,BK)):
        columns = part*BK+offsets
        base = (row*C+channel)*H+columns
        original = tl.load(ORIGINAL+base,columns<H,0).to(tl.float32)
        key = tl.load(KEY+base,columns<H,0).to(tl.float32)
        qw = tl.load(QW+channel*H+columns,columns<H,0).to(tl.float32)
        kw = tl.load(KW+channel*H+columns,columns<H,0).to(tl.float32)
        oo += tl.sum(original*original,0)
        kk += tl.sum(key*key,0)
        product += tl.sum((original*(qw*kw))*key,0)
    rstd = tl.rsqrt(oo/H+EPS)*tl.rsqrt(kk/H+EPS)
    dot = (product*rstd)*(H**-.5)
    magnitude = tl.sqrt(tl.maximum(tl.abs(dot),1.e-6))
    signed = tl.where(dot.to(tl.int32,bitcast=True)<0,-magnitude,magnitude)
    gate = 1./(1.+tl.exp(-signed))
    gate = tl.where(tl.load(MASK+row)!=0,gate,0.)
    for part in range(tl.cdiv(H,BK)):
        columns = part*BK+offsets
        base = (row*C+channel)*H+columns
        hidden = tl.load(HIDDEN+base,columns<H,0).to(tl.float32)
        value = tl.load(VALUE+row*H+columns,columns<H,0).to(tl.float32)
        tl.store(OUT+base,hidden+gate*value,columns<H)


def engram_gate(hidden, original, key, value, q_weight, k_weight, mask, eps):
    """Original rotated Engram gate with a row-independent reduction tree."""
    rows, channels, width = hidden.shape
    output = torch.empty_like(hidden)
    if not rows:
        return output
    qw = q_weight.float().reshape(-1,width).expand(channels,width).contiguous()
    kw = k_weight.float().reshape(-1,width).expand(channels,width).contiguous()
    _engram_gate[(rows,channels)](hidden.contiguous(),original.contiguous(),
        key.contiguous(),value.contiguous(),qw,kw,mask.contiguous(),output,
        channels,width,eps,enable_fp_fusion=False)
    return output


@triton.jit
def _mhc_coefficients(X, DOT, SCALE, BASE, PRE, POST, COMB,
                      C: tl.constexpr, H: tl.constexpr, EPS: tl.constexpr,
                      HC_EPS: tl.constexpr, REPEAT: tl.constexpr,
                      BC: tl.constexpr, BK: tl.constexpr = 1024):
    row = tl.program_id(0)
    k = tl.arange(0,BK)
    squares = tl.full((),0,tl.float32)
    for part in range(tl.cdiv(C*H,BK)):
        index = part*BK+k
        value = tl.load(X+row*C*H+index,index<C*H,0).to(tl.float32)
        squares += tl.sum(value*value,0)
    rstd = tl.rsqrt(squares/(C*H)+EPS)
    channels = tl.arange(0,BC)
    valid = channels<C
    a0 = tl.load(SCALE).to(tl.float32)
    a1 = tl.load(SCALE+1).to(tl.float32)
    a2 = tl.load(SCALE+2).to(tl.float32)
    n = C*(C+2)
    pre_dot = tl.load(DOT+row*n+channels,valid,0)*rstd
    post_dot = tl.load(DOT+row*n+C+channels,valid,0)*rstd
    b0 = tl.load(BASE+channels,valid,0).to(tl.float32)
    b1 = tl.load(BASE+C+channels,valid,0).to(tl.float32)
    pre = 1./(1.+tl.exp(-(pre_dot*a0+b0)))+HC_EPS
    post = 2./(1.+tl.exp(-(post_dot*a1+b1)))
    tl.store(PRE+row*C+channels,pre,valid)
    tl.store(POST+row*C+channels,post,valid)
    matrix_valid = valid[:,None]&valid[None,:]
    index = 2*C+channels[:,None]*C+channels[None,:]
    logits = tl.load(DOT+row*n+index,matrix_valid,0)*rstd*a2
    logits += tl.load(BASE+index,matrix_valid,0).to(tl.float32)
    logits = tl.where(matrix_valid,logits,-float('inf'))
    maximum = tl.where(valid,tl.max(logits,1),0.)
    exponent = tl.exp(logits-maximum[:,None])
    normalizer = tl.sum(exponent,1)
    combination = tl.where(matrix_valid,exponent/tl.maximum(normalizer[:,None],1.e-30)+HC_EPS,0.)
    combination = tl.where(matrix_valid,combination/(tl.sum(combination,0)[None,:]+HC_EPS),0.)
    for _ in range(REPEAT-1):
        combination = tl.where(matrix_valid,combination/(tl.sum(combination,1)[:,None]+HC_EPS),0.)
        combination = tl.where(matrix_valid,combination/(tl.sum(combination,0)[None,:]+HC_EPS),0.)
    tl.store(COMB+row*C*C+channels[:,None]*C+channels[None,:],combination,matrix_valid)


@triton.jit
def _mhc_apply(X, PREVIOUS, OUT, C: tl.constexpr, H: tl.constexpr,
                BH: tl.constexpr = 256):
    row = tl.program_id(0)
    columns = tl.program_id(1)*BH+tl.arange(0,BH)
    result = tl.full((BH,),0,tl.float32)
    for channel in range(C):
        x = tl.load(X+(row*C+channel)*H+columns,columns<H,0).to(tl.float32)
        weight = tl.load(PREVIOUS+row*C+channel).to(tl.float32)
        result += x*weight
    tl.store(OUT+row*H+columns,result,columns<H)


def mhc_pre(trunk, previous, fn, scale, base, eps, hc_eps, repeat):
    """Original prenorm projection, split mixes and Sinkhorn equations.

    One program owns a row's coefficients. A CANN nonlinear/reduction kernel
    may choose a different implementation for another batch/storage shape;
    this provider fixes the arithmetic within the row instead.
    """
    rows,channels,width = trunk.shape
    if (fn.shape!=(channels*(channels+2),channels*width) or scale.shape!=(3,)
            or base.shape!=(channels*(channels+2),) or repeat<1):
        raise ValueError('invalid original mHC projection or coefficient geometry')
    trunk = trunk.contiguous()
    previous = previous.reshape(rows,channels).contiguous()
    dots = linear(trunk.reshape(rows,channels*width).float(),fn.float())
    pre = torch.empty((rows,channels),device=trunk.device,dtype=torch.float32)
    post = torch.empty_like(pre)
    combination = torch.empty((rows,channels,channels),device=trunk.device,dtype=torch.float32)
    value = torch.empty((rows,width),device=trunk.device,dtype=torch.bfloat16)
    if rows:
        _mhc_coefficients[(rows,)](trunk,dots,scale,base,pre,post,combination,
            channels,width,eps,hc_eps,repeat,triton.next_power_of_2(channels),enable_fp_fusion=False)
        _mhc_apply[(rows,triton.cdiv(width,256))](trunk,previous,value,channels,width,enable_fp_fusion=False)
    return value,post,combination,pre


@triton.jit
def _mhc_post(VALUE, RESIDUAL, POST, COMB, OUT,
              C: tl.constexpr, H: tl.constexpr, BH: tl.constexpr = 256):
    row,channel = tl.program_id(0),tl.program_id(1)
    columns = tl.program_id(2)*BH+tl.arange(0,BH)
    result = tl.full((BH,),0,tl.float32)
    for source in range(C):
        x = tl.load(RESIDUAL+(row*C+source)*H+columns,columns<H,0).to(tl.float32)
        weight = tl.load(COMB+row*C*C+source*C+channel).to(tl.float32)
        result += x*weight
    value = tl.load(VALUE+row*H+columns,columns<H,0).to(tl.float32)
    weight = tl.load(POST+row*C+channel).to(tl.float32)
    result += value*weight
    tl.store(OUT+(row*C+channel)*H+columns,result,columns<H)


def mhc_post(value, residual, post, combination):
    rows,channels,width = residual.shape
    output = torch.empty_like(residual)
    if rows:
        _mhc_post[(rows,channels,triton.cdiv(width,256))](value.contiguous(),residual.contiguous(),
            post.contiguous(),combination.contiguous(),output,channels,width,enable_fp_fusion=False)
    return output
