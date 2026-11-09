"""Original Ascend fixed-point draw in one NPU kernel.

The pinned Ascend sampler folds uint32 weights by column modulo acc_slots,
then uses Philox4x32-10 and two inclusive integer scans. Keep that device
operation indivisible when a five-step DFlash graph reuses its scratch pool.
No request padding, host numerical work, or alternative RNG is involved.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _multiply_words(a, multiplier: tl.constexpr):
    # Exact 32x32->64 product using only 32-bit device arithmetic.
    lo = a & 65535
    hi = a >> 16
    p0 = lo * (multiplier & 65535)
    p1 = hi * (multiplier & 65535)
    p2 = lo * (multiplier >> 16)
    p3 = hi * (multiplier >> 16)
    middle = (p0 >> 16) + (p1 & 65535) + (p2 & 65535)
    low = (p0 & 65535) | (middle << 16)
    high = p3 + (p1 >> 16) + (p2 >> 16) + (middle >> 16)
    return low.to(tl.uint32), high.to(tl.uint32)


@triton.jit
def _philox_words(key0, key1, position, case):
    sequence = position * 3 + case
    c0 = tl.full((), 0, tl.uint32)
    c1 = tl.full((), 0, tl.uint32)
    c2 = sequence.to(tl.uint32)
    c3 = tl.where(sequence < 0, 0xffffffff, 0).to(tl.uint32)
    for _ in range(10):
        a0, a1 = _multiply_words(c0, 0xD2511F53)
        b0, b1 = _multiply_words(c2, 0xCD9E8D57)
        c0, c1, c2, c3 = ((b1 ^ c1 ^ key0).to(tl.uint32), b0.to(tl.uint32),
                          (a1 ^ c3 ^ key1).to(tl.uint32), a0.to(tl.uint32))
        key0 = (key0 + 0x9E3779B9).to(tl.uint32)
        key1 = (key1 + 0xBB67AE85).to(tl.uint32)
    return c0, c1


@triton.jit
def _mod_words(low, high, divisor):
    # Unsigned 64/32 remainder; carry handles divisors above 2**31 too.
    remainder = tl.full((), 0, tl.uint32)
    for bit in range(31, -1, -1):
        carry = remainder >> 31
        value = (remainder << 1) | ((high >> bit) & 1)
        remainder = tl.where((carry != 0) | (value >= divisor), value-divisor, value).to(tl.uint32)
    for bit in range(31, -1, -1):
        carry = remainder >> 31
        value = (remainder << 1) | ((low >> bit) & 1)
        remainder = tl.where((carry != 0) | (value >= divisor), value-divisor, value).to(tl.uint32)
    return remainder


@triton.jit
def _weights_at(W, PREVIOUS, VERIFY, row, index, mask,
                STRIDE: tl.constexpr, PREVIOUS_STRIDE: tl.constexpr,
                LOGPROBS: tl.constexpr, REJECT: tl.constexpr,
                PENALTY: tl.constexpr):
    value = tl.load(W+row*STRIDE+index, mask, 0)
    if LOGPROBS:
        probability = tl.exp(value.to(tl.float32))
        if REJECT:
            previous = tl.load(PREVIOUS+row*PREVIOUS_STRIDE+index, mask, 0).to(tl.float32)
            residual = tl.maximum(probability-tl.exp(previous-PENALTY), 0.)
            probability = tl.where(tl.load(VERIFY+row), residual, probability)
        scaled = probability*2147483648.
        # float32 cannot represent INT32_MAX; implement the original signed
        # saturating ROUND_Z conversion explicitly at its 2**31 boundary.
        value = tl.where(scaled >= 2147483648., 2147483647,
                         scaled.to(tl.int32)).to(tl.uint32)
    return tl.where(mask, value.to(tl.int32), 0).to(tl.uint32)


@triton.jit
def _draw(W, PREVIOUS, VERIFY, SEEDS, POSITIONS, CASE, FALLBACK, OUT, TOTAL,
          WIDTH: tl.constexpr, STRIDE: tl.constexpr, SLOTS: tl.constexpr,
          GROUPS: tl.constexpr, BG: tl.constexpr,
          TENSOR_CASE: tl.constexpr, SCALAR_CASE: tl.constexpr,
          PREVIOUS_STRIDE: tl.constexpr, LOGPROBS: tl.constexpr,
          REJECT: tl.constexpr, PENALTY: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, SLOTS)
    folded = tl.full((SLOTS,), 0, tl.uint32)
    for group in range(GROUPS):
        index = columns+group*SLOTS
        folded += _weights_at(W, PREVIOUS, VERIFY, row, index, index < WIDTH,
                              STRIDE, PREVIOUS_STRIDE, LOGPROBS, REJECT, PENALTY)
    # Signed storage has identical wraparound addition. Avoid the backend's
    # implicit uint32->uint64 vector promotion; compare unsigned via sign bias.
    cdf = tl.cumsum(folded.to(tl.int32), 0, dtype=tl.int32).to(tl.uint32)
    total = tl.sum(folded.to(tl.int32), 0, dtype=tl.int32).to(tl.uint32)
    key0 = tl.load(SEEDS+row*2).to(tl.uint32)
    key1 = tl.load(SEEDS+row*2+1).to(tl.uint32)
    position = tl.load(POSITIONS+row)
    case = tl.load(CASE+row) if TENSOR_CASE else SCALAR_CASE
    low, high = _philox_words(key0, key1, position, case)
    target = _mod_words(low, high, tl.maximum(total, 1).to(tl.uint32)).to(tl.uint32)
    sign = tl.full((), 0x80000000, tl.uint32)
    below = (cdf ^ sign).to(tl.int32) <= (target ^ sign).to(tl.int32)
    offset = tl.minimum(tl.sum(below.to(tl.int32), 0, dtype=tl.int32), SLOTS-1)
    before = tl.sum(tl.where(columns == offset-1, cdf.to(tl.int32), 0), 0, dtype=tl.int32).to(tl.uint32)
    remaining = (target-before).to(tl.uint32)
    groups = tl.arange(0, BG)
    index = offset+groups*SLOTS
    weights = _weights_at(W, PREVIOUS, VERIFY, row, index,
        (groups < GROUPS) & (index < WIDTH), STRIDE, PREVIOUS_STRIDE,
        LOGPROBS, REJECT, PENALTY)
    local_cdf = tl.cumsum(weights.to(tl.int32), 0, dtype=tl.int32).to(tl.uint32)
    below = (local_cdf ^ sign).to(tl.int32) <= (remaining ^ sign).to(tl.int32)
    group = tl.sum((below & (groups < GROUPS)).to(tl.int32), 0, dtype=tl.int32)
    selected = offset+group*SLOTS
    fallback = tl.load(FALLBACK+row)
    tl.store(OUT+row, tl.where(total > 0, selected, fallback))
    tl.store(TOTAL+row, total.to(tl.int64))


def _launch(weights, seeds, positions, case, fallback, previous=None, verify=None, *, logprobs=False):
    rows, width = weights.shape
    if width < 1 or weights.stride(1) != 1:
        raise ValueError('sampling weights require nonempty contiguous columns')
    slots = 4096 if width >= 8192 else min(triton.next_power_of_2(width), 2048)
    groups = triton.cdiv(width, slots)
    output = torch.empty((rows,), dtype=torch.int32, device=weights.device)
    total = torch.empty((rows,), dtype=torch.int64, device=weights.device)
    words = seeds.contiguous().view(torch.int32)
    tensor_case = isinstance(case, torch.Tensor)
    if rows:
        from deepinfer.model_executor.layers.tilelang_ops.common import DRAFT_PROB_PENALTY
        _draw[(rows,)](weights, weights if previous is None else previous,
            positions if verify is None else verify,
            words, positions, case if tensor_case else positions,
            fallback, output, total, width, weights.stride(0), slots, groups,
            triton.next_power_of_2(groups), tensor_case, 0 if tensor_case else case,
            0 if previous is None else previous.stride(0), logprobs,
            previous is not None, DRAFT_PROB_PENALTY, enable_fp_fusion=False)
    return output, total


def draw(weights, seeds, positions, case, fallback):
    """Exact original two-level selection; weights are saturated 31-bit ints."""
    return _launch(weights, seeds, positions, case, fallback)


def draw_logprobs(logprobs, seeds, positions, case, fallback, *, previous=None, verify=None):
    """Original FP32 exp/residual, quantization, RNG and scans in one kernel."""
    return _launch(logprobs, seeds, positions, case, fallback, previous, verify, logprobs=True)
