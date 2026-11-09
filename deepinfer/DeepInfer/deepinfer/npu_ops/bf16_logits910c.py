"""BF16 operands with an FP32 accumulator/output for original LM heads."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gemm(A, B, D, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          AS0: tl.constexpr, AS1: tl.constexpr, BS0: tl.constexpr, BS1: tl.constexpr,
          DS0: tl.constexpr, DS1: tl.constexpr, CS0: tl.constexpr, CS1: tl.constexpr,
          HAS_C: tl.constexpr, BIAS: tl.constexpr,
          BM: tl.constexpr = 16, BN: tl.constexpr = 128, BK: tl.constexpr = 64):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    result = tl.full((BM, BN), 0, tl.float32)
    for base in range(tl.cdiv(K, BK)):
        k = base * BK + ks
        a = tl.load(A + rows[:, None] * AS0 + k[None, :] * AS1,
                    (rows[:, None] < M) & (k[None, :] < K), 0)
        b = tl.load(B + cols[None, :] * BS0 + k[:, None] * BS1,
                    (cols[None, :] < N) & (k[:, None] < K), 0)
        result += tl.dot(a, b)
    if HAS_C:
        c_rows = tl.full((BM,), 0, tl.int32) if BIAS else rows
        result += tl.load(C + c_rows[:, None] * CS0 + cols[None, :] * CS1,
                          (rows[:, None] < M) & (cols[None, :] < N), 0).to(tl.float32)
    tl.store(D + rows[:, None] * DS0 + cols[None, :] * DS1, result,
             (rows[:, None] < M) & (cols[None, :] < N))


def bf16_gemm_nt(a, b, d, c=None):
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError('expected A[M,K], B[N,K]')
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError('910C head GEMM requires BF16 operands')
    if d.shape != (a.shape[0], b.shape[0]) or d.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError('expected BF16/FP32 output[M,N]')
    if any(x.device.type != 'npu' or x.device != a.device for x in (a, b, d)):
        raise ValueError('910C GEMM requires one NPU device')
    if c is not None and (c.shape not in (d.shape, (1, d.shape[1]))
                          or c.device != d.device or c.dtype != d.dtype):
        raise ValueError('C must match D or a one-row bias')
    if not a.shape[0] or not b.shape[0]:
        return
    m, k = a.shape
    n = b.shape[0]
    _gemm[(triton.cdiv(m, 16), triton.cdiv(n, 128))](
        a, b, d, d if c is None else c, m, n, k,
        *a.stride(), *b.stride(), *d.stride(), *(d.stride() if c is None else c.stride()),
        c is not None, c is not None and c.shape[0] == 1)
