"""HyperConnection prenorm forward GEMM, TileLang implementation for Ascend."""

import math

import torch
import tilelang
from tilelang.ascend import language as T
from tilelang.layout import make_ascend_compact_nz_layout

from .. import _C


def get_config(mhc_hidden_size: int, mhc_mult3: int, num_tokens: int, deterministic: bool = False) -> dict:
    """Return optimal kernel parameters for a given shape."""
    block_num_tokens = 128
    block_hidden = 256
    l0b_size_bytes = 64 * 1024
    l0c_size_bytes = 256 * 1024

    assert num_tokens > 0, f'HC prenorm fwd requires positive num_tokens, got {num_tokens}'
    assert 0 < mhc_mult3 <= 64 and mhc_mult3 % 8 == 0, f'HC prenorm fwd requires mhc_mult3 to be 8-aligned and no larger than 64, got {mhc_mult3}'
    assert mhc_hidden_size > 0 and mhc_hidden_size % block_hidden == 0, (
        f'HC prenorm fwd requires mhc_hidden_size to be a positive multiple of {block_hidden}, got {mhc_hidden_size}'
    )

    num_aic_cores = _C.get_num_sms()
    block_mhc_mult3 = (mhc_mult3 + 15) // 16 * 16
    num_hidden_blocks = mhc_hidden_size // block_hidden
    num_token_blocks = math.ceil(num_tokens / block_num_tokens)
    num_l0b_stages = l0b_size_bytes // (block_mhc_mult3 * block_hidden * 4)
    num_l0c_stages = l0c_size_bytes // (block_num_tokens * block_mhc_mult3 * 4)

    # K partitioning defines the per-row floating-point reduction.
    # M-dependent split selection improves L0C residency/B reuse,
    # but changes the reduction grouping and breaks batch invariance.
    num_k_splits_by_workload = (
        16 if deterministic else max(1, num_aic_cores * num_l0c_stages // num_token_blocks)
    )
    num_k_splits_by_hidden = num_hidden_blocks & -num_hidden_blocks
    num_k_splits_by_cores = num_aic_cores & -num_aic_cores
    split_limit = min(num_k_splits_by_workload, num_k_splits_by_hidden, num_k_splits_by_cores)
    num_k_splits = 1 << (split_limit.bit_length() - 1)

    active_aic_cores = num_k_splits * min(num_aic_cores // num_k_splits, num_token_blocks)
    cores_per_split = active_aic_cores // num_k_splits
    num_hidden_blocks_per_split = num_hidden_blocks // num_k_splits
    num_token_blocks_per_chunk = cores_per_split * num_l0c_stages

    return {
        'block_mhc_mult3': block_mhc_mult3,
        'num_aic_cores': active_aic_cores,
        'num_k_splits': num_k_splits,
        'cores_per_split': cores_per_split,
        'num_hidden_blocks_per_split': num_hidden_blocks_per_split,
        'num_l0b_stages': num_l0b_stages,
        'num_l0c_stages': num_l0c_stages,
        'num_token_blocks_per_chunk': num_token_blocks_per_chunk,
    }

@tilelang.jit
def tf32_hc_prenorm_gemm_kernel(
    x,
    fn,
    out_mul,
    sqrsum,
    block_mhc_mult3: int,
    num_aic_cores: int,
    num_k_splits: int,
    cores_per_split: int,
    num_hidden_blocks_per_split: int,
    num_l0b_stages: int,
    num_l0c_stages: int,
    num_token_blocks_per_chunk: int,
    deterministic: bool = False,
):
    block_num_tokens = 128
    block_hidden = 256
    mad_hidden = 64
    num_aivs = 2
    half_block_num_tokens = block_num_tokens // num_aivs
    num_mads_per_block = block_hidden // mad_hidden
    num_load_stages = 3
    num_cast_stages = 2
    num_l1_x_stages = 3
    num_mad_stages = 2

    num_tokens = T.dynamic('num_tokens')
    mhc_mult3 = T.const('mhc_mult3')
    mhc_hidden_size = T.const('mhc_hidden_size')

    x: T.Tensor[(num_tokens, mhc_hidden_size), T.bfloat16]
    fn: T.Tensor[(mhc_mult3, mhc_hidden_size), T.float32]
    out_mul: T.Tensor[(num_tokens, mhc_mult3), T.float32]
    sqrsum: T.Tensor[(num_tokens,), T.float32]

    num_store_phases = num_k_splits if deterministic else (2 if num_k_splits > 1 else 1)
    # In non-deterministic mode, phase 0 separates split 0's non-atomic store
    # from the remaining splits' atomic adds. The last phase only needs a
    # barrier when another token chunk follows. Encode that optional barrier
    # as a uniform 0/1-trip loop because AutoSchedule rejects inter-core waits
    # under conditional control flow inside T.PerCoreTask.
    num_always_synced_store_phases = num_store_phases if deterministic else (1 if num_k_splits > 1 else 0)
    num_chunk_end_syncs = 0 if deterministic or num_k_splits == 1 else 1
    nz_stage_rows = half_block_num_tokens + 1

    with T.MixedKernel(num_aic_cores) as (core_id, aiv_id):
        split_id = core_id // cores_per_split
        core_in_split = core_id % cores_per_split
        num_token_blocks = T.ceildiv(num_tokens, block_num_tokens)
        num_token_chunks = T.ceildiv(num_token_blocks, num_token_blocks_per_chunk)

        x_ub = T.alloc_shared((half_block_num_tokens, block_hidden), T.bfloat16)
        x_nz_ub = T.alloc_shared((nz_stage_rows, block_hidden), T.float32)
        T.annotate_layout({x_nz_ub: make_ascend_compact_nz_layout(x_nz_ub)})
        sqrsum_ub = T.alloc_shared((num_l0c_stages, half_block_num_tokens), T.float32)

        x_l1 = T.alloc_l1((block_num_tokens, block_hidden), T.float32)
        fn_l1 = T.alloc_l1((block_mhc_mult3, block_hidden), T.float32)
        x_l0 = T.alloc_l0a((block_num_tokens, mad_hidden), T.float32)
        fn_l0 = T.alloc_l0b((num_mads_per_block, block_mhc_mult3, mad_hidden), T.float32)
        out_mul_l0 = T.alloc_l0c((num_l0c_stages, block_num_tokens, block_mhc_mult3), T.float32)

        T.annotate_buffer_versions(
            {
                x_ub: num_load_stages,
                x_nz_ub: num_cast_stages,
                x_l1: num_l1_x_stages,
                fn_l1: num_l0b_stages,
                x_l0: num_mad_stages,
                fn_l0: num_l0b_stages,
            }
        )

        if split_id != 0:
            T.set_atomic('add', 'float32')
        T.set_hf32_mode('nearest_even')
        for chunk_idx in T.Serial(num_token_chunks):
            token_block_base = chunk_idx * num_token_blocks_per_chunk + core_in_split
            num_valid_token_stages = T.min(
                num_l0c_stages,
                T.max(T.ceildiv(num_token_blocks - token_block_base, cores_per_split), 0),
            )
            aiv_token_begin_base = token_block_base * block_num_tokens + aiv_id * half_block_num_tokens
            num_valid_sqrsum_stages = T.min(
                num_l0c_stages,
                T.max(
                    T.ceildiv(
                        num_tokens - aiv_token_begin_base,
                        cores_per_split * block_num_tokens,
                    ),
                    0,
                ),
            )

            for local_hidden_block in T.Pipelined(num_hidden_blocks_per_split, num_stages=2):
                global_hidden_block = split_id * num_hidden_blocks_per_split + local_hidden_block
                hidden_begin = global_hidden_block * block_hidden

                # Keep physical buffers block_mhc_mult3-aligned, but restrict
                # data movement and MAD regions to the logical mhc_mult3.
                T.copy(
                    fn[0:mhc_mult3, hidden_begin : hidden_begin + block_hidden],
                    fn_l1[0:mhc_mult3, :],
                )
                for hidden_mad in T.Serial(num_mads_per_block):
                    T.copy(
                        fn_l1[0:mhc_mult3, hidden_mad * mad_hidden : (hidden_mad + 1) * mad_hidden],
                        fn_l0[hidden_mad, 0:mhc_mult3, :],
                    )

                for token_stage in T.Serial(
                    num_valid_token_stages,
                    annotations={'multi_buffer_eligible': [x_l1]},
                ):
                    token_block = token_block_base + token_stage * cores_per_split
                    token_begin = token_block * block_num_tokens
                    actual_num_tokens = T.min(num_tokens - token_begin, block_num_tokens)
                    aiv_token_begin = token_begin + aiv_id * half_block_num_tokens

                    if aiv_token_begin < num_tokens:
                        actual_aiv_num_tokens = T.min(
                            num_tokens - aiv_token_begin,
                            half_block_num_tokens,
                        )
                        T.copy(
                            x[
                                aiv_token_begin : aiv_token_begin + actual_aiv_num_tokens,
                                hidden_begin : hidden_begin + block_hidden,
                            ],
                            x_ub[0:actual_aiv_num_tokens, :],
                            l2_cache_ctrl='notalloc_keep',
                            pad_value=0,
                        )

                        # Widen one AIV half-tile, pack padded NZ, and update row sums.
                        with T.SimdVF(latency=640):
                            bf16_mask = T.simd.pset(16)
                            f32_mask = T.simd.pset(32)
                            nz_stride = T.int32((nz_stage_rows << 16) | (8 * nz_stage_rows))
                            zero_bf16 = T.simd.vdup(T.bfloat16(0), 'bfloat16', bf16_mask)
                            row_indices = T.simd.vci(T.float32(0), 'float32')
                            current_row = T.simd.alloc_var('float32')
                            current_row = T.simd.vdup(T.float32(0), 'float32', f32_mask)
                            one = T.simd.vdup(T.float32(1), 'float32', f32_mask)
                            sqr_sums = T.simd.alloc_var('float32')
                            sqr_sums = T.simd.vdup(T.float32(0), 'float32', f32_mask)

                            for row in T.Serial(half_block_num_tokens):
                                row_acc = T.simd.alloc_var('float32')
                                row_acc = T.simd.vdup(T.float32(0), 'float32', f32_mask)
                                nz_ptr = T.simd.make_ubuf_ptr(
                                    T.access_ptr(
                                        x_nz_ub[row, 0],
                                        'w',
                                        1,
                                        block_hidden,
                                    ),
                                    'float32',
                                )

                                for pass_id in T.Serial(block_hidden // 128):
                                    packed = T.simd.vld(x_ub[row, pass_id * 128])
                                    lo_bits, hi_bits = T.simd.vintlv(zero_bf16, packed)
                                    lo = T.reinterpret(lo_bits, 'float32x64')
                                    hi = T.reinterpret(hi_bits, 'float32x64')

                                    row_acc = T.simd.vadd(
                                        row_acc,
                                        T.simd.vmul(lo, lo, f32_mask),
                                        f32_mask,
                                    )
                                    row_acc = T.simd.vadd(
                                        row_acc,
                                        T.simd.vmul(hi, hi, f32_mask),
                                        f32_mask,
                                    )
                                    nz_ptr = T.simd.vsstb(
                                        lo,
                                        nz_ptr,
                                        nz_stride,
                                        f32_mask,
                                        update=True,
                                    )
                                    nz_ptr = T.simd.vsstb(
                                        hi,
                                        nz_ptr,
                                        nz_stride,
                                        f32_mask,
                                        update=True,
                                    )

                                row_sum = T.simd.alloc_var('float32')
                                row_sum = T.simd.vcadd(row_acc, f32_mask)
                                row_sum_broadcast = T.simd.vdupv(row_sum, f32_mask)
                                row_mask = T.simd.vcmp(
                                    row_indices,
                                    current_row,
                                    f32_mask,
                                    'eq',
                                )
                                sqr_sums = T.simd.vsel(row_sum_broadcast, sqr_sums, row_mask)
                                current_row = T.simd.vadd(current_row, one, f32_mask)

                            if local_hidden_block != 0:
                                previous = T.simd.vld(sqrsum_ub[token_stage, 0])
                                sqr_sums = T.simd.vadd(sqr_sums, previous, f32_mask)
                            T.simd.vsts(sqrsum_ub[token_stage, 0], sqr_sums, f32_mask)

                        T.dual_copy(
                            x_nz_ub[0:half_block_num_tokens, 0:block_hidden],
                            x_l1[0:block_num_tokens, 0:block_hidden],
                        )

                    for hidden_mad in T.Serial(num_mads_per_block):
                        T.copy(
                            x_l1[
                                0:actual_num_tokens,
                                hidden_mad * mad_hidden : (hidden_mad + 1) * mad_hidden,
                            ],
                            x_l0[0:actual_num_tokens, :],
                        )
                        T.gemm(
                            x_l0[0:actual_num_tokens, :],
                            fn_l0[hidden_mad, 0:mhc_mult3, :],
                            out_mul_l0[token_stage, 0:actual_num_tokens, 0:mhc_mult3],
                            transpose_B=True,
                            clear_accum=(local_hidden_block == 0 and hidden_mad == 0),
                            unit_flag_ctrl=T.Select(
                                local_hidden_block == num_hidden_blocks_per_split - 1 and hidden_mad == num_mads_per_block - 1,
                                3,
                                2,
                            ),
                        )

            with T.PerCoreTask():
                for store_phase in T.Serial(num_store_phases):
                    store_this_phase = T.if_then_else(
                        deterministic,
                        split_id == store_phase,
                        T.if_then_else(store_phase == 0, split_id == 0, split_id != 0),
                    )
                    out_mul_flag = (chunk_idx * num_store_phases + store_phase) % 4

                    if store_this_phase:
                        with T.Task():
                            for token_stage in T.Serial(num_valid_token_stages):
                                token_block = token_block_base + token_stage * cores_per_split
                                token_begin = token_block * block_num_tokens
                                actual_num_tokens = T.min(num_tokens - token_begin, block_num_tokens)
                                T.copy(
                                    out_mul_l0[token_stage, 0:actual_num_tokens, 0:mhc_mult3],
                                    out_mul[
                                        token_begin : token_begin + actual_num_tokens,
                                        0:mhc_mult3,
                                    ],
                                    unit_flag_ctrl=3,
                                )
                    num_store_syncs = T.if_then_else(
                        store_phase < num_always_synced_store_phases,
                        1,
                        T.min(num_token_chunks - chunk_idx - 1, num_chunk_end_syncs),
                    )
                    for _ in T.Serial(num_store_syncs):
                        T.ascend_sync_inter_arrive('PIPE_FIX', out_mul_flag)
                        T.ascend_sync_inter_wait('PIPE_FIX', out_mul_flag)

            with T.PerCoreTask():
                for store_phase in T.Serial(num_store_phases):
                    store_this_phase = T.if_then_else(
                        deterministic,
                        split_id == store_phase,
                        T.if_then_else(store_phase == 0, split_id == 0, split_id != 0),
                    )
                    sqrsum_flag = 4 + (chunk_idx * num_store_phases + store_phase) % 4

                    if store_this_phase:
                        with T.Task():
                            for token_stage in T.Serial(num_valid_sqrsum_stages):
                                token_block = token_block_base + token_stage * cores_per_split
                                token_begin = token_block * block_num_tokens + aiv_id * half_block_num_tokens
                                actual_aiv_num_tokens = T.min(
                                    num_tokens - token_begin,
                                    half_block_num_tokens,
                                )
                                T.copy(
                                    sqrsum_ub[token_stage, 0:actual_aiv_num_tokens],
                                    sqrsum[token_begin : token_begin + actual_aiv_num_tokens],
                                    l2_cache_ctrl='normal_fv',
                                )
                    num_store_syncs = T.if_then_else(
                        store_phase < num_always_synced_store_phases,
                        1,
                        T.min(num_token_chunks - chunk_idx - 1, num_chunk_end_syncs),
                    )
                    for _ in T.Serial(num_store_syncs):
                        T.ascend_sync_inter_arrive('PIPE_MTE3', sqrsum_flag)
                        T.ascend_sync_inter_wait('PIPE_MTE3', sqrsum_flag)

        if split_id != 0:
            T.set_atomic_none()


def tf32_hc_prenorm_gemm(a: torch.Tensor,
                         b: torch.Tensor,
                         d: torch.Tensor,
                         sqr_sum: torch.Tensor,
                         num_splits: int | None = None) -> None:
    """Fused hyperconnection prenorm forward.

    D = float(A) @ B.T and sqr_sum = sum(float(A) ** 2, dim=-1), computed in one kernel.

    Args:
        a: [M, K] BF16 input.
        b: [N, K] FP32 weight.
        d: [M, N] FP32 output, or [num_splits, M, N] when num_splits is given.
        sqr_sum: [M] FP32 output, or [num_splits, M] when num_splits is given.
        num_splits: K split count. This backend reduces the split inside the kernel, so
            only None (no split dimension) and 1 are accepted.
    """
    m, k = a.shape
    n = b.shape[0]

    if num_splits is None:
        out_mul, out_sqr_sum = d, sqr_sum
    else:
        # The K split is reduced inside the kernel, so the public result only carries one partial.
        assert num_splits == 1, f'This backend reduces the K split internally, got {num_splits=}'
        out_mul, out_sqr_sum = d[0], sqr_sum[0]

    if m == 0:
        return
    deterministic = _C.get_deterministic_algorithms()
    config = get_config(k, n, m, deterministic=deterministic)
    if _C.get_dry_run():
        # par_compile() sets the dry run flag: build the kernel but execute nothing.
        tf32_hc_prenorm_gemm_kernel.compile(a, b, out_mul, out_sqr_sum, deterministic=deterministic, **config)
        return
    tf32_hc_prenorm_gemm_kernel(a, b, out_mul, out_sqr_sum, deterministic=deterministic, **config)
