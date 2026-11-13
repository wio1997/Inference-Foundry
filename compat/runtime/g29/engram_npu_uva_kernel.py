# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-project/vllm-ascend commit
# 77cf58266b084b517a658c19ba2d3c97fec27df9,
# vllm_ascend/ops/triton/engram_lookup.py.
# Copyright contributors to the vLLM project.
import triton
import triton.language as tl


@triton.jit
def _decode_e8m0(scale):
    exponent = scale.to(tl.int32) & 0xFF
    bits = exponent << 23
    bits = tl.where(exponent == 0, 0x00400000, bits)
    bits = tl.where(exponent == 255, 0x7FC00000, bits)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _gather_int8_e8m0(codes_ptrs, scales_ptrs, ids, output, total_rows,
                      WIDTH: tl.constexpr, GROUP: tl.constexpr, CHUNK: tl.constexpr):
    row = tl.program_id(0)
    index = tl.load(ids + row).to(tl.int64)
    # Match the reference's last line of defence: do not dereference a
    # pointer-table entry selected by an invalid/dummy hash id.
    owned = (index >= 0) & (index < total_rows)
    safe_index = tl.where(owned, index, 0)
    chunk = safe_index // CHUNK
    local = safe_index % CHUNK
    codes = tl.load(codes_ptrs + chunk).to(tl.pointer_type(tl.int8))
    scales = tl.load(scales_ptrs + chunk).to(tl.pointer_type(tl.uint8))
    col = tl.arange(0, WIDTH)
    values = tl.load(codes + local * WIDTH + col).to(tl.float32)
    groups = tl.arange(0, WIDTH // GROUP)
    scale = _decode_e8m0(tl.load(scales + local * (WIDTH // GROUP) + groups))
    values = tl.reshape(values, (WIDTH // GROUP, GROUP)) * scale[:, None]
    result = tl.reshape(values, (WIDTH,)).to(tl.bfloat16)
    tl.store(output + row * WIDTH + col, tl.where(owned, result, tl.zeros_like(result)))
