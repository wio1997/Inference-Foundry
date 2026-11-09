import tilelang.language as T


@T.macro
def ue8m0_exp_scale(scale: T.float32) -> T.int32:
    """Round scale up to a power of two: returns -ceil(log2(scale)), the exponent of the
    quantization multiplier 2^exp_scale (the ue8m0 dequant sf byte is -exp_scale + 127)."""
    scale_bits = T.reinterpret("uint32", scale)
    return -T.reinterpret("int32", ((scale_bits - 1) >> 23) + 1 - 127)
