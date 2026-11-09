from dataclasses import dataclass

from tilelang.ascend import language as T


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class CastOutputConfig:
    # MegaMoE baseline uses FP8 with per-32-channel, packed column-major UE8M0 scales
    sf_block: tuple[int, int] = (1, 32)
    with_sf: bool = True
    use_tma_aligned_col_major_sf: bool = True
    use_packed_ue8m0: bool = True
    round_sf: bool = True
    clamp_min_value: float = 1e-4
    dtype: T.dtype = T.float8_e4m3fn
    sf_dtype: T.dtype = T.uint8


def get_packed_ue8m0_pack_factor() -> int:
    return 2


def get_sf_shape(shape, config: CastOutputConfig):
    # Each int16 packs two scales; the kernel writes the underlying byte view
    return shape[1] // config.sf_block[1] // 2, shape[0] * 2
