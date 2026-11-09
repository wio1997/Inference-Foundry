"""Construct attention-specific metadata only when an attention backend needs it."""
from functools import lru_cache


@lru_cache(maxsize=1)
def _schedule_type():
    from deepinfer import PLATFORM

    if PLATFORM.is_910c_reference:
        from deepinfer.npu_ops.attention910c import ReferenceMLASchedMeta

        return ReferenceMLASchedMeta
    from flash_mla.flash_mla_interface import FlashMLASchedMeta

    return FlashMLASchedMeta


def new_attention_schedule():
    return _schedule_type()()
