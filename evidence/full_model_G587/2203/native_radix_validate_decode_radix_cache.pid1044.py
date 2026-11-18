from __future__ import annotations

def validate_decode_radix_cache(*, model_config: ModelConfig, token_to_kv_pool: object, is_hybrid_swa: bool, enable_hierarchical_cache: bool) -> None:
    """Reject decode-side radix cache on KV layouts its prefix reuse cannot serve.

    SWA models go through the unified tree, whose component pools preserve the
    full-attention prefix while the SWA window is transferred fresh. Hybrid
    SSM/KDA uses UnifiedRadixCache's Mamba component (match + lock + CoW), the
    same path as colocated serving.
    """
    if not is_hybrid_swa:
        return
    if enable_hierarchical_cache:
        raise ValueError('--disaggregation-decode-enable-radix-cache with sliding window attention (SWA) models currently supports only device-resident cache and is incompatible with --enable-hierarchical-cache.')
    if (getattr(model_config, 'is_deepseek_v4_arch', False) and (not getattr(token_to_kv_pool, '_unified_kv', False))) and (not _ds41_lowratio_supported(model_config, token_to_kv_pool)):
        raise ValueError('--disaggregation-decode-enable-radix-cache with DeepSeek-V4 requires the unified KV layout (SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton).')
    if getattr(model_config, 'is_hybrid_swa_compress', False):
        raise ValueError('--disaggregation-decode-enable-radix-cache does not support SWA-compress models (e.g. Gemma4 / MiMo-V2) yet.')
