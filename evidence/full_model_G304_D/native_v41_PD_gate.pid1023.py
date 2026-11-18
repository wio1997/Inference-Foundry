from __future__ import annotations

def validate_deepseek_v41_features(server_args: ServerArgs) -> None:
    from sglang.srt.layers.attention.dsv4.unified_kv_gate import is_unified_kv_triton
    cfg = resolving_view(server_args)
    _ds41_ascend_scope_check(cfg)
    if model_config_of(server_args).hf_config.model_type != 'deepseek_v41':
        if cfg.enable_encoder_swa_bounded_replay:
            raise ValueError('--enable-encoder-swa-bounded-replay requires DeepSeek-V4.1')
        return
    if cfg.dsv4_attn_backend == 'trtllm' and cfg.cuda_graph_config.prefill.backend not in (Backend.DISABLED, Backend.BREAKABLE):
        raise ValueError('DeepSeek-V4.1 TRT-LLM requires --cuda-graph-backend-prefill disabled or breakable')
    if cfg.enable_encoder_swa_bounded_replay:
        incompatible = (('hardware other than CUDA or gfx950', not (get_platform().is_cuda or (get_platform().is_hip and is_gfx95_supported()))), ('prefill CUDA graphs other than breakable', cfg.cuda_graph_config.prefill.backend not in (Backend.DISABLED, Backend.BREAKABLE)), ('the prefill CUDA graph on ROCm', cfg.cuda_graph_config.prefill.backend != Backend.DISABLED and get_platform().is_hip), ('context parallelism', cfg.attn_cp_size > 1), ('DP attention on ROCm', get_platform().is_hip and attn_dp_enabled_of(cfg)), ('unified memory', cfg.enable_unified_memory), ('PD decode', cfg.disaggregation_mode == 'decode'), ('PD prefill on ROCm', cfg.disaggregation_mode == 'prefill' and get_platform().is_hip), ('mixed prefill/decode', cfg.enable_mixed_chunk), ('LoRA', cfg.enable_lora), ('radix sessions', cfg.enable_session_radix_cache))
        for feature, enabled in incompatible:
            if enabled:
                raise ValueError(f'--enable-encoder-swa-bounded-replay does not support {feature} yet')
    unsupported = (('speculative decoding other than DSpark', cfg.speculative_algorithm is not None and str(cfg.speculative_algorithm).upper() != 'DSPARK'), ('HiSparse', cfg.enable_hisparse), ('the unified KV layout', is_unified_kv_triton()), ('TRT-LLM with encoder SWA bounded replay', cfg.dsv4_attn_backend == 'trtllm' and cfg.enable_encoder_swa_bounded_replay), ('two-batch overlap', cfg.enable_two_batch_overlap), ('pipeline parallelism', cfg.pp_size > 1))
    for feature, enabled in unsupported:
        if enabled:
            raise ValueError(f'DeepSeek-V4.1 does not support {feature} yet; disable it to serve this model.')
    if cfg.disaggregation_mode != 'null' and cfg.speculative_algorithm is not None:
        from sglang.srt.speculative.ragged_verify import RaggedVerifyMode, read_ragged_verify_mode
        supported_prefill_cp = cfg.disaggregation_mode == 'prefill' and cfg.enable_prefill_cp and (cfg.cp_strategy == 'interleave')
        if read_ragged_verify_mode() is not RaggedVerifyMode.STATIC or cfg.disaggregation_transfer_backend not in ('mooncake', 'nixl', 'mori', 'ascend') or ((cfg.enable_prefill_cp or cfg.attn_cp_size != 1) and (not supported_prefill_cp)) or (cfg.dcp_size != 1):
            raise ValueError('DeepSeek-V4.1 DSpark PD requires static verify, Mooncake, NIXL, or MoRI, DCP=1, and CP=1 on decode. Prefill CP requires --enable-prefill-cp with the interleave strategy. DP attention is supported when both servers use the same block size and target/draft KV layout.')
    incompatible = (('the prefill CUDA graph', cfg.cuda_graph_config.prefill.backend not in (Backend.DISABLED, Backend.BREAKABLE)), ('DP attention other than attention TP 1 with supported MoE backends', attn_dp_enabled_of(cfg) and (not (cfg.attn_dp_size == cfg.tp_size and cfg.attn_cp_size == 1 and (get_platform().is_hip and cfg.moe_a2a_backend == 'none' or (get_platform().is_cuda and cfg.moe_a2a_backend in ('none', 'megamoe')))))))
    if cfg.enable_decoder_swa_bounded_replay:
        for feature, unsupported in incompatible:
            if unsupported:
                raise ValueError(f'--enable-decoder-swa-bounded-replay cannot be combined with {feature} yet; disable one of them.')
        return
    if cfg.enable_decoder_swa_bounded_replay is False:
        return
    enabled = (get_platform().is_cuda or get_platform().is_hip) and (cfg.disaggregation_mode == 'null' or get_platform().is_cuda) and (not (cfg.enable_prefill_cp or cfg.attn_cp_size != 1 or cfg.dcp_size != 1)) and (not cfg.enable_return_hidden_states) and (not any((enabled for _, enabled in incompatible)))
    declare_resolution(server_args, 'validate_deepseek_v41_features', enable_decoder_swa_bounded_replay=enabled)
    logger.info('DeepSeek-V4.1 decoder SWA bounded replay resolved to %s', enabled)
