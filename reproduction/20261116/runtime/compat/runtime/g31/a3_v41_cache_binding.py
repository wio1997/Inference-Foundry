"""Target-scoped binding helpers. Imported only by the generated native overlay.

These helpers do not patch modules themselves. Draft workers retain the native
NPU pool and shared target allocator; only their three SWA layers are priced by
the target budget. Whole-model integration is not yet verified.
"""
from sglang.srt.utils import is_npu


def use_a3_v41_target_cache(kvc):
    if getattr(kvc, "is_draft_worker", False) or not is_npu():
        return False
    hf = kvc.model_config.hf_config
    if getattr(hf, "model_type", None) != "deepseek_v41":
        return False
    from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
    if is_npu_arch35():
        return False
    # Guard the exact checkpoint topology, rather than changing unrelated V4
    # or V4.1 models which happen to use the same architecture class.
    cfg = kvc.model_config
    ratios = list(cfg.compress_ratios)
    expected = [0, 0] + [2] * 18 + [1] * 20
    checks = {
        "target_ratios": ratios[:40] == expected and not any(ratios[40:]),
        "target_layers": int(getattr(hf, "num_hidden_layers", 0)) == 40,
        "kv_sources": list(hf.kv_source_layer_ids) == [2, 8, 14, 20],
        "index_sources": list(hf.index_source_layer_ids) == [2, 8, 14, 20, 24, 28, 32, 36],
        "head_geometry": cfg.qk_nope_head_dim == 448 and cfg.qk_rope_head_dim == 64 and cfg.index_head_dim == 128,
        "hidden_size": int(hf.hidden_size) == 5120,
        "query_heads": int(hf.num_attention_heads) == 64,
        "index_heads": int(hf.index_n_heads) == 32,
        "index_topk": int(hf.index_topk) == 512,
        "candidate": (int(hf.candidate_source_layer_id), int(hf.candidate_block_size), int(hf.candidate_topk_blocks)) == (20,8,2048),
        "bf16_kv": kvc.kv_cache_dtype_str == "bfloat16",
        "single_pp_stage": kvc.layer_info.start_layer == 0 and kvc.layer_info.end_layer == 40,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError("A3 V4.1 cache binding topology mismatch: " + ", ".join(failed))
    return True


def select_a3_v41_pool(kvc, native_pool_class):
    if not use_a3_v41_target_cache(kvc):
        return native_pool_class
    from a3_v41_pool_candidate import A3V41TokenToKVPool
    return A3V41TokenToKVPool


def select_a3_v41_allocator(kvc, native_allocator_class):
    if not use_a3_v41_target_cache(kvc):
        return native_allocator_class
    from a3_v41_pool_candidate import A3V41TokenToKVPoolAllocator
    return A3V41TokenToKVPoolAllocator


def create_a3_v41_budget(kvc):
    if not use_a3_v41_target_cache(kvc):
        return None
    layers = 0
    if not kvc.spec_algorithm.is_none():
        if not kvc.spec_algorithm.is_dspark():
            raise ValueError("A3 V4.1 candidate binding requires the verified DSpark draft layout")
        from sglang.srt.speculative.dspark_components.dspark_config import parse_dspark_draft_config
        aux = kvc.spec_aux_config
        draft_cfg = getattr(aux, "draft_model_config", None)
        draft_hf = draft_cfg.hf_config if draft_cfg is not None else kvc.model_config.hf_config
        parsed = parse_dspark_draft_config(draft_hf_config=draft_hf)
        layers = (len(parsed.target_layer_ids) if parsed.target_layer_ids is not None
                  else int(getattr(draft_hf, "num_nextn_predict_layers", 1) or 1))
        if layers != 3:
            raise ValueError(f"Expected the verified 3-stage A3 DSpark draft; got {layers}")
        aux_layers = getattr(aux, "dspark_num_stages", None)
        if aux_layers is not None and int(aux_layers) != layers:
            raise ValueError("DSpark native auxiliary stage count disagrees with model construction")
    from a3_v41_budget_candidate import A3V41PoolConfigurator
    return A3V41PoolConfigurator(kvc, draft_swa_layers=layers)
