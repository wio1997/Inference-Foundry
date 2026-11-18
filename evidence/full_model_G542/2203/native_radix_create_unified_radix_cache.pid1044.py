from __future__ import annotations

def create_unified_radix_cache(ctx: TreeCacheBuildContext, *, cache_class: type[UnifiedRadixCache] | None=None) -> BasePrefixCache:
    """Initialize a UnifiedRadixCache with proper components and optional HiCache."""
    server_args, params = (ctx.server_args, ctx.params)
    if get_disagg().disaggregation_decode_retraction_backup == 'host_pool':
        if ctx.is_hybrid_ssm:
            raise ValueError('Host-pool retraction does not support Mamba models.')
        if ctx.is_hybrid_swa and ctx.full_tokens_per_layer == 0:
            raise ValueError('Host-pool retraction does not support pure-SWA models.')
    from sglang.srt.mem_cache.kv.plan import TREE_BLOCK, TREE_STATE, TreeSlot
    from sglang.srt.mem_cache.static_kv_storage import kv_storage_of
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    allocator = params.token_to_kv_pool_allocator
    storage = kv_storage_of(params.req_to_token_pool, allocator, shape=_static_tree_shape(ctx, allocator))
    tree_slots = list(storage.tree_slots(window_reach=params.draft_swa_window))
    if hasattr(params.req_to_token_pool, 'req_to_c128_sidecar') and _ds41_has_real_c128(ctx):
        from sglang.srt.hardware_backend.npu.dsv4.c128_sidecar_component import C128SidecarComponent
        tree_slots.append(TreeSlot(component='C128', family=TREE_BLOCK))
        params.component_registry_override = {**(params.component_registry_override or {}), TREE_BLOCK: C128SidecarComponent}
    params.tree_slots = tuple(tree_slots)
    if use_mlx() and ctx.is_hybrid_ssm:
        from sglang.srt.hardware_backend.mlx.kv_cache.auxiliary_state import MlxAuxiliaryStateComponent
        params.component_registry_override = {TREE_STATE: MlxAuxiliaryStateComponent}
    cache = (cache_class or UnifiedRadixCache)(params)
    if ctx.enable_hierarchical_cache or get_disagg().disaggregation_decode_retraction_backup == 'host_pool':
        cache.init_hicache(server_args, params)
        ctx.tp_worker.register_hicache_layer_transfer_counter(cache.cache_controller.layer_done_counter)
    elif get_memory().enable_unified_cache_external_linker:
        from sglang.srt.mem_cache.hicache_storage import PoolName
        from sglang.srt.mem_cache.unified_cache.linker_factory import create_unified_cache_linker, get_startup_linker_config
        linker = create_unified_cache_linker(server_args, params, backend=get_memory().unified_cache_external_linker_backend, config=get_startup_linker_config(), components=set(cache.components), forward_stream=ctx.tp_worker.model_runner.forward_stream)
        cache.init_cache_linker(linker)
        counter = linker.layer_done_counter
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        kvcache.register_layer_transfer_counter(counter)
        if PoolName.MAMBA in linker.pools:
            kvcache.layer_router.load_parts_layer_wise()
        ctx.tp_worker.register_hicache_layer_transfer_counter(counter)
    return cache
