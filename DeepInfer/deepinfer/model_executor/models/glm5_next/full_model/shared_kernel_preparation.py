"""Prepare finite GLM SDK JIT templates before serving, identically in both arms.

This module does not import DeepInfer and may be copied unchanged to the reference
plugin. It compiles existing kernels with Triton's warmup=True API; it does not
launch them, run model forwards, consume evaluation prompts, or change math.
The helper is deliberately bounded to the pinned SDK and current <=8-row setup.
"""
from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import time


PINS = {
    'compute_slot_mapping': 'f349fdd683bc36ef3f13b78b91c0e84b930d4850f172220e2627e5ea98de996b',
    'glm5_next_lightning_indexer': '001db81395feab10e83c1dd593d6f9716d47f8d9f191b935c744ed1774a86ba0',
    'triton_jit': 'ae77892fa64fedb5209101beba757adaa3f3c46329a1131346f1fb454799e0b8',
}


def request_powers(max_requests):
    if not 1 <= max_requests <= 8:
        raise ValueError('preparation is bounded to 1..8 requests')
    return sorted({1 << (n - 1).bit_length() for n in range(1, max_requests + 1)})


def slot_tiles(max_requests, max_tokens):
    request_powers(max_requests)
    if not 1 <= max_tokens <= 2048:
        raise ValueError('preparation is bounded to 1..2048 step tokens')
    # num_reqs >= 2 is required by the adaptive branch. For tile>16, two
    # requests need at least tile+1 total tokens to select that power of two.
    return [tile for tile in (16, 32, 64, 128, 256, 512)
            if max_requests >= 2 and (2 if tile == 16 else tile + 1) <= max_tokens]


def indexer_variants(seed, max_requests, max_tokens, max_context, constants):
    """Vary only actual finite launch branches; keep observed tensor contracts."""
    families = [dict(seed)]
    if max_tokens >= constants.TRITON_PREFILL_MIN_TOKENS:
        prefill = dict(seed, OUTER_POOL=max(constants.TRITON_PREFILL_POOL_TILE, seed['BLOCK_POOL']))
        families.append(prefill)
        # Packing is legal only for the source's INDEX32 path. BF16 cache is
        # verified at installation; this capacity check is config-derived.
        pool_count = (max_context + seed['POOL'] - 1) // seed['POOL']
        if seed['INDEX32'] and max_requests * pool_count * seed['HEAD_DIM'] * 2 <= constants.TRITON_PACKED_CACHE_BYTES:
            families.append(dict(prefill, PACKED_CACHE=True, cache_stride_b=0,
                                 cache_stride_t=seed['HEAD_DIM'], cache_stride_d=1))
    seen = set()
    for family in families:
        for power in request_powers(max_requests):
            item = dict(family, REQ_POW2=power)
            # Tensor values in the seed have already become dtype objects.
            key = tuple((k, str(v)) for k, v in item.items())
            if key not in seen:
                seen.add(key)
                yield item


def _save(path, receipt):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def install_shared_kernel_preparation(stats_dir):
    """Call before worker compilation; the same function is used in both arms."""
    from vllm_ascend.worker.worker import NPUWorker
    if getattr(NPUWorker, '_glm53_shared_kernel_preparation', False):
        raise RuntimeError('shared preparation already installed')
    original = NPUWorker.compile_or_warm_up_model

    @functools.wraps(original)
    def prepare(worker, *args, **kwargs):
        import torch
        jit = importlib.import_module('triton.runtime.jit')
        from vllm_ascend.ops.triton import compute_slot_mapping as slots
        from vllm_ascend.ops.triton import glm5_next_lightning_indexer as indexer
        from vllm.v1.attention.backends.utils import PAD_SLOT_ID

        for name, module in [('compute_slot_mapping', slots),
                             ('glm5_next_lightning_indexer', indexer), ('triton_jit', jit)]:
            if hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest() != PINS[name]:
                raise RuntimeError('shared preparation source drift: ' + name)
        config = worker.vllm_config
        max_requests = config.scheduler_config.max_num_seqs
        max_tokens = config.scheduler_config.max_num_batched_tokens
        max_context = config.model_config.max_model_len
        tiles = slot_tiles(max_requests, max_tokens)
        request_powers(max_requests)
        kernel = indexer._glm5_next_lightning_indexer_score_kernel
        seeds = {}

        def capture(*values, **named):
            bound = dict(zip(kernel.arg_names, values))
            bound.update({k: v for k, v in named.items() if k in kernel.arg_names})
            if set(bound) != set(kernel.arg_names):
                raise RuntimeError('incomplete actual indexer signature')
            if bound['cache'].dtype != torch.bfloat16:
                raise RuntimeError('preparation capacity contract requires BF16 cache')
            # Retain types and scalar launch metadata only, never live tensors.
            bound = {k: v.dtype if isinstance(v, torch.Tensor) else v for k, v in bound.items()}
            key = tuple((name, str(bound[name])) for name in (
                'cache_stride_b', 'cache_stride_t', 'cache_stride_d', 'table_stride_r',
                'table_stride_p', 'CACHE_BLOCK', 'HEAD_DIM', 'POOL', 'BLOCK_POOL',
                'INDEX32', 'PACKED_CACHE', 'OUTER_POOL'))
            seeds.setdefault(key, bound)
            if len(seeds) > 4:
                raise RuntimeError('unbounded indexer template families')

        kernel.add_pre_run_hook(capture)
        try:
            result = original(worker, *args, **kwargs)
        finally:
            kernel.pre_run_hooks.remove(capture)
        if not seeds:
            raise RuntimeError('SDK warmup did not expose an indexer signature')

        started = time.monotonic_ns()
        path = Path(stats_dir) / f'rank{worker.rank}-kernel-preparation.json'
        receipt = {'rank': worker.rank, 'scope': 'shared compile-only startup preparation',
                   'status': 'preparing', 'model_forwards_added': 0, 'kernel_launches_added': 0,
                   'max_requests': max_requests, 'max_step_tokens': max_tokens,
                   'max_context': max_context, 'source_pins': PINS,
                   'prepared': [], 'late_compile_events': []}
        _save(path, receipt)
        table = worker.model_runner.input_batch.block_table
        if not table._can_fuse_slot_mapping:
            raise RuntimeError('pinned GLM fused slot mapping not active')
        slot_kernel = slots._compute_slot_mapping_fused_groups_adaptive_kernel
        minimum = table._fused_min_block_size
        for tile in tiles:
            compiled = slot_kernel.warmup(
                1, table._fused_max_num_batched_tokens, torch.int32, torch.int64,
                table._fused_block_table_addrs.dtype, table._fused_slot_mapping_addrs.dtype,
                table._fused_block_table_strides.dtype, table._fused_block_sizes.dtype,
                table._fused_is_circular.dtype, HAS_CIRCULAR=True, PAD_ID=PAD_SLOT_ID,
                SMALL_TILE_BLOCK_SIZE=tile,
                SMALL_BLOCK_TABLE_WINDOW_SIZE=slots._next_power_of_2((tile + minimum - 1) // minimum + 1),
                LARGE_BLOCK_TABLE_WINDOW_SIZE=slots._next_power_of_2((1024 + minimum - 1) // minimum + 1),
                grid=(table._fused_slot_mapping_group_count, max_requests + 1))
            receipt['prepared'].append({'kernel': slot_kernel.__name__, 'tile': tile, 'hash': compiled.hash})
        seen = set()
        for seed in seeds.values():
            for variant in indexer_variants(seed, max_requests, max_tokens, max_context, indexer):
                values = [variant[name] for name in kernel.arg_names]
                key = tuple(map(str, values))
                if key in seen:
                    continue
                seen.add(key)
                if len(seen) > 24:
                    raise RuntimeError('indexer preparation exceeded bounded family budget')
                compiled = kernel.warmup(*values, grid=(1,))
                receipt['prepared'].append({'kernel': kernel.__name__, 'request_power': variant['REQ_POW2'],
                    'packed': variant['PACKED_CACHE'], 'outer_pool': variant['OUTER_POOL'], 'hash': compiled.hash})
        receipt.update(status='prepared', preparation_ms=(time.monotonic_ns() - started) / 1e6)
        _save(path, receipt)

        # Called on compilation cache misses only, never on every model step.
        # Both benchmark arms receive this same observation. Persist at once,
        # so non-driver os._exit cannot lose evidence as it did in diagnostic97.
        def observe_late(fn):
            original_compile = fn.compile
            def compile_late(src, *a, **kw):
                begin = time.monotonic_ns()
                try:
                    return original_compile(src, *a, **kw)
                finally:
                    receipt['late_compile_events'].append({'kernel': fn.__name__,
                        'elapsed_ms': (time.monotonic_ns() - begin) / 1e6,
                        'constants': {str(k): v for k, v in src.constants.items()}})
                    _save(path, receipt)
            fn.compile = compile_late
        observe_late(slot_kernel)
        observe_late(kernel)
        return result

    NPUWorker.compile_or_warm_up_model = prepare
    NPUWorker._glm53_shared_kernel_preparation = True
