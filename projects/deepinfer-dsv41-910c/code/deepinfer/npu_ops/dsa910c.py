# SPDX-License-Identifier: Apache-2.0
"""Standalone CANN DSA operators for DeepInfer on 910C.

The compiled operators come from the pinned vLLM-Ascend image. No vLLM
Python package, engine, scheduler or forward context is imported. Cache
coordinates and batching belong to DeepInfer. Importing this module is CPU
safe; loading the extension and executing an operator are explicit actions.

ABI: BF16 SWA/compressed MLA; INT8 index keys, FP16 index scales. A storage
page contains 128 original tokens, hence 128//ratio compressed rows. See
the image source models/deepseek_v41/{cache_config,indexer}.py and
attention/dsa_v41.py (Apache-2.0) for the operator ABI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import importlib.machinery
import importlib.util
import os
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def load_ops():
    """Register just the extension BEFORE the first NPU device initialization.

    CANN caches its custom OPP search path at runtime initialization. Setting
    the path after set_device can register Python schemas successfully while
    dispatching old system kernels or failing to find custom tiling code.
    """
    import torch_npu  # noqa: F401

    package = importlib.util.find_spec('vllm_ascend')
    if package is None or package.origin is None:
        raise RuntimeError('the pinned Ascend operator package is unavailable')
    root = Path(package.origin).parent
    vendor = root / '_cann_ops_custom/vendors/custom_transformer'
    if vendor.is_dir():
        if (torch.npu.is_initialized()
                and str(vendor) not in os.environ.get('ASCEND_CUSTOM_OPP_PATH', '').split(':')):
            raise RuntimeError('call dsa910c.load_ops() before initializing the NPU runtime')
        for name, path in [('ASCEND_CUSTOM_OPP_PATH', vendor),
                           ('LD_LIBRARY_PATH', vendor / 'op_api/lib')]:
            entries = [entry for entry in os.environ.get(name, '').split(':') if entry]
            if str(path) not in entries:
                os.environ[name] = ':'.join([str(path), *entries])
    files = [root / ('vllm_ascend_C' + suffix)
             for suffix in importlib.machinery.EXTENSION_SUFFIXES
             if (root / ('vllm_ascend_C' + suffix)).is_file()]
    if len(files) != 1:
        raise RuntimeError(f'expected one Ascend extension, found {files}')
    spec = importlib.util.spec_from_file_location('vllm_ascend_C', files[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Retain the module for the lifetime of the registered C++ operators.
    return module


@dataclass(frozen=True)
class DSABatch:
    """Host-known request lengths plus matching device INT32 metadata.

    seq_lengths includes cached tokens and the entire current query chunk.
    For C2 the remainder is essential: two chunks ending at 32768/32769
    have identical compressed lengths but different causal query positions.
    """
    query_lengths: tuple[int, ...]
    seq_lengths: tuple[int, ...]
    cu_query: torch.Tensor
    seq_used: torch.Tensor
    metadata_cache: dict = field(default_factory=dict, compare=False, repr=False)
    position_values: torch.Tensor | None = field(default=None, compare=False, repr=False)

    @classmethod
    def build(cls, query_lengths, seq_lengths, device):
        q, s = tuple(query_lengths), tuple(seq_lengths)
        if not q or len(q) != len(s) or any(type(x) is not int for x in (*q, *s)):
            raise ValueError('nonempty equal-length integer request lengths required')
        if any(a < 1 or b < a for a, b in zip(q, s)):
            raise ValueError('each sequence must contain its positive query chunk')
        cumulative = [0]
        for n in q:
            cumulative.append(cumulative[-1] + n)
        return cls(q, s, torch.tensor(cumulative, dtype=torch.int32, device=device),
                   torch.tensor(s, dtype=torch.int32, device=device))

    def compressed(self, ratio):
        if ratio not in (1, 2):
            raise ValueError('compressed ratio must be one or two')
        if ratio == 1:
            return self.seq_used, None
        return torch.div(self.seq_used, 2, rounding_mode='floor'), self.seq_used.remainder(2)

    def positions(self):
        if self.position_values is not None:
            return self.position_values
        if 'positions' not in self.metadata_cache:
            self.metadata_cache['positions'] = torch.cat([
                torch.arange(s-q, s, device=self.seq_used.device, dtype=torch.int32)
                for q, s in zip(self.query_lengths, self.seq_lengths)])
        return self.metadata_cache['positions']


def quantize_query(query):
    """Reference vector path: FP16-rounded scales BEFORE INT8 query rounding.

    Index key quantization differs: npu_dynamic_quant rounds with FP32 scales
    and the stored scale is converted to FP16 afterwards. Mixing these rules
    silently changes index scores. This path is vectorized but not yet fused.
    """
    if query.ndim != 3 or query.shape[-1] != 128 or not query.is_floating_point():
        raise ValueError('index query must be floating [tokens,heads,128]')
    value = query.float()
    scale = (value.abs().amax(-1) / 127.).half().float().clamp_min(2.**-24)
    codes = (value / scale[..., None]).round().clamp(-127, 127).to(torch.int8)
    return codes.contiguous(), scale.half().contiguous()


def prepare_indices(selected, positions, ratio):
    """Causal chronological indices; FP32 sort avoids an INT64 AICPU sort.

    Restricted to exactly representable token positions (<2**24). A Triton
    fused implementation can replace this after independent parity testing.
    """
    if ratio not in (1, 2) or selected.ndim != 2 or selected.dtype != torch.int32:
        raise ValueError('expected INT32 index rows and compression ratio one/two')
    if positions.shape != selected.shape[:1] or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError('one integer position per index row required')
    visible = torch.div(positions+1, ratio, rounding_mode='floor')[:, None]
    valid = (selected >= 0) & (selected < visible) & (selected < 2**24)
    ordered = torch.where(valid, selected, 2**24).float().sort(-1).values
    return torch.where(ordered < 2**24, ordered.to(torch.int32), -1)


def page_rows(rows, block_size):
    """Pack a single logical cache for diagnostics; production owns page pools."""
    if rows.ndim < 2 or block_size < 1:
        raise ValueError('cache rows and a positive page size required')
    blocks = (rows.shape[0]+block_size-1)//block_size
    if not blocks:
        raise ValueError('empty cache has no physical page')
    cache = rows.new_zeros((blocks*block_size, *rows.shape[1:]))
    cache[:rows.shape[0]].copy_(rows)
    table = torch.arange(blocks, dtype=torch.int32, device=rows.device)[None]
    return cache.reshape(blocks, block_size, *rows.shape[1:]), table


def select(query, key, weights, query_scale, key_scale, batch, block_table,
           *, ratio, candidate_mode=3, candidates=None, topk=512,
           candidate_topk_blocks=2048, candidate_block_size=8):
    """Native paged QLI including C2 residual and cross-layer candidates."""
    if (query.ndim != 3 or query.dtype != torch.int8 or query.shape[-1] != 128
            or key.ndim != 4 or key.shape[-2:] != (1, 128) or key.dtype != torch.int8
            or weights.shape != query.shape[:2] or weights.dtype != torch.float16
            or query_scale.shape != query.shape[:2] or query_scale.dtype != torch.float16
            or key_scale.shape != key.shape[:3] or key_scale.dtype != torch.float16
            or query.shape[0] != sum(batch.query_lengths)):
        raise ValueError('invalid INT8/FP16 QLI ABI shapes or dtypes')
    if candidate_mode not in (1, 2, 3) or (candidate_mode == 2) != (candidates is not None):
        raise ValueError('only candidate-filter mode takes candidate block indices')
    if block_table.dtype != torch.int32 or block_table.shape[0] != len(batch.seq_lengths):
        raise ValueError('INT32 page table must match request count')
    load_ops()
    seq, residual = batch.compressed(ratio)
    metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
        query.shape[1], 1, 128, topk, 2, cu_seqlens_q=batch.cu_query,
        seqused_k=seq, cmp_residual_k=residual, batch_size=len(batch.seq_lengths),
        max_seqlen_q=max(batch.query_lengths), max_seqlen_k=max(batch.seq_lengths)//ratio,
        layout_q='TND', layout_k='PA_BBND', mask_mode=3, cmp_ratio=ratio)
    selected, _, emitted = torch.ops._C_ascend.npu_quant_lightning_indexer_v3(
        query, key, weights, query_scale, key_scale, topk, 2,
        candidate_topk_index=candidates, cu_seqlens_q=batch.cu_query,
        seqused_k=seq, cmp_residual_k=residual, block_table=block_table,
        metadata=metadata, max_seqlen_q=max(batch.query_lengths),
        layout_q='TND', layout_k='PA_BBND', mask_mode=3, cmp_ratio=ratio,
        candidate_mode=candidate_mode, candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size)
    return prepare_indices(selected.squeeze(1), batch.positions(), ratio), emitted


def sparse_mla(query, swa_cache, batch, swa_table, *, sink, compressed_cache=None,
               compressed_table=None, compressed_indices=None, ratio=0, window=128):
    """Paged BF16 SWA + source-shared sparse compressed MLA in one CANN call."""
    has_compressed = ratio in (1, 2)
    if ratio not in (0, 1, 2) or query.ndim != 3 or query.shape[-1] != 512 or query.dtype != torch.bfloat16:
        raise ValueError('expected BF16 [tokens,heads,512] and ratio zero/one/two')
    if query.shape[0] != sum(batch.query_lengths) or window < 1:
        raise ValueError('query chunk or sliding window mismatch')
    if swa_cache.ndim != 4 or swa_cache.shape[-2:] != (1, 512) or swa_cache.dtype != torch.bfloat16:
        raise ValueError('SWA cache must be BF16 [pages,128,1,512]')
    if has_compressed:
        if (compressed_cache is None or compressed_table is None or compressed_indices is None
                or compressed_cache.ndim != 4 or compressed_cache.shape[-2:] != (1, 512)
                or compressed_cache.dtype != torch.bfloat16
                or compressed_indices.shape != (query.shape[0], 512)
                or compressed_indices.dtype != torch.int32):
            raise ValueError('compressed MLA requires BF16 pages, table and 512 INT32 indices per query')
    elif any(x is not None for x in (compressed_cache, compressed_table, compressed_indices)):
        raise ValueError('SWA-only MLA cannot take compressed inputs')
    load_ops()
    seq, residual = batch.compressed(ratio) if has_compressed else (None, None)
    # This pinned image's custom host ABI explicitly requires ratio0 when
    # has_cmp_kv=False; the public CANN9.1 default1 belongs to another version.
    operator_ratio = ratio
    common = dict(cu_seqlens_q=batch.cu_query, seqused_ori_kv=batch.seq_used,
                  seqused_cmp_kv=seq, cmp_residual_kv=residual, cmp_ratio=operator_ratio,
                  ori_mask_mode=4, cmp_mask_mode=3 if has_compressed else 0,
                  ori_win_left=window-1, ori_win_right=0,
                  layout_q='TND', layout_kv='PA_BBND')
    key = (query.shape[1], ratio, window)
    retained = batch.metadata_cache.get(key)
    if retained is None:
        metadata = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
            query.shape[1], 1, 512, batch_size=len(batch.seq_lengths),
            max_seqlen_q=max(batch.query_lengths), max_seqlen_ori_kv=max(batch.seq_lengths),
            max_seqlen_cmp_kv=max(batch.seq_lengths)//ratio if has_compressed else 0,
            ori_topk=0, cmp_topk=512 if has_compressed else 0,
            has_ori_kv=True, has_cmp_kv=has_compressed, **common)
        # Keep metadata and derived lengths alive for the whole model forward,
        # and reuse identical per-batch tiling across source-sharing layers.
        batch.metadata_cache[key] = (metadata, seq, residual)
    else:
        metadata, seq, residual = retained
        common['seqused_cmp_kv'], common['cmp_residual_kv'] = seq, residual
    output, _ = torch.ops._C_ascend.npu_sparse_flash_mla(
        query, ori_kv=swa_cache, cmp_kv=compressed_cache,
        cmp_sparse_indices=compressed_indices[:, None, :] if has_compressed else None,
        ori_block_table=swa_table, cmp_block_table=compressed_table,
        sinks=sink, metadata=metadata, softmax_scale=512**-.5,
        topk_value_mode=1, return_softmax_lse=False, **common)
    return output
