# SPDX-License-Identifier: Apache-2.0
"""DeepInfer-owned paged caches for packed 910C prefill and decode.

SWA uses a ring of physical pages large enough for an entire query chunk
plus its history. Compressed source KV and index K remain linear and are
shared by downstream layers. Cache/page management imports no NPU runtime.
Host write plans reject overflow/aliasing before constructing device indices;
the actual write is one index_copy_, with no per-token device synchronization.
"""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class CacheWritePlan:
    physical_rows: torch.Tensor
    count: int


class PagedPlane:
    def __init__(self, *, requests, logical_rows, block_size, width, dtype,
                 device, resident_pages=None):
        if min(requests, logical_rows, block_size, width) < 1:
            raise ValueError('cache dimensions must be positive')
        self.requests, self.logical_rows = requests, logical_rows
        self.block_size, self.width = block_size, width
        self.logical_pages = math.ceil(logical_rows/block_size)
        self.resident_pages = self.logical_pages if resident_pages is None else min(resident_pages, self.logical_pages)
        if self.resident_pages < 1:
            raise ValueError('cache requires at least one resident page')
        self.cache = torch.zeros((requests*self.resident_pages, block_size, 1, width), dtype=dtype, device=device)
        page = torch.arange(self.logical_pages, dtype=torch.int32, device=device).remainder(self.resident_pages)
        request = torch.arange(requests, dtype=torch.int32, device=device)[:, None]
        self.block_table = page[None]+request*self.resident_pages
        self._active_tables = {}

    def plan(self, positions, request_ids):
        """Host token coordinates -> validated contiguous device write indices."""
        positions, request_ids = list(positions), list(request_ids)
        if len(positions) != len(request_ids) or any(type(x) is not int for x in (*positions, *request_ids)):
            raise ValueError('equal-length host integer positions and request slots required')
        if any(not 0 <= p < self.logical_rows for p in positions) or any(not 0 <= s < self.requests for s in request_ids):
            raise ValueError('cache write outside logical capacity or request slots')
        indices = [((s*self.resident_pages+(p//self.block_size)%self.resident_pages)*self.block_size+p%self.block_size)
                   for p, s in zip(positions, request_ids)]
        if len(indices) != len(set(indices)):
            raise ValueError('write plan aliases physical rows; reduce chunk size or increase ring capacity')
        return CacheWritePlan(torch.tensor(indices, dtype=torch.int64, device=self.cache.device), len(indices))

    def write(self, values, plan):
        if (values.shape != (plan.count, 1, self.width) or values.dtype != self.cache.dtype
                or values.device != self.cache.device or plan.physical_rows.device != self.cache.device):
            raise ValueError('cache payload does not match write plan/dtype/device')
        self.cache.flatten(0, 1).index_copy_(0, plan.physical_rows, values)

    def table(self, request_slots):
        slots = tuple(request_slots)
        if any(type(s) is not int or not 0 <= s < self.requests for s in slots):
            raise ValueError('invalid active request slots')
        if not slots:
            return self.block_table[:0]
        # Serving slots are normally contiguous. The table is immutable, so
        # return its device view instead of synchronously uploading four IDs
        # again at every attention/indexer layer.
        if slots == tuple(range(slots[0], slots[0]+len(slots))):
            return self.block_table[slots[0]:slots[0]+len(slots)]
        if slots in self._active_tables:
            return self._active_tables[slots]
        indices = torch.tensor(slots, dtype=torch.int64, device=self.cache.device)
        table = self.block_table.index_select(0, indices).contiguous()
        if len(self._active_tables) >= 32:
            self._active_tables.pop(next(iter(self._active_tables)))
        self._active_tables[slots] = table
        return table

    def physical_indices(self, logical_rows, request_ids):
        """Translate padded per-query logical rows for the native eager path."""
        valid = (logical_rows >= 0) & (logical_rows < self.logical_rows)
        safe = logical_rows.clamp(0, self.logical_rows-1)
        pages = self.block_table[request_ids.long()[:, None], torch.div(safe, self.block_size, rounding_mode='floor').long()]
        physical = pages*self.block_size+safe.remainder(self.block_size)
        return torch.where(valid, physical, -1).int()


class W8A8PagedCache:
    def __init__(self, config, device, *, max_tokens, requests, max_query_tokens):
        c = config['text_config']
        if max_tokens < 1 or requests < 1 or max_query_tokens < 1 or c['head_dim'] != 512:
            raise ValueError('invalid 910C paged cache capacity or head dimension')
        self.max_tokens, self.requests = max_tokens, requests
        self.max_query_tokens = max_query_tokens
        self.config = c
        # An interval may begin near a page end; include the extra physical
        # page for that alignment, in addition to the preceding SWA window.
        swa_pages = math.ceil((max_query_tokens+c['sliding_window']+128-2)/128)
        self.swa = {i: PagedPlane(requests=requests, logical_rows=max_tokens, block_size=128,
                                 width=512, dtype=torch.bfloat16, device=device,
                                 resident_pages=swa_pages)
                    for i in range(c['num_hidden_layers'])}
        self.compressed, self.index_key, self.index_scale = {}, {}, {}
        for source in c['kv_source_layer_ids']:
            ratio = c['compress_ratios'][source]
            if ratio not in (1, 2):
                raise ValueError('compressed KV source requires ratio one or two')
            common = dict(requests=requests, logical_rows=max(1, max_tokens//ratio),
                          block_size=128//ratio, device=device)
            self.compressed[source] = PagedPlane(width=512, dtype=torch.bfloat16, **common)
            self.index_key[source] = PagedPlane(width=128, dtype=torch.int8, **common)
            self.index_scale[source] = PagedPlane(width=1, dtype=torch.float16, **common)

    def swa_plan(self, layer_id, batch, request_slots):
        slots = list(request_slots)
        if len(slots) != len(batch.query_lengths) or len(set(slots)) != len(slots):
            raise ValueError('one distinct request slot per packed query required')
        if max(batch.query_lengths) > self.max_query_tokens:
            raise ValueError('query chunk exceeds SWA ring capacity')
        positions, request_ids = [], []
        for q, s, slot in zip(batch.query_lengths, batch.seq_lengths, slots):
            positions.extend(range(s-q, s))
            request_ids.extend([slot]*q)
        return self.swa[layer_id].plan(positions, request_ids)

    def compressed_plans(self, source, source_positions, request_ids):
        ratio = self.config['compress_ratios'][source]
        positions = list(source_positions)
        request_ids = list(request_ids)
        if any(type(p) is not int or p % ratio for p in positions):
            raise ValueError('compressor source position must begin a completed group')
        logical = [p//ratio for p in positions]
        # All three source planes have identical page coordinates. Their
        # payload widths/dtypes differ, but their write indices do not.
        plan = self.compressed[source].plan(logical, request_ids)
        return (plan, plan, plan)

    def prepare_compressed(self, batch, request_slots):
        """Stage source coordinates before model compute, once per ratio.

        An odd-start C2 chunk completes the pending pair from the previous
        chunk; a trailing even token does not write a completed cache row.
        """
        slots = tuple(request_slots)
        if len(slots) != len(batch.query_lengths) or len(set(slots)) != len(slots):
            raise ValueError('one distinct request slot per packed query required')
        by_ratio, result = {}, {}
        for source in self.compressed:
            ratio = self.config['compress_ratios'][source]
            if ratio not in by_ratio:
                positions, requests = [], []
                for q, end, slot in zip(batch.query_lengths, batch.seq_lengths, slots):
                    start = end-q
                    rows = list(range(start, end)) if ratio == 1 else list(range(start//2*2, end-1, 2))
                    positions.extend(rows)
                    requests.extend([slot]*len(rows))
                device_positions = torch.tensor(positions, dtype=torch.int32, device=batch.seq_used.device)
                plans = self.compressed_plans(source, positions, requests)
                by_ratio[ratio] = (tuple(positions), device_positions, plans)
            result[source] = by_ratio[ratio]
        return result

    def resident_bytes(self):
        """Payload bytes; excludes small page-table/host-plan allocations."""
        return sum(plane.cache.numel()*plane.cache.element_size()
                   for mapping in (self.swa, self.compressed, self.index_key, self.index_scale)
                   for plane in mapping.values())
