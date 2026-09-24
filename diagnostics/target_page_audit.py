"""Opt-in source-backed target physical page coverage for fixed c12."""

from __future__ import annotations

import json
from pathlib import Path

import torch


class TargetPageAudit:
    def __init__(self, *, assets, attn_metadata, static_context, group_bindings,
                 metadata_sources, output_dir, rank, limit=2):
        self.assets = assets
        self.attn_metadata = attn_metadata
        self.output_dir = Path(output_dir)
        self.rank = rank
        self.limit = limit
        self.rows = []
        self.table_to_gid = {
            int(table.data_ptr()): gid
            for gid, (table, _mapping, _size) in enumerate(group_bindings)
        }
        if not self.table_to_gid:
            for source in metadata_sources:
                req = attn_metadata[source.layer_name].req_metadata
                pointer = int(req.block_table.data_ptr())
                self.table_to_gid.setdefault(pointer, len(self.table_to_gid))
        self.ratio_by_gid = {}
        for source in metadata_sources:
            req = attn_metadata[source.layer_name].req_metadata
            gid = self.table_to_gid[int(req.block_table.data_ptr())]
            previous = self.ratio_by_gid.setdefault(gid, int(source.ratio))
            if previous != int(source.ratio):
                raise RuntimeError(f"KV group {gid} has mixed compressor ratios")

        aliases = {}

        def visit(value, layer_name):
            if torch.is_tensor(value):
                aliases.setdefault(int(value.data_ptr()), set()).add(layer_name)
            elif isinstance(value, (tuple, list)):
                for child in value:
                    visit(child, layer_name)

        for layer_name, layer in static_context.items():
            value = getattr(layer, "kv_cache", None)
            if value is not None:
                visit(value, layer_name)
        self.cache_aliases = {}
        for cache in assets.caches:
            names = tuple(sorted(aliases.get(cache.data_ptr, ())))
            if not names:
                raise RuntimeError(f"cache has no layer alias: {cache.name}")
            for name in names:
                metadata = attn_metadata.get(name)
                req = getattr(metadata, "req_metadata", None)
                table = getattr(req, "block_table", None)
                if table is None or int(table.data_ptr()) not in self.table_to_gid:
                    raise RuntimeError(f"cache alias has no target table: {name}")
            self.cache_aliases[cache.name] = names
        if len(self.cache_aliases) != len(assets.caches):
            raise RuntimeError("target cache alias inventory incomplete")

    @staticmethod
    def _table_pages(req, ratio, kind):
        count = int(req.num_reqs_actual)
        starts = req.start_pos[:count].to("cpu", dtype=torch.int64).tolist()
        qsl = req.query_start_loc[:count + 1].to("cpu", dtype=torch.int64).tolist()
        if len(starts) != 12 or len(qsl) != 13 or qsl[0] != 0:
            raise RuntimeError("fixed target request geometry changed")
        row_ids = []
        logical_ids = []
        for row, start in enumerate(starts):
            length = qsl[row + 1] - qsl[row]
            if start < 0 or length < 0:
                raise RuntimeError("negative target start or query length")
            if kind == "compressed":
                first = start // ratio
                end = (start + length) // ratio
            else:
                size = int(req.block_size)
                first = start // size
                end = (start + length - 1) // size + 1 if length else first
            for position in range(first, end):
                logical_ids.append(
                    position // int(req.block_size) if kind == "compressed" else position
                )
                row_ids.append(row)
        if not row_ids:
            return torch.empty(0, dtype=torch.int64, device=req.block_table.device)
        if min(logical_ids) < 0 or max(logical_ids) >= req.block_table.shape[1]:
            raise RuntimeError(f"{kind} target page outside block table")
        device = req.block_table.device
        r = torch.tensor(row_ids, dtype=torch.int64, device=device)
        c = torch.tensor(logical_ids, dtype=torch.int64, device=device)
        pages = torch.unique(req.block_table[r, c].to(torch.int64))
        if bool((pages < 0).any()):
            raise RuntimeError(f"{kind} target page has negative physical block")
        return pages

    def _source_pages(self, layer_name):
        metadata = self.attn_metadata[layer_name]
        req = metadata.req_metadata
        gid = self.table_to_gid[int(req.block_table.data_ptr())]
        if layer_name.endswith("swa_cache"):
            slots = req.slot_mapping
            if slots is None or slots.ndim != 2 or slots.shape[1] != 2:
                raise RuntimeError(f"SWA slot format changed: {layer_name}")
            pages = torch.unique(slots[:int(metadata.num_actual_tokens), 0].to(torch.int64))
            kind = "swa_scatter"
            if bool((pages < 0).any()):
                raise RuntimeError(f"SWA target page negative: {layer_name}")
        elif layer_name.endswith("state_cache"):
            pages = self._table_pages(req, 1, "state")
            kind = "compressor_state"
        elif layer_name.endswith("indexer.k_cache") or layer_name.endswith(".attn"):
            ratio = self.ratio_by_gid[gid]
            if ratio not in (4, 128):
                raise RuntimeError(f"compressed cache ratio invalid: {layer_name}")
            pages = self._table_pages(req, ratio, "compressed")
            kind = "compressed_scatter"
        else:
            raise RuntimeError(f"unclassified cache write source: {layer_name}")
        return gid, kind, pages

    @torch.inference_mode()
    def observe(self, cycle):
        if cycle >= self.limit:
            return
        if cycle != len(self.rows):
            raise RuntimeError("target page audit cycle discontinuity")
        source_cache = {}
        pages_by_cache = {}
        no_write = set()
        for cache in self.assets.caches:
            parts = []
            for name in self.cache_aliases[cache.name]:
                if name not in source_cache:
                    source_cache[name] = self._source_pages(name)
                parts.append(source_cache[name][2])
            pages = torch.unique(torch.cat(parts))
            if pages.numel() == 0:
                no_write.add(cache.name)
                continue
            if bool((pages >= cache.tensor.shape[0]).any()):
                raise RuntimeError(f"target page outside cache tensor: {cache.name}")
            pages_by_cache[cache.name] = pages
        snapshot = self.assets.snapshot_pages(
            pages_by_cache, strict=True, no_write_caches=no_write
        )
        coverage = snapshot.coverage()
        row = {
            "cycle": cycle, "rank": self.rank,
            "source_layers": len(source_cache),
            "cache_views": len(self.assets.caches),
            "cache_views_snapshotted": len(coverage["captured_names"]),
            "cache_views_certified_no_write": len(coverage["certified_no_write"]),
            "snapshot_entries": coverage["captured_rows"],
            "skipped": coverage["skipped"],
            "page_count_min": min(int(x.numel()) for x in pages_by_cache.values()),
            "page_count_max": max(int(x.numel()) for x in pages_by_cache.values()),
            "page_count_sum": sum(int(x.numel()) for x in pages_by_cache.values()),
            "sources": [
                {"layer": name, "gid": gid, "kind": kind, "physical_pages": int(pages.numel())}
                for name, (gid, kind, pages) in sorted(source_cache.items())
            ],
            "scope": "pre-target candidate pages; no KV value or post-target restore",
        }
        if (row["cache_views_snapshotted"] +
                row["cache_views_certified_no_write"] != len(self.assets.caches)):
            raise RuntimeError("target page snapshot missed a cache view")
        self.rows.append(row)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / f"rank{self.rank}.json").write_text(
            json.dumps({"rank": self.rank, "rows": self.rows}, indent=2) + "\n"
        )

