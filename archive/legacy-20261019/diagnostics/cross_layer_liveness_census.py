"""Opt-in compact cross-layer alias/lifetime census for three layer2 backings.

All recorded read domains are source-derived envelopes. An overlap is only a
candidate address hazard; it is never a native read-from or semantic proof.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def _leaves(value, path):
    if torch.is_tensor(value):
        yield path, value
    elif isinstance(value, (tuple, list)):
        for i, child in enumerate(value):
            yield from _leaves(child, f"{path}[{i}]")
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _leaves(child, f"{path}.{key}")


def _view(path, tensor):
    storage = tensor.untyped_storage()
    item = int(tensor.element_size())
    lo = int(tensor.storage_offset()) * item
    size = 0 if tensor.numel() == 0 else item
    if size:
        for n, stride in zip(tensor.shape, tensor.stride()):
            size += (int(n) - 1) * int(stride) * item
    hi = lo + size
    if size and hi > int(storage.nbytes()):
        raise RuntimeError(f"typed view outside storage: {path}")
    return {"path": path, "dtype": str(tensor.dtype), "element_size": item,
            "shape": list(tensor.shape), "stride": list(tensor.stride()),
            "storage_ptr": int(storage.data_ptr()), "storage_nbytes": int(storage.nbytes()),
            "storage_offset": int(tensor.storage_offset()), "byte_envelope": [lo, hi],
            "absolute_byte_envelope": [int(storage.data_ptr()) + lo, int(storage.data_ptr()) + hi]}


def _host(tensor, count=None):
    if tensor is None:
        return None
    return tensor[:count].to("cpu", dtype=torch.int64).tolist() if count is not None else tensor.to("cpu", dtype=torch.int64).tolist()


class CrossLayerLivenessCensus:
    self_replay = False
    limit = 0

    def __init__(self, *, static_context, attn_metadata, target_model,
                 proposer, metadata_sources, output_dir, rank):
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.rank = int(rank)
        self.attn_metadata = attn_metadata
        self.path = self.root / f"rank{self.rank}.jsonl"
        self.path.write_text("")
        self.cohort = 0
        self.last_cycle = -1
        self.first_parking = False
        self.parking_next = -1
        self.previous_candidate = None

        all_views = []
        for name, layer in static_context.items():
            cache = getattr(layer, "kv_cache", None)
            if cache is not None:
                all_views.extend(_view(path, tensor) for path, tensor in _leaves(cache, f"{name}.kv_cache"))
        layer2_ptrs = {v["storage_ptr"] for v in all_views if v["path"].startswith("model.layers.2.")}
        if len(layer2_ptrs) != 3:
            raise RuntimeError(f"expected three layer2 backing stores, got {len(layer2_ptrs)}")
        selected = [v for v in all_views if v["storage_ptr"] in layer2_ptrs]
        draft_names = sorted(getattr(proposer, "_draft_attn_layer_names", ()))
        draft = [v for v in all_views if any(v["path"].startswith(name + ".") for name in draft_names)]
        if len(draft) != 3:
            raise RuntimeError(f"expected three actual Draft typed leaves, got {len(draft)}")
        if any(v["storage_ptr"] in layer2_ptrs for v in draft):
            raise RuntimeError("Draft cache shares a selected layer2 backing")
        grouped = {}
        for v in selected:
            grouped.setdefault(v["path"].split(".kv_cache", 1)[0], []).append(v)
        if len(grouped) != 14:
            raise RuntimeError(f"expected 14 Target alias source names, got {len(grouped)}")
        by_table = {}
        for source in metadata_sources:
            req = attn_metadata[source.layer_name].req_metadata
            ptr = int(req.block_table.data_ptr())
            item = (int(source.ratio), int(source.block_size))
            if ptr in by_table and by_table[ptr] != item:
                raise RuntimeError("one block table has conflicting cache address ratios")
            by_table[ptr] = item
        self.sources = {}
        for name, views in sorted(grouped.items()):
            if name not in attn_metadata:
                raise RuntimeError(f"alias source lacks target metadata: {name}")
            req = attn_metadata[name].req_metadata
            table_ptr = int(req.block_table.data_ptr())
            if table_ptr not in by_table:
                raise RuntimeError(f"alias source table has no source spec: {name}")
            position_ratio, block_size = by_table[table_ptr]
            if block_size != int(req.block_size):
                raise RuntimeError(f"alias block size differs from spec: {name}")
            layer_index = int(name.split(".")[2])
            attn = target_model.get_submodule(f"model.layers.{layer_index}.self_attn")
            producer_ratio = int(attn.compress_ratio)
            compressor = getattr(attn, "compressor", None)
            producer_coff = None if compressor is None else int(compressor.coff)
            if producer_ratio == 1 and producer_coff is not None:
                raise RuntimeError("SWA layer unexpectedly has Compressor")
            if producer_ratio in (4, 128) and producer_coff != (2 if producer_ratio == 4 else 1):
                raise RuntimeError(f"Compressor ratio/coff mismatch: {name}")
            if name.endswith("swa_cache") or name.endswith("state_cache"):
                expected_position_ratio = 1
            elif name.endswith(".attn") or name.endswith("indexer.k_cache"):
                expected_position_ratio = producer_ratio
            else:
                raise RuntimeError(f"unclassified Target alias source {name}")
            if position_ratio != expected_position_ratio:
                raise RuntimeError(f"position/producer ratio relationship changed: {name}")
            window = int(attn.window_size)
            if window != 128:
                raise RuntimeError(f"SWA window changed: {window}")
            self.sources[name] = {
                "name": name, "layer_index": layer_index,
                "storage_ptr": views[0]["storage_ptr"], "typed_views": views,
                "table_ptr": table_ptr, "cache_position_ratio": position_ratio,
                "producer_compress_ratio": producer_ratio, "producer_coff": producer_coff,
                "block_size": block_size, "swa_window": window,
            }
            if len({v["storage_ptr"] for v in views}) != 1:
                raise RuntimeError(f"one alias source spans multiple backings: {name}")
        registry = {
            "rank": self.rank, "all_typed_leaf_count": len(all_views),
            "selected_target_alias_count": len(self.sources),
            "selected_target_views": selected, "draft_names": draft_names,
            "draft_typed_views": draft, "sources": list(self.sources.values()),
            "limits": "typed view and page domains reveal address reuse, not native reads, values, mathematical necessity or critical-path savings",
        }
        (self.root / f"rank{self.rank}_registry.json").write_text(json.dumps(registry, indent=2) + "\n")

    @staticmethod
    def _write_slots(req, ratio, block_size, starts):
        mapping = getattr(req, "slot_mapping", None)
        if mapping is not None:
            slots = _host(mapping, 96)
            if len(slots) != 96 or any(len(x) != 2 for x in slots):
                raise RuntimeError("source slot mapping is not 96x2")
            return [slots[i * 8:(i + 1) * 8] for i in range(12)], "metadata_slot_mapping"
        table = req.block_table
        result = []
        for i, start in enumerate(starts):
            logical_slots = sorted({pos // ratio for pos in range(start, start + 8)})
            indices = sorted({pos // block_size for pos in logical_slots})
            physical = _host(table[i, torch.tensor(indices, device=table.device)])
            page_map = dict(zip(indices, physical))
            result.append([[int(page_map[pos // block_size]), int(pos % block_size)]
                           for pos in logical_slots])
        return result, "conservative_source_derived_compressed_slots"

    @torch.inference_mode()
    def observe(self, cycle, state):
        if cycle == 0 and self.last_cycle >= 0:
            self.cohort += 1
            self.last_cycle = -1
            self.first_parking = False
            self.parking_next = -1
            self.previous_candidate = None
        if cycle != self.last_cycle + 1:
            raise RuntimeError("cross-layer census cycle discontinuity")
        self.last_cycle = cycle
        active = _host(state.active_mask, 12)
        parked = [i for i, value in enumerate(active) if not value]
        first_parking = bool(parked) and not self.first_parking
        if first_parking:
            self.first_parking = True
            self.parking_next = cycle + 1
        selected = cycle in (0, 1, 63, 64, 127, 128, 255, 256, 297, 298)
        if not (selected or first_parking or cycle == self.parking_next):
            self.previous_candidate = None
            return
        owner = {i for i in range(12)
                 if max(0, min((self.rank + 1) * 12, (i + 1) * 8)
                        - max(self.rank * 12, i * 8))}
        records = []
        candidate = {}
        for name, spec in self.sources.items():
            metadata = self.attn_metadata[name]
            req = metadata.req_metadata
            if int(req.num_reqs_actual) != 12 or int(metadata.num_actual_tokens) != 96:
                raise RuntimeError(f"source geometry changed: {name}")
            starts = _host(req.start_pos, 12)
            seq = _host(req.seq_lens, 12)
            qsl = _host(req.query_start_loc, 13)
            if qsl != list(range(0, 97, 8)):
                raise RuntimeError(f"source qsl changed: {name}")
            slots, precision = self._write_slots(req, spec["cache_position_ratio"], spec["block_size"], starts)
            write_pages = [sorted({int(x[0]) for x in ss}) for ss in slots]
            if spec["layer_index"] == 2:
                pages = candidate.setdefault(spec["storage_ptr"], {})
                for req_index in range(12):
                    if req_index in owner:
                        continue
                    for page in write_pages[req_index]:
                        pages.setdefault(page, []).append(name)
            records.append({
                "name": name, "layer_index": spec["layer_index"],
                "storage_ptr": spec["storage_ptr"], "table_ptr": spec["table_ptr"],
                "cache_position_ratio": spec["cache_position_ratio"],
                "producer_compress_ratio": spec["producer_compress_ratio"],
                "producer_coff": spec["producer_coff"],
                "block_size": spec["block_size"],
                "start_pos": starts, "seq_lens": seq,
                "current_write_slots_by_request": slots,
                "current_write_pages_by_request": write_pages,
                "write_precision": precision,
                "read_overlap": [],
            })
        for record in records:
            spec = self.sources[record["name"]]
            req = self.attn_metadata[record["name"]].req_metadata
            table = req.block_table
            ratio, size = spec["cache_position_ratio"], spec["block_size"]
            current = candidate.get(spec["storage_ptr"], {})
            previous = (self.previous_candidate["pages"].get(spec["storage_ptr"], {})
                        if self.previous_candidate is not None
                        and self.previous_candidate["cycle"] == cycle - 1 else {})
            for req_index, start in enumerate(record["start_pos"]):
                if record["name"].endswith("swa_cache"):
                    first = max(0, start - spec["swa_window"] + 1)
                    last = start + 8
                    logical_indices = sorted({pos // size for pos in range(first, last)})
                    read_precision = "source_swa_128_window_page_envelope"
                else:
                    last = max(record["seq_lens"][req_index], start + 8)
                    logical_indices = list(range((last + ratio * size - 1) // (ratio * size)))
                    read_precision = "conservative_full_prefix_page_envelope"
                if not logical_indices or (not current and not previous):
                    continue
                if logical_indices[-1] >= table.shape[1]:
                    raise RuntimeError(f"read domain outside table: {record['name']}")
                physical = _host(table[req_index, torch.tensor(logical_indices, device=table.device)])
                for edge, candidates in (("same_cycle", current), ("previous_sample_cycle", previous)):
                    hits = []
                    for logical, page in zip(logical_indices, physical):
                        if int(page) in candidates:
                            hits.append({"logical_page": int(logical), "physical_page": int(page),
                                         "candidate_producers": sorted(set(candidates[int(page)]))})
                    if hits:
                        record["read_overlap"].append({"request_slot": req_index, "edge": edge,
                                                       "precision": read_precision, "hits": hits})
            record["same_backing_write_overlap"] = [
                {"request_slot": i, "physical_pages": sorted(set(pages) & set(current))}
                for i, pages in enumerate(record["current_write_pages_by_request"])
                if set(pages) & set(current)
            ]
        row = {"rank": self.rank, "cohort": self.cohort, "cycle": cycle,
               "phase": "pre_target_graph_replay_after_metadata_update",
               "active_mask": active, "parked_slots": parked,
               "first_parking": first_parking, "owner_request_slots": sorted(owner),
               "candidate_scope": "layer2_nonowner_current_write_source_envelopes",
               "page_epoch": "unknown", "sources": records}
        with self.path.open("a") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        self.previous_candidate = {"cycle": cycle, "pages": candidate}
