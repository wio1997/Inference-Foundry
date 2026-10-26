"""Opt-in read-only typed KV registry and Target pre-replay metadata census.

This is an address/lifetime screen, not a native memory trace or a proof that
an observed write is mathematically compulsory.
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


def _tensor_row(path, tensor):
    storage = tensor.untyped_storage()
    element_size = tensor.element_size()
    extent = element_size
    for size, stride in zip(tensor.shape, tensor.stride()):
        if int(size) == 0:
            extent = 0
            break
        extent += (int(size) - 1) * int(stride) * element_size
    start = int(tensor.storage_offset()) * element_size
    end = start + extent
    if end > int(storage.nbytes()):
        raise RuntimeError(f"typed view beyond storage: {path}")
    return {
        "path": path, "dtype": str(tensor.dtype), "element_size": element_size,
        "shape": list(tensor.shape), "stride": list(tensor.stride()),
        "storage_ptr": int(storage.data_ptr()), "storage_nbytes": int(storage.nbytes()),
        "storage_offset": int(tensor.storage_offset()),
        "byte_envelope": [start, end], "data_ptr": int(tensor.data_ptr()),
    }


def _host(tensor, count=None):
    if tensor is None:
        return None
    if count is not None:
        tensor = tensor[:count]
    return tensor.to("cpu", dtype=torch.int64).tolist()


class StorageLivenessCensus:
    self_replay = False
    limit = 0

    def __init__(self, *, static_context, assets, attn_metadata, proposer,
                 group_bindings, output_dir, rank):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rank = int(rank)
        self.attn_metadata = attn_metadata
        self._cohort = 0
        self._last_cycle = -1
        self._last_parked = None
        self._parking_next = -1
        self._first_parking_seen = False
        self._rows = 0
        self._path = self.output_dir / f"rank{self.rank}.jsonl"
        self._path.write_text("")

        draft_names = sorted(getattr(proposer, "_draft_attn_layer_names", ()))
        all_views = []
        for layer_name, layer in static_context.items():
            cache = getattr(layer, "kv_cache", None)
            if cache is None:
                continue
            for leaf_path, tensor in _leaves(cache, f"{layer_name}.kv_cache"):
                all_views.append(_tensor_row(leaf_path, tensor))
        if not all_views:
            raise RuntimeError("no typed static-context KV leaves")
        layer2_ptrs = {
            view["storage_ptr"] for view in all_views
            if view["path"].startswith("model.layers.2.")
        }
        if len(layer2_ptrs) != 3:
            raise RuntimeError(f"expected three layer2 backing stores, got {len(layer2_ptrs)}")
        selected = [view for view in all_views if view["storage_ptr"] in layer2_ptrs]
        asset_views = [_tensor_row(f"asset.{cache.name}", cache.tensor)
                       for cache in assets.caches if int(cache.tensor.untyped_storage().data_ptr()) in layer2_ptrs]
        draft_views = [view for view in all_views
                       if any(view["path"].startswith(name + ".") for name in draft_names)]
        bindings = []
        for gid, (table, mapping, block_size) in enumerate(group_bindings):
            bindings.append({"gid": gid, "block_size": int(block_size),
                             "table_ptr": int(table.data_ptr()),
                             "mapping_ptr": int(mapping.data_ptr()),
                             "table_shape": list(table.shape),
                             "mapping_shape": list(mapping.shape)})
        self._registry = {
            "rank": self.rank, "scope": "full typed static-context leaves; detailed subset is all aliases of three layer2 backing stores",
            "all_typed_leaf_count": len(all_views),
            "draft_names": draft_names,
            "draft_typed_leaf_count": len(draft_views),
            "draft_overlap_selected_storage": [v for v in draft_views if v["storage_ptr"] in layer2_ptrs],
            "layer2_storage_ptrs": sorted(layer2_ptrs),
            "selected_typed_views": sorted(selected, key=lambda v: v["path"]),
            "selected_asset_views": sorted(asset_views, key=lambda v: v["path"]),
            "group_bindings": bindings,
            "limits": "view interval is a conservative envelope; address alias is not a value conflict, RAW proof or compulsory traffic",
        }
        (self.output_dir / f"rank{self.rank}_registry.json").write_text(
            json.dumps(self._registry, indent=2) + "\n")

    @torch.inference_mode()
    def observe(self, cycle, state):
        if cycle == 0 and self._last_cycle >= 0:
            self._cohort += 1
            self._last_cycle = -1
            self._last_parked = None
            self._parking_next = -1
            self._first_parking_seen = False
        if cycle != self._last_cycle + 1:
            raise RuntimeError("storage census cycle discontinuity")
        self._last_cycle = cycle
        active = _host(state.active_mask, 12)
        parked = [i for i, value in enumerate(active) if not value]
        first_parking = bool(parked) and not self._first_parking_seen
        if first_parking:
            self._first_parking_seen = True
            self._parking_next = cycle + 1
        selected = cycle in (0, 1, 63, 64, 127, 128, 255, 256, 297, 298)
        if not (selected or first_parking or cycle == self._parking_next):
            self._last_parked = parked
            return
        row = {"rank": self.rank, "cohort": self._cohort, "cycle": cycle,
               "phase": "pre_target_graph_replay_after_metadata_update",
               "active_mask": active, "parked_slots": parked,
               "first_parking": first_parking,
               "parking_transition": self._last_parked != parked,
               "num_computed_tokens": _host(state.num_computed_tokens, 12),
               "emitted_token_count": _host(state.emitted_token_count, 12),
               "sources": []}
        self._last_parked = parked
        names = sorted(name for name in self.attn_metadata
                       if name.startswith("model.layers.2.self_attn.")
                       and (name.endswith("swa_cache") or name.endswith(".attn")
                            or name.endswith("state_cache") or name.endswith("k_cache")))
        for name in names:
            metadata = self.attn_metadata[name]
            req = metadata.req_metadata
            n = int(req.num_reqs_actual)
            if n != 12:
                raise RuntimeError(f"{name}: request count {n} != 12")
            cp = getattr(req, "cp_metadata", None)
            table = req.block_table
            qsl = _host(req.query_start_loc, n + 1)
            starts = _host(req.start_pos, n)
            seq = _host(req.seq_lens, n)
            if qsl != list(range(0, 97, 8)):
                raise RuntimeError(f"{name}: frozen qsl changed")
            # Capture only each request's valid historical page prefix, not the
            # whole allocated table. Exact native read indices remain unknown.
            ratio = 4 if name.endswith(".attn") or name.endswith("indexer.k_cache") else 1
            block_size = int(req.block_size)
            prefix_pages = []
            for i in range(n):
                logical_tokens = max(seq[i], starts[i] + qsl[i + 1] - qsl[i])
                count = (logical_tokens + ratio * block_size - 1) // (ratio * block_size)
                if count > table.shape[1]:
                    raise RuntimeError(f"{name}: prefix outside table")
                prefix_pages.append(_host(table[i, :count]))
            row["sources"].append({
                "layer": name, "ratio": ratio, "block_size": block_size,
                "table_ptr": int(table.data_ptr()), "table_shape": list(table.shape),
                "qsl": qsl, "start_pos": starts, "seq_lens": seq,
                "prefix_physical_pages_by_request": prefix_pages,
                "prefix_precision": "conservative_envelope_not_native_read_trace",
                "cp_local_start_end": None if cp is None else [int(cp.local_start), int(cp.local_end)],
                "slot_mapping": _host(getattr(req, "slot_mapping", None), int(metadata.num_actual_tokens)),
                "slot_precision": "source_metadata_pre_replay_not_observed_scatter",
            })
        if len(row["sources"]) < 5:
            raise RuntimeError(f"only {len(row['sources'])} layer2 source metadata rows")
        with self._path.open("a") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._rows += 1
