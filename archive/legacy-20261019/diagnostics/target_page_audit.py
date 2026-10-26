"""Opt-in source-backed target physical page coverage for fixed c12."""

from __future__ import annotations

import json
import os
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
        self.verify_compressor_operator = os.getenv("EXTREME_COMPRESSOR_SLOT_VERIFY") == "1"
        self.self_replay = os.getenv("EXTREME_TARGET_SELF_REPLAY") == "1"
        self.continuous = os.getenv("EXTREME_TARGET_PAGE_AUDIT_CONTINUOUS") == "1"
        if self.continuous and self.self_replay:
            raise ValueError("continuous page audit cannot self-replay every cycle")
        self.pending_snapshot = None
        self.pending_metadata = None
        self.ownership_probe = os.getenv("EXTREME_CP_OWNERSHIP_PROBE") == "1"
        if self.ownership_probe:
            number = getattr(TargetPageAudit, "_ownership_cohort_seq", 0)
            TargetPageAudit._ownership_cohort_seq = number + 1
            self.ownership_cohort_seq = number
            self._ownership_previous = {}
            self._ownership_last_cycle = -1
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
    def _operator_checks(self, source_cache):
        if not self.verify_compressor_operator:
            return []
        from vllm_ascend.device.device_op import DeviceOperator

        checked = {}
        records = []
        for name, (gid, kind, candidate_pages) in sorted(source_cache.items()):
            if kind != "compressed_scatter":
                continue
            req = self.attn_metadata[name].req_metadata
            ratio = self.ratio_by_gid[gid]
            signature = (
                int(req.block_table.data_ptr()),
                int(req.start_pos.data_ptr()),
                int(req.query_start_loc.data_ptr()),
                int(req.block_size),
                int(ratio),
                int(req.num_compressed_tokens),
                int(req.num_reqs_actual),
            )
            if signature in checked:
                checked[signature]["source_layers"] += 1
                continue
            cos = req.full_compress_cos.view(
                req.full_compress_cos.shape[0], req.full_compress_cos.shape[-1]
            )
            sin = req.full_compress_sin.view(
                req.full_compress_sin.shape[0], req.full_compress_sin.shape[-1]
            )
            generated = torch.ops._C_ascend.compressor_metadata(
                cos, sin, req.query_start_loc, req.start_pos, req.block_table,
                req.block_size,
                DeviceOperator.get_dsa_compressor_slot_mapping_format(),
                ratio, req.num_compressed_tokens, req.num_reqs_actual,
            )[2]
            if generated.ndim == 2 and generated.shape[1] == 2:
                physical_pages = generated[:, 0].to(torch.int64)
            elif generated.ndim == 1:
                physical_pages = torch.div(
                    generated.to(torch.int64), int(req.block_size),
                    rounding_mode="floor",
                )
            else:
                raise RuntimeError(f"compressor slot output shape changed: {tuple(generated.shape)}")
            valid = physical_pages[physical_pages >= 0]
            actual_pages = set(torch.unique(valid).cpu().tolist())
            candidate = set(torch.unique(candidate_pages).cpu().tolist())
            missing = sorted(actual_pages - candidate)
            extra = sorted(candidate - actual_pages)
            record = {
                "layer": name,
                "gid": gid,
                "ratio": ratio,
                "source_layers": 1,
                "generated_rows": int(physical_pages.numel()),
                "valid_rows": int(valid.numel()),
                "actual_pages": len(actual_pages),
                "candidate_pages": len(candidate),
                "missing_actual_pages": len(missing),
                "extra_candidate_pages": len(extra),
                "first_missing_pages": missing[:8],
            }
            checked[signature] = record
            records.append(record)
        return records

    @torch.inference_mode()
    def observe_ownership(self, cycle, state):
        """Read-only fixed-c12 request-owner and physical-frontier census."""
        if cycle != self._ownership_last_cycle + 1:
            raise RuntimeError("ownership probe cycle discontinuity")
        self._ownership_last_cycle = cycle
        name = "model.layers.2.self_attn.attn"
        req = self.attn_metadata[name].req_metadata
        cp = req.cp_metadata
        n = int(req.num_reqs_actual)
        if n != 12 or int(self.attn_metadata[name].num_actual_tokens) != 96:
            raise RuntimeError("ownership probe frozen target shape changed")

        def values(tensor, count=None):
            if tensor is None:
                return None
            if count is not None:
                tensor = tensor[:count]
            return tensor.to("cpu", dtype=torch.int64).flatten().tolist()

        qsl = values(req.query_start_loc, n + 1)
        local_qsl = values(cp.local_query_start_loc, n + 1)
        seq = values(req.seq_lens, n)
        local_seq = values(cp.local_seq_lens, n)
        start = values(req.start_pos, n)
        a, b = int(cp.local_start), int(cp.local_end)
        if qsl[0] != 0 or qsl[-1] != 96 or b - a != 12:
            raise RuntimeError("ownership probe CP partition changed")
        local_counts = [max(0, min(b, qsl[i + 1]) - max(a, qsl[i]))
                        for i in range(n)]
        if local_qsl != [sum(local_counts[:i]) for i in range(n + 1)]:
            raise RuntimeError("ownership probe local query prefix disagrees")
        owners = [i for i, count in enumerate(local_counts) if count]
        full_rows = sum(qsl[i + 1] - qsl[i] for i in owners)
        if state is None:
            raise RuntimeError("ownership probe requires FixedDecodeState")
        metadata = {
            "active_mask": values(state.active_mask, n),
            "num_computed_tokens": values(state.num_computed_tokens, n),
            "emitted_token_count": values(state.emitted_token_count, n),
            "state_target_positions": values(state.target_positions, 96),
            "rank": self.rank, "local_constructor_cohort_seq": self.ownership_cohort_seq,
            "cycle": cycle, "global_qsl": qsl, "local_qsl": local_qsl,
            "global_seq_lens": seq, "local_seq_lens": local_seq,
            "start_pos": start, "cp_local_start_end": [a, b],
            "local_query_rows": sum(local_counts), "local_query_lengths": local_counts,
            "owner_request_slots": owners, "owner_full_update_rows": full_rows,
            "nonowner_full_update_rows": 96 - full_rows,
            "target_input_positions": values(req.input_positions, 96),
            "source_count": len(self.cache_aliases),
            "physical_frontiers": [],
        }
        if cycle == 0:
            metadata["layer2_cache_views"] = [
                {"name": cache.name, "aliases": list(self.cache_aliases[cache.name]),
                 "storage_ptr": int(cache.tensor.untyped_storage().data_ptr()),
                 "storage_nbytes": int(cache.tensor.untyped_storage().nbytes()),
                 "storage_offset": int(cache.tensor.storage_offset()),
                 "data_ptr": int(cache.tensor.data_ptr()),
                 "shape": list(cache.tensor.shape), "stride": list(cache.tensor.stride()),
                 "dtype": str(cache.tensor.dtype)}
                for cache in self.assets.caches
                if any(name.startswith("model.layers.2.")
                       for name in self.cache_aliases[cache.name])]
        for layer in ("model.layers.2.self_attn.indexer.k_cache",
                      "model.layers.2.self_attn.attn",
                      "model.layers.2.self_attn.compressor.state_cache",
                      "model.layers.2.self_attn.indexer.compressor.state_cache"):
            source = self.attn_metadata.get(layer)
            if source is None:
                raise RuntimeError(f"ownership source missing {layer}")
            r = source.req_metadata
            ratio = 4 if (layer.endswith(".attn") or layer.endswith("indexer.k_cache")) else 1
            block_size = int(r.block_size)
            table = r.block_table
            row = {"layer": layer, "ratio": ratio, "block_size": block_size,
                   "table_ptr": int(table.data_ptr()), "table_shape": list(table.shape),
                   "consumer_history_read_scope": "unknown_historical_state_read;new_input_pages_only" if ratio == 1 else "conservative_prefix_page_envelope_including_partial"}
            ranges = []
            for i in range(n):
                if ratio == 4:
                    end = max(seq[i], start[i] + qsl[i + 1] - qsl[i])
                    count = (end + ratio * block_size - 1) // (ratio * block_size)
                    first = 0
                else:
                    length = qsl[i + 1] - qsl[i]
                    first = max(0, start[i] // block_size)
                    count = (start[i] + length + block_size - 1) // block_size
                if count > table.shape[1] or first > count:
                    raise RuntimeError(f"ownership table frontier outside storage: {layer}, request {i}")
                ranges.append((first, count))
            fingerprint = (tuple(ranges), tuple(tuple(values(table[i, lo:hi]))
                                                for i, (lo, hi) in enumerate(ranges)))
            previous = self._ownership_previous.get(layer)
            changed = previous != fingerprint
            self._ownership_previous[layer] = fingerprint
            row["changed"] = changed
            row["page_counts"] = [hi - lo for lo, hi in ranges]
            if changed or cycle == 0:
                row["physical_pages_by_request"] = [
                    {"request_slot": i, "logical_interval": [lo, hi],
                     "physical_pages": list(fingerprint[1][i])}
                    for i, (lo, hi) in enumerate(ranges)]
            metadata["physical_frontiers"].append(row)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"rank{self.rank}_cohort{self.ownership_cohort_seq}.jsonl"
        with path.open("a") as handle:
            handle.write(json.dumps(metadata, separators=(",", ":")) + "\n")

    @torch.inference_mode()
    def observe(self, cycle, state=None):
        if self.ownership_probe:
            self.observe_ownership(cycle, state)
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
        if self.self_replay:
            if self.pending_snapshot is not None:
                raise RuntimeError("unconsumed target self-replay snapshot")
            metadata = []
            seen = set()

            def capture(value, path):
                if id(value) in seen:
                    return
                seen.add(id(value))
                if torch.is_tensor(value):
                    metadata.append((path, value, value.clone()))
                elif isinstance(value, dict):
                    for key, child in value.items():
                        capture(child, f"{path}.{key}")
                elif isinstance(value, (list, tuple)):
                    for index, child in enumerate(value):
                        capture(child, f"{path}[{index}]")
                elif (hasattr(value, "__dict__") and
                      value.__class__.__module__.startswith("vllm_ascend")):
                    for key, child in vars(value).items():
                        capture(child, f"{path}.{key}")

            capture(self.attn_metadata, "attention")
            self.pending_snapshot = snapshot
            self.pending_metadata = metadata
        coverage = snapshot.coverage()
        operator_checks = self._operator_checks(source_cache)
        cache_names = set(self.cache_aliases)
        captured_cache_names = cache_names & set(coverage["captured_names"])
        row = {
            "cycle": cycle, "rank": self.rank,
            "source_layers": len(source_cache),
            "cache_views": len(self.assets.caches),
            "cache_views_snapshotted": len(captured_cache_names),
            "cache_views_certified_no_write": len(coverage["certified_no_write"]),
            "snapshot_entries": coverage["captured_rows"],
            "skipped": coverage["skipped"],
            "operator_checks": operator_checks,
            "metadata_tensor_count": len(self.pending_metadata or ()),
            "metadata_bytes": sum(value.numel() * value.element_size()
                                  for _, value, _ in (self.pending_metadata or ())),
            "operator_missing_actual_pages": sum(
                item["missing_actual_pages"] for item in operator_checks
            ),
            "page_count_min": min(int(x.numel()) for x in pages_by_cache.values()),
            "page_count_max": max(int(x.numel()) for x in pages_by_cache.values()),
            "page_count_sum": sum(int(x.numel()) for x in pages_by_cache.values()),
            "sources": [] if self.continuous else [
                {"layer": name, "gid": gid, "kind": kind, "physical_pages": int(pages.numel())}
                for name, (gid, kind, pages) in sorted(source_cache.items())
            ],
            "source_kind_counts": {
                kind: sum(source_kind == kind for _, source_kind, _ in source_cache.values())
                for kind in sorted({source_kind for _, source_kind, _ in source_cache.values()})
            },
            "scope": "pre-target candidate pages; no KV value or post-target restore",
        }
        if (row["cache_views_snapshotted"] +
                row["cache_views_certified_no_write"] != len(self.assets.caches)):
            raise RuntimeError("target page snapshot missed a cache view")
        self.rows.append(row)
        if (not self.continuous or (cycle + 1) % 8 == 0 or
                cycle + 1 == self.limit or row["operator_missing_actual_pages"]):
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / f"rank{self.rank}.json").write_text(
                json.dumps({"rank": self.rank, "rows": self.rows}, indent=2) + "\n"
            )
        if row["operator_missing_actual_pages"]:
            raise RuntimeError(
                f"compressor operator wrote outside candidate pages: {row['operator_missing_actual_pages']}"
            )

    @torch.inference_mode()
    def _execute_cp_fork_parity(self, target, state, acceptance):
        """One-layer eager A0/B/A0/immediate/A0 from one physical pre-state."""
        if os.getenv("EXTREME_RUNTIME_TARGET_GRAPH") == "1":
            raise RuntimeError("CP parity requires eager target; env cannot select a captured graph")
        if os.getenv("EXTREME_CP_FORK_SCOPE") != "one":
            raise RuntimeError("CP parity requires one-layer scope")
        if self.pending_snapshot is None or self.pending_metadata is None:
            raise RuntimeError("CP parity missing pre-target snapshot")
        snapshot, metadata = self.pending_snapshot, self.pending_metadata
        prior_mode = os.getenv("EXTREME_CP_FORK_MODE")
        selected = [row for row in snapshot.rows
                    if any(name.startswith("model.layers.2.")
                           for name in self.cache_aliases.get(row[0].name, ()))]
        if not selected:
            raise RuntimeError("CP parity has no layer2 cache rows")

        def restore():
            snapshot.restore()
            for _, tensor, value in metadata:
                tensor.copy_(value)
            status = snapshot.verify_restored()
            if not status["exact"] or not all(torch.equal(t, v) for _, t, v in metadata):
                raise RuntimeError("CP parity failed to restore pre-state")

        def execute(mode):
            os.environ["EXTREME_CP_FORK_MODE"] = mode
            output = target.execute(state)
            accepted = acceptance.execute(state, output)
            torch.npu.synchronize()
            values = {
                "logits": output.logits.clone(),
                "hidden": output.hidden_states.clone(),
                "aux_hidden": tuple(x.clone() for x in output.aux_hidden_states),
                "counts": accepted.num_sampled.clone(),
                "tokens": accepted.sampled_token_ids.clone(),
                "post_cache": [(row[0].name, row[0].tensor.index_select(0, row[1]).clone())
                               for row in selected],
            }
            return output, accepted, values

        def tensor_comparison(a, b):
            same = torch.equal(a, b)
            af, bf = a.float(), b.float()
            diff = (af - bf).abs()
            finite = diff[torch.isfinite(diff)]
            return {"exact": bool(same), "shape": list(a.shape),
                    "different": int(torch.count_nonzero(a != b).item()),
                    "max_abs": float(finite.max().item()) if finite.numel() else None}

        def compare(left, right):
            result = {key: tensor_comparison(left[key], right[key])
                      for key in ("logits", "hidden", "counts", "tokens")}
            if len(left["aux_hidden"]) != len(right["aux_hidden"]):
                raise RuntimeError("CP parity aux hidden count changed")
            result["aux_hidden"] = [tensor_comparison(a, b)
                                    for a, b in zip(left["aux_hidden"], right["aux_hidden"])]
            if [x[0] for x in left["post_cache"]] != [x[0] for x in right["post_cache"]]:
                raise RuntimeError("CP parity cache inventory changed")
            result["post_cache"] = [dict(name=name, **tensor_comparison(a, b))
                                    for (name, a), (_, b) in zip(left["post_cache"], right["post_cache"])]
            result["argmax_equal"] = int((left["logits"].argmax(-1) ==
                                            right["logits"].argmax(-1)).sum().item())
            result["argmax_total"] = int(left["logits"].shape[0])
            return result

        try:
            modes = ("off", "overlap", "off", "immediate", "off")
            executions = []
            for index, mode in enumerate(modes):
                if index:
                    restore()
                executions.append(execute(mode))
            summaries = [x[2] for x in executions]
            row = self.rows[-1]
            row["cp_fork_parity"] = {
                "modes": list(modes),
                "cache_views": [x[0] for x in summaries[0]["post_cache"]],
                "a0_repeat_0_2": compare(summaries[0], summaries[2]),
                "a0_repeat_0_4": compare(summaries[0], summaries[4]),
                "overlap_vs_a0": compare(summaries[0], summaries[1]),
                "immediate_vs_a0": compare(summaries[0], summaries[3]),
                "overlap_vs_immediate": compare(summaries[1], summaries[3]),
            }
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / f"rank{self.rank}.json").write_text(
                json.dumps({"rank": self.rank, "rows": self.rows}, indent=2) + "\n")
            self.pending_snapshot = None
            self.pending_metadata = None
            return executions[4][0], executions[4][1]
        finally:
            if prior_mode is None:
                os.environ.pop("EXTREME_CP_FORK_MODE", None)
            else:
                os.environ["EXTREME_CP_FORK_MODE"] = prior_mode

    @torch.inference_mode()
    def execute_with_self_replay(self, target, state, acceptance):
        """Replay the sampled live target from its exact pre-target state."""
        if os.getenv("EXTREME_CP_FORK_PARITY") == "1":
            return self._execute_cp_fork_parity(target, state, acceptance)
        if self.pending_snapshot is None or self.pending_metadata is None:
            raise RuntimeError("target self-replay missing pre-target snapshot")
        snapshot = self.pending_snapshot
        metadata = self.pending_metadata
        def execute():
            output = target.execute(state)
            accepted = acceptance.execute(state, output)
            top2 = torch.topk(output.logits, k=2, dim=-1)
            return output, accepted, (
                top2.indices.clone(), top2.values.clone(),
                accepted.num_sampled.clone(), accepted.sampled_token_ids.clone(),
            )

        def restore():
            snapshot.restore()
            for _, tensor, value in metadata:
                tensor.copy_(value)
            pages = snapshot.verify_restored()
            metadata_exact = all(torch.equal(tensor, value)
                                 for _, tensor, value in metadata)
            if not pages["exact"] or not metadata_exact:
                raise RuntimeError("target self-replay pre-state restoration failed")
            return pages, metadata_exact

        first, first_accept, first_values = execute()
        changed = [path for path, tensor, value in metadata
                   if not torch.equal(tensor, value)]
        restored_pages, restored_metadata = restore()
        second, second_accept, second_values = execute()
        second_restore, second_metadata_exact = restore()
        third, third_accept, third_values = execute()

        def compare(left, right):
            ids_l, scores_l, counts_l, tokens_l = left
            ids_r, scores_r, counts_r, tokens_r = right
            mismatch = (ids_l[:, 0] != ids_r[:, 0]).nonzero().flatten().cpu().tolist()
            details = [{
                "flat_position": int(i),
                "request_slot": int(i // 8),
                "draft_position": int(i % 8),
                "left_top2_ids": ids_l[i].cpu().tolist(),
                "right_top2_ids": ids_r[i].cpu().tolist(),
                "left_top2_scores": scores_l[i].float().cpu().tolist(),
                "right_top2_scores": scores_r[i].float().cpu().tolist(),
            } for i in mismatch]
            return {
                "argmax_equal": int((ids_l[:, 0] == ids_r[:, 0]).sum().item()),
                "counts_equal": int((counts_l == counts_r).sum().item()),
                "accepted_equal": int((tokens_l == tokens_r).sum().item()),
                "argmax_mismatches": details,
                "accepted_mismatch_positions":
                    (tokens_l != tokens_r).nonzero().cpu().tolist(),
            }

        row = self.rows[-1]
        row["self_replay"] = {
            "metadata_changed_by_first_target": changed,
            "metadata_restored_exact": restored_metadata,
            "cache_restored": restored_pages,
            "second_metadata_restored_exact": second_metadata_exact,
            "second_cache_restored": second_restore,
            "argmax_total": int(first_values[0].shape[0]),
            "counts_total": int(first_values[2].numel()),
            "accepted_total": int(first_values[3].numel()),
            "ab": compare(first_values, second_values),
            "ac": compare(first_values, third_values),
            "bc": compare(second_values, third_values),
            "counts": [values[2].cpu().tolist() for values in
                       (first_values, second_values, third_values)],
        }
        (self.output_dir / f"rank{self.rank}.json").write_text(
            json.dumps({"rank": self.rank, "rows": self.rows}, indent=2) + "\n"
        )
        self.pending_snapshot = None
        self.pending_metadata = None
        return third, third_accept
