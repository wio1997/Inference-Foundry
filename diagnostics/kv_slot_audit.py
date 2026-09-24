"""Read-only, bounded KV slot ownership audit for the fixed Extreme cohort."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch


class KVSlotAudit:
    def __init__(self, bindings, output_dir: str, rank: int, limit: int = 8):
        if limit < 2:
            raise ValueError("KV slot audit needs at least two cycles")
        self.bindings = tuple(bindings)
        if not self.bindings:
            raise ValueError("KV slot audit requires group bindings")
        self.output_dir = Path(output_dir)
        self.rank = rank
        self.limit = limit
        self.rows = []
        self.stream = os.getenv("EXTREME_KV_SLOT_AUDIT_STREAM") == "1"
        self._slot_owner: dict[int, dict[int, int]] = {}

    @torch.inference_mode()
    def observe(self, cycle: int, positions: torch.Tensor) -> None:
        if cycle >= self.limit:
            return
        if cycle != len(self.rows):
            raise RuntimeError("KV slot audit cycle discontinuity")
        if positions.numel() != 96:
            raise RuntimeError("KV slot audit requires 12x8 target positions")
        pos = positions.view(12, 8).to(torch.int64)
        request = torch.arange(12, device=pos.device).unsqueeze(1)
        groups = []
        for gid, (table, mapping, size) in enumerate(self.bindings):
            if size <= 0 or table.shape[0] < 12 or mapping.numel() < 96:
                raise RuntimeError(f"KV group {gid} binding shape invalid")
            logical = torch.div(pos, size, rounding_mode="floor")
            supported = bool((logical >= 0).all() and (logical < table.shape[1]).all())
            row = {"group": gid, "block_size": size,
                   "table_width": table.shape[1], "supported": supported,
                   "logical_min": int(logical.min().item()),
                   "logical_max": int(logical.max().item())}
            if supported:
                physical = table[request, logical].to(torch.int64)
                expected = physical * size + pos.remainder(size)
                actual = mapping[:96].view(12, 8).to(torch.int64)
                row.update(mismatched=int((expected != actual).sum().item()),
                           negative_blocks=int((physical < 0).sum().item()),
                           zero_blocks=int((physical == 0).sum().item()),
                           first_expected=expected[:, 0].cpu().tolist(),
                           first_actual=actual[:, 0].cpu().tolist())
            groups.append(row)
        self.rows.append({"cycle": cycle, "groups": groups})
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.stream:
            self._append("pre_target", self.rows[-1])
        else:
            self._flush()

    @torch.inference_mode()
    def observe_dspark_context(self, cycle: int, positions: torch.Tensor, proposer) -> None:
        """Check the actual DSpark context-scatter slot input after proposal."""
        if cycle >= self.limit:
            return
        if cycle >= len(self.rows) or self.rows[cycle]["cycle"] != cycle:
            raise RuntimeError("DSpark context audit has no matching target cycle")
        pos = positions.view(12, 8).to(torch.int64)
        req = torch.arange(12, device=pos.device).unsqueeze(1)
        groups = []
        for group in proposer.draft_attn_groups:
            gid = group.kv_cache_group_id
            table = proposer._per_group_block_tables[gid]
            mapping = proposer._per_group_slot_mappings[gid]
            context = proposer._per_group_context_slot_mapping_buffers[gid]
            size = proposer._per_group_kernel_block_sizes[gid]
            logical = torch.div(pos, size, rounding_mode="floor")
            if not bool((logical >= 0).all() and (logical < table.shape[1]).all()):
                raise RuntimeError(f"DSpark group {gid} context positions outside table")
            physical = table[req, logical].to(torch.int64)
            if not bool((physical > 0).all()):
                raise RuntimeError(f"DSpark group {gid} context uses unowned physical block")
            expected = (physical * size + pos.remainder(size)).flatten()
            actual = context[:96].to(torch.int64)
            source = mapping[:96].to(torch.int64)
            actual_cpu = actual.cpu().tolist()
            owner = self._slot_owner.setdefault(gid, {})
            owner_conflicts = 0
            for index, address in enumerate(actual_cpu):
                prior = owner.setdefault(address, index // 8)
                if prior != index // 8:
                    owner_conflicts += 1
            groups.append({"gid": gid, "context_vs_expected": int((actual != expected).sum().item()),
                           "context_vs_source": int((actual != source).sum().item()),
                           "source_vs_expected": int((source != expected).sum().item()),
                           "context_negative": int((actual < 0).sum().item()),
                           "context_zero": int((actual == 0).sum().item()),
                           "same_cycle_duplicate_slots": 96 - int(torch.unique(actual).numel()),
                           "cross_request_owner_conflicts": owner_conflicts,
                           "first_context": actual.view(12, 8)[:, 0].cpu().tolist(),
                           "first_expected": expected.view(12, 8)[:, 0].cpu().tolist()})
        self.rows[cycle]["dspark_context"] = groups
        if self.stream:
            self._append("post_proposer", self.rows[cycle])
        else:
            self._flush()

    def _append(self, phase: str, row: dict) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / f"rank{self.rank}.jsonl").open("a") as stream:
            stream.write(json.dumps({"rank": self.rank, "phase": phase, **row}, separators=(",", ":")) + "\n")

    def _flush(self) -> None:
        payload = {"rank": self.rank, "cycles_recorded": len(self.rows),
                   "limit": self.limit, "rows": self.rows}
        (self.output_dir / f"rank{self.rank}.json").write_text(
            json.dumps(payload, separators=(",", ":")) + "\n")
