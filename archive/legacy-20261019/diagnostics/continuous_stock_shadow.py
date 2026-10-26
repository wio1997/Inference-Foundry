"""Read-only continuous Stock state/ownership observer for the fixed c12 ABI.

The Stock model remains the sole executor. Its accepted tokens and drafts drive
an independent product state machine shadow; this module never writes Stock KV.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch

from runtime.fixed_decode import FixedDecodeConfig
from runtime.greedy_accept import greedy_accept
from runtime.oracle_shadow import FixedDecodeOracleShadow


class ContinuousStockShadow:
    def __init__(self, output_dir: str, *, limit: int = 256) -> None:
        if limit < 29:
            raise ValueError("long Stock shadow must extend the proven 28 cycles")
        self.output_dir = Path(output_dir)
        self.limit = limit
        self.shadow = FixedDecodeOracleShadow(FixedDecodeConfig())
        self.request_ids: tuple[str, ...] | None = None
        self.rank: int | None = None
        self.rows: list[dict[str, object]] = []
        self.pending = False
        self.done = False
        self._draft_tokens: torch.Tensor | None = None
        self.first_acceptance_divergence: dict[str, object] | None = None

    @torch.inference_mode()
    def before_target(
        self,
        *,
        rank: int,
        request_ids: Sequence[str],
        block_table: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        slot_mapping: torch.Tensor,
        group_block_tables: Sequence[torch.Tensor],
        group_slot_mappings: Sequence[torch.Tensor],
        group_block_sizes: Sequence[int],
        group_spec_types: Sequence[str],
    ) -> None:
        if self.done:
            return
        if self.pending:
            raise RuntimeError("Stock shadow missed a cycle output")
        ids = tuple(request_ids)
        if len(ids) != 12 or len(set(ids)) != 12:
            raise RuntimeError("Stock shadow requires 12 distinct fixed slots")
        if self.request_ids is None:
            self.request_ids = ids
            self.rank = rank
        elif ids != self.request_ids or rank != self.rank:
            raise RuntimeError("Stock shadow request-slot order changed")
        if not (len(group_block_tables) == len(group_slot_mappings)
                == len(group_block_sizes) == len(group_spec_types)):
            raise RuntimeError("Stock shadow group bindings are incomplete")
        if not group_block_tables:
            raise RuntimeError("Stock shadow has no KV groups")

        comparison = self.shadow.observe_target_inputs(
            block_table=block_table,
            input_ids=input_ids,
            positions=positions,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            slot_mapping=slot_mapping,
        )
        pos = positions[:96].view(12, 8).to(torch.int64)
        req = torch.arange(12, device=pos.device, dtype=torch.int64).unsqueeze(1)
        groups = []
        for gid, (table, mapping, size) in enumerate(zip(
            group_block_tables, group_slot_mappings, group_block_sizes
        )):
            if size <= 0:
                raise RuntimeError(f"Stock shadow invalid block size in group {gid}")
            logical = torch.div(pos, size, rounding_mode="floor")
            in_range = bool((logical >= 0).all() and
                            (logical < table.shape[1]).all())
            scalar_mapping = mapping[:96].numel() == 96
            group = {
                "group": gid,
                "spec_type": group_spec_types[gid],
                "block_size": int(size),
                "table_shape": list(table.shape),
                "mapping_shape": list(mapping.shape),
                "logical_min": int(logical.min().item()),
                "logical_max": int(logical.max().item()),
                "mapping_supported": in_range and scalar_mapping,
                "mapping_equal": None,
                "negative_physical": None,
                "zero_physical": None,
            }
            if in_range and scalar_mapping:
                physical = table[req, logical].to(torch.int64)
                expected = physical * size + pos.remainder(size)
                actual = mapping[:96].view(12, 8).to(torch.int64)
                group["mapping_equal"] = bool(torch.equal(expected, actual))
                group["negative_physical"] = int((physical < 0).sum().item())
                group["zero_physical"] = int((physical == 0).sum().item())
            groups.append(group)
        row = {
            "cycle": len(self.rows),
            "target_input_ids": input_ids[:96].view(12, 8).tolist(),
            "target_positions": pos.tolist(),
            "target_abi_exact": None if comparison is None else comparison.exact,
            "input_ids_equal": None if comparison is None else comparison.input_ids_equal,
            "positions_equal": None if comparison is None else comparison.positions_equal,
            "query_start_equal": None if comparison is None else comparison.query_start_equal,
            "seq_lens_equal": None if comparison is None else comparison.seq_lens_equal,
            "slot_mapping_equal": None if comparison is None else comparison.slot_mapping_equal,
            "groups": groups,
        }
        self.rows.append(row)
        if (comparison is not None and not comparison.exact) or any(
            (g["mapping_supported"] and (not g["mapping_equal"] or g["negative_physical"]))
            for g in groups
        ):
            self._write("FAIL", "first_integer_or_ownership_divergence")
            raise RuntimeError("Stock continuous shadow first divergence")
        self._draft_tokens = input_ids[:96].view(12, 8)[:, 1:].clone()
        self.pending = True

    @torch.inference_mode()
    def after_cycle(
        self, sampled_token_ids: torch.Tensor, next_draft_tokens: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        if self.done or not self.pending:
            return
        if tuple(sampled_token_ids.shape[:2]) != (12, 8):
            raise RuntimeError("Stock shadow accepted-token shape changed")
        if tuple(next_draft_tokens.shape[:2]) != (12, 7):
            raise RuntimeError("Stock shadow draft-token shape changed")
        if self._draft_tokens is None or logits.shape[0] < 96:
            raise RuntimeError("Stock shadow acceptance inputs missing")
        predicted = logits[:96].argmax(dim=-1).view(12, 8)
        expected, expected_counts = greedy_accept(
            self._draft_tokens, predicted[:, :7], predicted[:, 7]
        )
        actual = sampled_token_ids[:12, :8]
        actual_counts = actual.ne(-1).sum(dim=1).to(torch.int32)
        mismatch = expected.to(torch.int64) != actual.to(torch.int64)
        row = self.rows[-1]
        row["target_argmax"] = predicted.tolist()
        row["sampled"] = actual.tolist()
        row["next_draft"] = next_draft_tokens[:12, :7].tolist()
        row["acceptance_equal"] = not bool(mismatch.any())
        row["acceptance_mismatch_count"] = int(mismatch.sum().item())
        row["greedy_counts"] = expected_counts.tolist()
        row["stock_counts"] = actual_counts.tolist()
        if bool(mismatch.any()) and self.first_acceptance_divergence is None:
            slot, position = mismatch.nonzero()[0].tolist()
            self.first_acceptance_divergence = {
                "cycle": row["cycle"], "slot": slot, "position": position,
                "greedy_token": int(expected[slot, position].item()),
                "stock_token": int(actual[slot, position].item()),
                "greedy_counts": expected_counts.tolist(),
                "stock_counts": actual_counts.tolist(),
            }
        self.shadow.observe_cycle_outputs(sampled_token_ids, next_draft_tokens)
        self._draft_tokens = None
        self.pending = False
        if len(self.rows) % 32 == 0 and len(self.rows) < self.limit:
            self._write("CHECKPOINT", "continuous_integer_shadow_in_progress")
        if len(self.rows) == self.limit:
            unsupported = any(
                not group["mapping_supported"]
                for row in self.rows for group in row["groups"]
            )
            self._write(
                "FAIL" if self.first_acceptance_divergence else
                ("PARTIAL" if unsupported else "PASS"),
                "stock_vs_fixed_greedy_acceptance_divergence"
                if self.first_acceptance_divergence else
                ("continuous_integer_shadow_complete_with_unsupported_group_geometry"
                 if unsupported else "continuous_integer_shadow_complete"),
            )
            self.done = True

    def _write(self, status: str, reason: str) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "rank": self.rank,
            "status": status,
            "reason": reason,
            "limit": self.limit,
            "cycles_recorded": len(self.rows),
            "request_ids": self.request_ids,
            "first_acceptance_divergence": self.first_acceptance_divergence,
            "rows": self.rows,
        }
        (self.output_dir / f"rank{self.rank}.json").write_text(
            json.dumps(result, separators=(",", ":")) + "\n"
        )
