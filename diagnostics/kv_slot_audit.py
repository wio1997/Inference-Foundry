"""Read-only, bounded KV slot ownership audit for the fixed Extreme cohort."""
from __future__ import annotations

import json
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
        payload = {"rank": self.rank, "cycles_recorded": len(self.rows),
                   "limit": self.limit, "rows": self.rows}
        (self.output_dir / f"rank{self.rank}.json").write_text(
            json.dumps(payload, separators=(",", ":")) + "\n")
