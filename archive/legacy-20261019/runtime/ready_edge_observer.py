"""Typed, bounded extension of the existing Run637 current-stream Event packet.

This records CPU-visible tensor metadata only. It never reads tensor contents,
synchronizes the device, or treats a Python version as physical readiness.
"""
from __future__ import annotations

import time

import torch

from .bound_observer import BoundEventRecorder


class ReadyEdgeRecorder(BoundEventRecorder):
    MAX_TYPED_ROWS = 3 * 128

    def __init__(self, cycles, *, current_stream, copy_stream=None):
        super().__init__(cycles, current_stream=current_stream, copy_stream=copy_stream)
        self.typed_rows = []
        self._seen = set()

    def observe(self, cycle: int, role: str, tensor: torch.Tensor,
                *, after_label: str) -> None:
        if cycle not in self.cycles:
            return
        if not isinstance(role, str) or not role or len(role) > 80:
            raise RuntimeError("invalid typed role")
        if len(self.typed_rows) >= self.MAX_TYPED_ROWS:
            raise RuntimeError("typed row capacity exceeded")
        key = (cycle, role)
        if key in self._seen:
            raise RuntimeError("duplicate typed role in cycle")
        mark = self.rows.get((cycle, after_label))
        if mark is None or mark["generation"] != 1:
            raise RuntimeError("typed descriptor lacks completed issue marker")
        if mark["stream"] != "current":
            raise RuntimeError("typed descriptor marker is not current stream")
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError("typed descriptor is not a tensor")
        actual_stream = torch.npu.current_stream(tensor.device)
        stream_id = self._stream_identity(actual_stream)
        if stream_id != self.stream_ids["current"]:
            raise RuntimeError("typed descriptor actual stream drift")
        self._seen.add(key)
        self.typed_rows.append({
            "cycle": cycle,
            "role": role,
            "after_label": after_label,
            "marker_issue_ordinal": mark["issue_ordinal"],
            "host_observed_ns": time.monotonic_ns(),
            "stream": "current",
            "stream_id": list(stream_id),
            "storage_ptr": int(tensor.untyped_storage().data_ptr()),
            "data_ptr": int(tensor.data_ptr()),
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "python_version": int(tensor._version),
        })

    def export_after_existing_drain(self, identity: dict) -> dict:
        packet = super().export_after_existing_drain(identity)
        required = (
            "after_state.last_sampled_tokens",
            "after_state.num_computed_tokens",
            "after_proposer.next_draft",
            "after_commit.draft_tokens",
            "after_prepare.target_input_ids",
            "after_prepare.target_positions",
            "after_prepare.target_slot_mapping",
            "target_forward.target_input_ids",
        )
        for cycle in self.cycles:
            roles = {row["role"] for row in self.typed_rows
                     if row["cycle"] == cycle}
            missing = set(required) - roles
            if missing:
                raise RuntimeError(f"missing typed roles for cycle {cycle}: {sorted(missing)}")
        self._check_lineage()
        packet["typed_rows"] = self.typed_rows
        packet["typed_scope"] = (
            "Python storage/version and actual current-stream identity at marked "
            "writer/consumer boundaries. Version is not content or physical "
            "completion. Device Event marker supplies only same-stream ordering."
        )
        return packet

    def _check_lineage(self) -> None:
        by = {(row["cycle"], row["role"]): row for row in self.typed_rows}
        for cycle in self.cycles:
            commit = by[cycle, "after_commit.draft_tokens"]
            state = by[cycle, "after_state.last_sampled_tokens"]
            if commit["storage_ptr"] == 0 or state["storage_ptr"] == 0:
                raise RuntimeError("null state storage")
            prep = by[cycle, "after_prepare.target_input_ids"]
            read = by[cycle, "target_forward.target_input_ids"]
            if (prep["storage_ptr"], prep["data_ptr"], prep["python_version"]) != (
                read["storage_ptr"], read["data_ptr"], read["python_version"]
            ):
                raise RuntimeError("target input changed between prepare and forward")
            if commit["shape"] != [12, 7] or prep["shape"] != [96]:
                raise RuntimeError("fixed c12 tensor shape drift")
        for earlier, later in zip(self.cycles, self.cycles[1:]):
            prev = by[earlier, "after_commit.draft_tokens"]
            curr = by[later, "after_commit.draft_tokens"]
            if prev["storage_ptr"] != curr["storage_ptr"]:
                raise RuntimeError("draft state storage moved across cycles")
            if curr["python_version"] <= prev["python_version"]:
                raise RuntimeError("draft state Python write version did not advance")
