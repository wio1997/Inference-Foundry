"""Bounded nonblocking status probes of prior-cycle current-stream events.

True means the recorded stream reached the marker before this Host query ended.
It does not certify other streams, native resource availability, or savings.
"""
from __future__ import annotations

import time
import torch

from .ready_edge_observer import ReadyEdgeRecorder


class ReadyQueryRecorder(ReadyEdgeRecorder):
    POINTS = ("cycle_begin", "prepare_target", "target_before")
    SOURCES = ("proposer_after", "draft_commit_after")

    def __init__(self, cycles, *, current_stream, copy_stream=None):
        super().__init__(cycles, current_stream=current_stream, copy_stream=copy_stream)
        self.query_rows = []
        self._query_seen = set()

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
        if mark is None or mark["generation"] != 1 or mark["stream"] != "current":
            raise RuntimeError("typed descriptor lacks current-stream issue marker")
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError("typed descriptor is not a tensor")
        actual_stream = torch.npu.current_stream(tensor.device)
        stream_id = self._stream_identity(actual_stream)
        if stream_id != self.stream_ids["current"]:
            raise RuntimeError("typed descriptor actual stream drift")
        self._seen.add(key)
        self.typed_rows.append({
            "cycle": cycle, "role": role, "after_label": after_label,
            "marker_issue_ordinal": mark["issue_ordinal"],
            "host_observed_ns": time.monotonic_ns(),
            "stream": "current", "stream_id": list(stream_id),
            "storage_ptr": int(tensor.untyped_storage().data_ptr()),
            "data_ptr": int(tensor.data_ptr()),
            "shape": list(tensor.shape), "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype), "device": str(tensor.device),
            "tensor_version": None,
        })

    def _check_lineage(self) -> None:
        by = {(row["cycle"], row["role"]): row for row in self.typed_rows}
        for cycle in self.cycles:
            commit = by[cycle, "after_commit.draft_tokens"]
            state = by[cycle, "after_state.last_sampled_tokens"]
            if commit["storage_ptr"] == 0 or state["storage_ptr"] == 0:
                raise RuntimeError("null state storage")
            prep = by[cycle, "after_prepare.target_input_ids"]
            read = by[cycle, "target_forward.target_input_ids"]
            if (prep["storage_ptr"], prep["data_ptr"]) != (
                    read["storage_ptr"], read["data_ptr"]):
                raise RuntimeError("target input address changed before forward")
            if prep["marker_issue_ordinal"] >= read["marker_issue_ordinal"]:
                raise RuntimeError("target input marker order reversed")
            if commit["shape"] != [12, 7] or prep["shape"] != [96]:
                raise RuntimeError("fixed c12 tensor shape drift")
        for earlier, later in zip(self.cycles, self.cycles[1:]):
            prev = by[earlier, "after_commit.draft_tokens"]
            curr = by[later, "after_commit.draft_tokens"]
            if prev["storage_ptr"] != curr["storage_ptr"]:
                raise RuntimeError("draft state storage moved across cycles")
            if prev["marker_issue_ordinal"] >= curr["marker_issue_ordinal"]:
                raise RuntimeError("draft commit marker order reversed")

    def poll_prior(self, cycle: int, point: str) -> None:
        prior = cycle - 1
        if cycle not in self.cycles or prior not in self.cycles:
            return
        if point not in self.POINTS:
            raise RuntimeError("unknown readiness query point")
        marker = self.rows.get((cycle, point))
        if marker is None or marker["generation"] != 1:
            raise RuntimeError("readiness query lacks current Host marker")
        for source in self.SOURCES:
            key = (cycle, point, source)
            if key in self._query_seen:
                raise RuntimeError("duplicate readiness query")
            prior_event = self.rows[(prior, source)]
            if prior_event["generation"] != 1 or prior_event["stream"] != "current":
                raise RuntimeError("prior event generation/stream drift")
            before = time.monotonic_ns()
            completed = prior_event["event"].query()
            after = time.monotonic_ns()
            if type(completed) is not bool or after < before:
                raise RuntimeError("invalid readiness query result")
            self.query_rows.append({
                "cycle": cycle, "point": point, "source_cycle": prior,
                "source": source, "query_begin_ns": before,
                "query_end_ns": after, "completed": completed,
            })
            self._query_seen.add(key)

    def export_after_existing_drain(self, identity: dict) -> dict:
        packet = super().export_after_existing_drain(identity)
        expected = (len(self.cycles) - 1) * len(self.POINTS) * len(self.SOURCES)
        if len(self.query_rows) != expected:
            raise RuntimeError("incomplete readiness queries")
        by = {(row["cycle"], row["point"], row["source"]): row
              for row in self.query_rows}
        for cycle in self.cycles[1:]:
            for source in self.SOURCES:
                values = [by[cycle, point, source]["completed"] for point in self.POINTS]
                if any(a and not b for a, b in zip(values, values[1:])):
                    raise RuntimeError("prior event completion regressed")
        packet["query_rows"] = self.query_rows
        packet["typed_scope"] = (
            "Storage/data pointers and explicit Event issue ordinals only. "
            "Inference tensors do not expose Python version counters; "
            "same pointer plus marker order does not prove tensor content."
        )
        packet["query_scope"] = (
            "Nonblocking query of prior-cycle current-stream Event at three Host "
            "issue points. True brackets readiness before query end only on this "
            "stream; false gives no completion time. No cross-stream or native "
            "resource readiness is certified. Query overhead perturbs ON run."
        )
        return packet
