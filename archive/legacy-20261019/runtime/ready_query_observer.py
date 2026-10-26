"""Bounded nonblocking status probes of prior-cycle current-stream events.

True means the recorded stream reached the marker before this Host query ended.
It does not certify other streams, native resource availability, or savings.
"""
from __future__ import annotations

import time

from .ready_edge_observer import ReadyEdgeRecorder


class ReadyQueryRecorder(ReadyEdgeRecorder):
    POINTS = ("cycle_begin", "prepare_target", "target_before")
    SOURCES = ("proposer_after", "draft_commit_after")

    def __init__(self, cycles, *, current_stream, copy_stream=None):
        super().__init__(cycles, current_stream=current_stream, copy_stream=copy_stream)
        self.query_rows = []
        self._query_seen = set()

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
        packet["query_scope"] = (
            "Nonblocking query of prior-cycle current-stream Event at three Host "
            "issue points. True brackets readiness before query end only on this "
            "stream; false gives no completion time. No cross-stream or native "
            "resource readiness is certified. Query overhead perturbs ON run."
        )
        return packet
