"""Private protocol ledger for a future segmented Extreme serving gate.

This module does not alter scheduling or run the model. It records the exact
early response bytes that a later, single Scheduler bulk settlement must
produce, so a continuation cannot silently change already-published output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class EarlySegment:
    generation: int
    request_id: str
    client_index: int
    prior_token_ids: tuple[int, ...]
    new_token_ids: tuple[int, ...]
    max_tokens: int


class GhostPublicationLedger:
    """Exactly-once early publication; final Scheduler settlement still owns KV."""

    def __init__(
        self, *, generation: int, request_ids: Sequence[str], max_tokens: int
    ) -> None:
        if generation < 0 or len(request_ids) != 12 or len(set(request_ids)) != 12:
            raise ValueError("one unique twelve-request cohort is required")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.generation = generation
        self.request_ids = frozenset(request_ids)
        self.max_tokens = max_tokens
        self._early: dict[str, EarlySegment] = {}
        self._settled = False
        self._failed = False

    @property
    def early_request_ids(self) -> frozenset[str]:
        return frozenset(self._early)

    def publish(
        self,
        *,
        request_id: str,
        client_index: int,
        prior_token_ids: Sequence[int],
        runtime_token_ids: Sequence[int],
    ) -> EarlySegment:
        if self._settled or self._failed:
            raise RuntimeError("cohort is no longer publishable")
        if request_id not in self.request_ids or request_id in self._early:
            raise ValueError("unknown or duplicate early request")
        prior = tuple(int(x) for x in prior_token_ids)
        runtime = tuple(int(x) for x in runtime_token_ids)
        remaining = self.max_tokens - len(prior)
        if remaining <= 0 or len(runtime) < remaining:
            raise ValueError("early Runtime output does not reach request limit")
        if any(x < 0 for x in prior + runtime[:remaining]):
            raise ValueError("output token IDs must be nonnegative")
        segment = EarlySegment(
            generation=self.generation,
            request_id=request_id,
            client_index=client_index,
            prior_token_ids=prior,
            new_token_ids=runtime[:remaining],
            max_tokens=self.max_tokens,
        )
        self._early[request_id] = segment
        return segment

    def settle_once(
        self, final_new_token_ids: Mapping[str, Sequence[int]]
    ) -> frozenset[str]:
        """Check final Scheduler outputs; caller filters duplicate external rows.

        The caller must invoke Scheduler.update_from_output exactly once before
        calling this method. It must keep all old KV pages until then.
        """
        if self._settled or self._failed:
            raise RuntimeError("cohort already settled or invalid")
        if set(final_new_token_ids) != self.request_ids:
            raise ValueError("final settlement must cover the original twelve")
        for request_id, segment in self._early.items():
            actual = tuple(int(x) for x in final_new_token_ids[request_id])
            if actual != segment.new_token_ids:
                self._failed = True
                raise ValueError(f"final output diverges after early publication: {request_id}")
        self._settled = True
        return self.early_request_ids

    def fail_continuation(self) -> None:
        """Mark the entire experimental cohort invalid; early replies cannot retract."""
        if self._settled:
            raise RuntimeError("already settled")
        self._failed = True
