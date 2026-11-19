"""Annotate a committed rank-zero Decode transfer with its full prompt fingerprint."""

from functools import wraps
from hashlib import sha256
import json
from pathlib import Path

from decode_prefix_annotation_g1387 import (
    DECODE_SOURCE_SHA256, PREFIX_SOURCE_SHA256, SPAN_SOURCE_SHA256, prefix_match_values,
)


def prompt_ids_sha256(input_ids):
    # The frozen profile manifest hashes the default JSON serialization of the ID list.
    ids = input_ids if isinstance(input_ids, list) else list(input_ids)
    return sha256(json.dumps(ids).encode("ascii")).hexdigest()


def annotate(original, enabled, record, is_fake_transfer, abort_type):
    @wraps(original)
    def scoped(self, decode_request):
        if not enabled() or self.tp_rank != 0 or is_fake_transfer(decode_request.req):
            return original(self, decode_request)
        req = decode_request.req
        match = prefix_match_values(decode_request)
        room = req.bootstrap_room
        prompt_ids = req.origin_input_ids
        rebootstrap = int(decode_request.is_rebootstrap)
        result = original(self, decode_request)
        if isinstance(req.finished_reason, abort_type):
            return result
        opaque_room = sha256(str(room).encode()).hexdigest()[:16] if room else "unavailable"
        name = f"ds41.profile.decode_prefix_commit room={opaque_room} prompt={len(prompt_ids)} rebootstrap={rebootstrap}"
        if match is None:
            name += " match=unavailable"
        else:
            device, host, storage, declared = match
            name += f" match=known l1={device} l2={host} l3={storage} declared={declared}"
        with record(f"{name} prompt_ids_sha256={prompt_ids_sha256(prompt_ids)}"):
            pass
        return result

    scoped._ds41_decode_prefix_trace_marker = True
    scoped._ds41_decode_prefix_prompt_marker = True
    return scoped


def install(*, role):
    """Call during startup of an explicitly gated Decode profiling process."""
    if role != "decode":
        raise ValueError("Decode role required")
    import torch
    from sglang.srt.disaggregation import decode, decode_hicache_mixin
    from sglang.srt.model_executor import step_span_utils

    for module, digest in ((decode, DECODE_SOURCE_SHA256), (decode_hicache_mixin, PREFIX_SOURCE_SHA256),
                           (step_span_utils, SPAN_SOURCE_SHA256)):
        if sha256(Path(module.__file__).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Unexpected source for {module.__name__}")
    queue = decode.DecodeTransferQueue
    commit = queue._commit_transfer_to_req
    if getattr(commit, "_ds41_decode_prefix_prompt_marker", False):
        return
    if getattr(commit, "_ds41_decode_prefix_trace_marker", False):
        raise ValueError("A different Decode prefix marker is already installed")
    queue._commit_transfer_to_req = annotate(
        commit, step_span_utils.detailed_annotations_enabled, torch.profiler.record_function,
        decode._is_fake_transfer, decode.FINISH_ABORT,
    )
