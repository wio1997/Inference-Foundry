"""Opt-in Decode prefix match records at the fixed SGLang KV commit boundary."""

from functools import wraps
from hashlib import sha256
from pathlib import Path


DECODE_SOURCE_SHA256 = '39cf972b6bcc04cc8c150c650d845daaa68ccb511a69a71c5b0b8bdf7322a70d'
PREFIX_SOURCE_SHA256 = 'e0d36d56efd0817d6a7bbe97e35cd60742f59b17af6fcc9b54fecfc8cea60367'
SPAN_SOURCE_SHA256 = 'aef800c2f362094103ff57f04231267c27d2f3d6ae6be00af348d3a1d6651340'


def prefix_match_values(decode_request):
    match = decode_request.prefix_match
    if match is None:
        return None
    device = match.l1_prefix_len
    host = match.l2_host_hit_length
    storage = match.l3_storage_hit_length
    declared = match.decode_prefix_len
    if min(device, host, storage) < 0 or device + host + storage != declared:
        return None
    return device, host, storage, declared


def annotate(original, enabled, record, is_fake_transfer, abort_type):
    @wraps(original)
    def scoped(self, decode_request):
        if not enabled() or self.tp_rank != 0 or is_fake_transfer(decode_request.req):
            return original(self, decode_request)
        match = prefix_match_values(decode_request)
        req = decode_request.req
        room = req.bootstrap_room
        prompt_tokens = len(req.origin_input_ids)
        rebootstrap = int(decode_request.is_rebootstrap)
        result = original(self, decode_request)
        if isinstance(req.finished_reason, abort_type):
            return result
        opaque_room = sha256(str(room).encode()).hexdigest()[:16] if room else 'unavailable'
        name = f'ds41.profile.decode_prefix_commit room={opaque_room} prompt={prompt_tokens} rebootstrap={rebootstrap}'
        if match is None:
            name += ' match=unavailable'
        else:
            device, host, storage, declared = match
            name += f' match=known l1={device} l2={host} l3={storage} declared={declared}'
        with record(name):
            pass
        return result

    scoped._ds41_decode_prefix_trace_marker = True
    return scoped


def install(*, role):
    """Call during startup of an explicitly gated Decode profiling process."""
    assert role == 'decode'
    import torch
    from sglang.srt.disaggregation import decode, decode_hicache_mixin
    from sglang.srt.model_executor import step_span_utils

    for module, pin in ((decode, DECODE_SOURCE_SHA256), (decode_hicache_mixin, PREFIX_SOURCE_SHA256),
                        (step_span_utils, SPAN_SOURCE_SHA256)):
        assert sha256(Path(module.__file__).read_bytes()).hexdigest() == pin
    queue = decode.DecodeTransferQueue
    original = queue._commit_transfer_to_req
    if not getattr(original, '_ds41_decode_prefix_trace_marker', False):
        queue._commit_transfer_to_req = annotate(original, step_span_utils.detailed_annotations_enabled,
                                                 torch.profiler.record_function,
                                                 decode._is_fake_transfer, decode.FINISH_ABORT)
