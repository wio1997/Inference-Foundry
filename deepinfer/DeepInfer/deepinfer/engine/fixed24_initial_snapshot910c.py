# SPDX-License-Identifier: Apache-2.0
"""Construct a fixed-profile device-state snapshot; not a serving entrypoint.

The caller must hold the model exclusively between inspection and staging and
must execute on the same stream as the producer of the pending tensors. This
module does not start a graph, verify tokens, recover failures or implement ZOS.
"""
from dataclasses import dataclass

SOURCES = (2, 8, 14)


@dataclass(frozen=True)
class HostSnapshot:
    positions: tuple
    remaining: tuple
    golden: tuple
    finished: tuple
    tail: tuple
    pending: tuple  # (source, tuple of source-owned value/gate or None)
    history_width: int


def inspect_model(model, frame, golden, remaining, finished):
    """Validate the exact pre-forward CPU state without mutating the model.

    The snapshot contains references to pending NPU tensors, not copies. They
    must remain live and unchanged until stage_snapshot returns on its stream.
    Only the four-slot fixed full-query, ignore-EOS profile is supported.
    """
    if (frame is not model.dspark_graphs.full or frame.model is not model or
            frame.count != 4 or frame.slots != [0, 1, 2, 3] or
            frame.query_tokens != 6 or frame.tokens != 24 or frame.prepared):
        raise ValueError('expected the unprepared four-slot full target frame')
    if not all(len(x) == 4 for x in (golden, remaining, finished)):
        raise ValueError('four request states required')
    pos = tuple(model.next_positions[:4])
    if len(pos) != 4 or any(type(x) is not int or x < 0 for x in pos):
        raise ValueError('invalid host positions')
    width = int(model.cfg['engram_max_ngram_size']) - 1
    vocab = int(model.cfg['vocab_size'])
    if not 1 <= width <= 7 or not model.histories or len(model.histories) < 4:
        raise ValueError('unsupported Engram history')
    if any(type(x) is not int or x < 0 for x in remaining):
        raise ValueError('invalid remaining output lengths')
    if any(type(x) is not bool for x in finished):
        raise ValueError('finished must contain booleans')
    if any(done != (left == 0) for done, left in zip(finished, remaining)):
        raise ValueError('requires length-finished ignore-EOS requests')
    if any(type(x) is not int or not 0 <= x < vocab for x in golden):
        raise ValueError('invalid golden tokens')
    if any(p + 6 > model.max_tokens for p in pos):
        raise ValueError('full verification exceeds context capacity')
    tail = []
    for i, p in enumerate(pos):
        history = model.histories[i].history
        if len(history) != p:
            raise ValueError('Engram history and model position differ')
        row = list(history[-width:])
        if any(type(x) is not int or not 0 <= x < vocab for x in row):
            raise ValueError('invalid raw history token')
        tail.append(tuple([-1] * (width - len(row)) + row))
    configured = tuple(sorted(layer.layer_id for layer in model.layers
        if layer.layer_id in model.cfg['kv_source_layer_ids'] and layer.ratio == 2))
    if tuple(sorted(frame.even)) != SOURCES or configured != SOURCES:
        raise ValueError('unexpected C2 source layout')
    pending = []
    for source in SOURCES:
        template = frame.even[source]
        if (len(template) != 2 or any(t.ndim != 2 or t.shape[0] != 4 or
                t.shape[1] <= 0 or t.shape[1] % 8 or
                t.device != model.device or not t.is_contiguous()
                for t in template)):
            raise ValueError('unexpected C2 frame shape')
        source_pending = model.pending.get(source, {})
        slots = []
        for slot, p in enumerate(pos):
            pair = source_pending.get(slot)
            if (pair is not None) != bool(p % 2):
                raise ValueError('C2 pending validity does not match position parity')
            if pair is not None:
                if len(pair) != 2:
                    raise ValueError('C2 pending must have value and gate')
                for actual, expected in zip(pair, template):
                    if (tuple(actual.shape) != (1, expected.shape[1]) or
                            actual.device != expected.device or actual.dtype != expected.dtype or
                            not actual.is_contiguous()):
                        raise ValueError('C2 pending tensor is incompatible with the frame')
            slots.append(pair)
        pending.append((source, tuple(slots)))
    return HostSnapshot(pos, tuple(remaining), tuple(golden), tuple(finished),
                        tuple(tail), tuple(pending), width)


def stage_snapshot(host, model, frame):
    """Stage a validated snapshot onto model.device on the caller's stream.

    Invoke immediately after inspect_model while the model is exclusively held.
    The returned buffers do not alias model pending or frame buffers. CPU state
    remains authoritative until a separately validated integration replaces it.
    """
    import torch
    from deepinfer.npu_ops.state910c import StateBuffers
    from deepinfer.npu_ops.c2_pending910c import allocate
    from deepinfer.engine.fixed24_state_candidate910c import PendingC2, Snapshot

    if not isinstance(host, HostSnapshot):
        raise TypeError('validated HostSnapshot required')
    state = StateBuffers.allocate(model.device, host.history_width)
    for dest, value, dtype in ((state.pos, host.positions, torch.int32),
            (state.remaining, host.remaining, torch.int32),
            (state.golden, host.golden, torch.int32),
            (state.finished, host.finished, torch.bool),
            (state.tail, host.tail, torch.int32)):
        dest.copy_(torch.tensor(value, dtype=dtype, device=model.device))
    state.status.zero_()
    result = {}
    for source, slots in host.pending:
        value, gate, valid, status = allocate(*frame.even[source])
        value.zero_(); gate.zero_(); status.zero_()
        valid.copy_(torch.tensor([pair is not None for pair in slots],
                                 dtype=torch.bool, device=model.device))
        for slot, pair in enumerate(slots):
            if pair is not None:
                value[slot:slot+1].copy_(pair[0]); gate[slot:slot+1].copy_(pair[1])
        result[source] = PendingC2(value, gate, valid, status)
    return Snapshot(state, result)
