"""Keep backend auxiliary state addressable at original committed positions.

The original Scheduler supplies positions and owns IDs. This stores only the
small C2 projection span needed to undo unaccepted target verification rows;
it introduces no admission, replacement, stopping or acceptance policy.
"""
from dataclasses import dataclass, field
import torch


@dataclass
class StateSpan:
    start: int
    end: int
    valid_before: int
    pending_before: dict
    projected: dict = field(default_factory=dict)


@dataclass
class DeviceStateSpan:
    """Owned target rows until original CPU post-processing catches up."""
    starts: torch.Tensor
    positions: torch.Tensor
    slots: torch.Tensor
    before: dict
    projected: dict


class W8A8StateHistory:
    def __init__(self, model):
        self.model = model
        self.spans = {}
        self.recording = {}
        self.device_spans = {}

    def clear(self):
        self.spans.clear()
        self.recording.clear()
        self.device_spans.clear()

    def clear_slot(self, slot):
        self.spans.pop(slot, None)
        self.recording.pop(slot, None)
        self.device_spans.pop(slot, None)

    def seed_device(self, frame, starts):
        """Select original committed C2 pairs without reading device lengths.

        The preceding target span includes rejected verification rows. The
        current original metadata supplies the committed start on device;
        matching start-1 selects its pair, or the retained pre-span seed.
        Fresh/restored slots keep the existing host-owned seed.
        """
        frame._prepare_sequence()
        for slot, span in self.device_spans.items():
            if slot not in frame.slots:
                continue
            row = frame.slots.index(slot)
            start = starts[row]
            match = (span.slots == slot) & (span.positions == start - 1)
            index = match.to(torch.int32).argmax().reshape(1)
            found = match.any()
            odd = start.remainder(2) != 0
            for source, pair in frame.even.items():
                for target, projected, before in zip(pair, span.projected[source],
                                                       span.before[source], strict=True):
                    selected = projected.index_select(0, index)
                    value = torch.where(found, selected, before[row:row+1])
                    target[row:row+1].copy_(torch.where(odd, value, 0))

    def retain_device(self, frame, starts):
        # A later replay may overwrite every frame buffer. Both the selected
        # projection rows and the pre-span seed must own their storage.
        copy = lambda pairs: {source: tuple(t.clone() for t in pair)
                              for source, pair in pairs.items()}
        return DeviceStateSpan(starts.clone(), frame.positions.clone(),
                               frame.query_requests.clone(), copy(frame.even),
                               copy(frame.even_rows))

    def publish_device(self, span, slots):
        for slot in slots:
            self.device_spans[slot] = span

    def validate_position(self, slot, position):
        model = self.model
        if (type(slot) is not int or not 0 <= slot < model.requests
                or type(position) is not int or not 0 <= position <= model.max_tokens):
            raise ValueError('original state position is outside backend capacity')
        current = model.next_positions[slot]
        if position and model.histories and len(model.histories[slot].history) != current:
            raise ValueError('Engram history differs from backend state position')
        if position == 0 or position == current:
            if position % 2:
                for layer in model.layers:
                    if getattr(layer, 'ratio', 0) == 2 and hasattr(layer, 'compressor_wkv_weight'):
                        if slot not in model.pending.get(layer.layer_id, {}):
                            raise ValueError('committed odd position lacks a C2 pending pair')
            return
        span = self.spans.get(slot)
        if span is None or not span.start <= position <= span.end or position > current:
            raise ValueError('original committed position has no retained backend state')
        if position % 2 and position > span.start:
            layers = {layer.layer_id for layer in model.layers
                      if getattr(layer, 'ratio', 0) == 2 and hasattr(layer, 'compressor_wkv_weight')}
            if not layers <= span.projected.keys():
                raise ValueError('incomplete C2 projection span at original committed position')

    def prepare(self, positions):
        # Validate every state before modifying any auxiliary owner.
        for slot, position in positions.items():
            self.validate_position(slot, position)
        model = self.model
        for slot, position in positions.items():
            current = model.next_positions[slot]
            if position == 0:
                for values in model.pending.values():
                    values.pop(slot, None)
                if model.histories:
                    model.histories[slot].history = []
                model.yoco_swa_valid_start[slot] = 0
                model.restored_prefix_slots.discard(slot)
                self.clear_slot(slot)
            elif position != current:
                span = self.spans[slot]
                for values in model.pending.values():
                    values.pop(slot, None)
                if position == span.start:
                    pending = span.pending_before
                elif position % 2:
                    index = position - 1 - span.start
                    pending = {layer: tuple(t[index:index + 1] for t in pair)
                               for layer, pair in span.projected.items()}
                else:
                    pending = {}
                for layer, pair in pending.items():
                    model.pending.setdefault(layer, {})[slot] = pair
                if model.histories:
                    del model.histories[slot].history[position:]
                model.yoco_swa_valid_start[slot] = max(span.valid_before, position - 128, 0)
            model.next_positions[slot] = position

    def begin(self, slots, lengths):
        if self.recording:
            raise RuntimeError('the previous auxiliary state recording is unfinished')
        limit = 1 + (0 if self.model.dspark is None else self.model.dspark.nextn)
        if (len(slots) != len(lengths) or len(set(slots)) != len(slots)
                or any(type(slot) is not int or not 0 <= slot < self.model.requests for slot in slots)
                or any(type(length) is not int or not 1 <= length <= limit for length in lengths)
                or any(self.model.next_positions[slot] + length > self.model.max_tokens
                       for slot, length in zip(slots, lengths))):
            raise ValueError('verification state recording exceeds original query capacity')
        for slot in slots:
            self.validate_position(slot, self.model.next_positions[slot])
        for slot, length in zip(slots, lengths, strict=True):
            start = self.model.next_positions[slot]
            self.recording[slot] = StateSpan(start, start + length,
                self.model.yoco_swa_valid_start[slot],
                {layer: values[slot] for layer, values in self.model.pending.items() if slot in values})

    def record(self, layer, slots, lengths, projected, scores):
        if not self.recording:
            return
        if (projected.shape[0] != sum(lengths) or scores.shape[0] != sum(lengths)
                or projected.dtype != torch.float32 or scores.dtype != torch.float32):
            raise ValueError('C2 state history requires complete FP32 projections')
        offset = 0
        for slot, length in zip(slots, lengths, strict=True):
            span = self.recording[slot]
            if length != span.end - span.start:
                raise ValueError('C2 projection span differs from original query rows')
            span.projected[layer] = (projected[offset:offset + length], scores[offset:offset + length])
            offset += length

    def finish(self):
        layers = {layer.layer_id for layer in self.model.layers
                  if getattr(layer, 'ratio', 0) == 2 and hasattr(layer, 'compressor_wkv_weight')}
        if any(not layers <= span.projected.keys() for span in self.recording.values()):
            raise ValueError('target state recording lacks C2 source projections')
        self.spans.update(self.recording)
        self.recording = {}

    def clone_slot(self, source, destination):
        if self.recording:
            raise RuntimeError('state clone requires a completed target verification span')
        # CacheEngine has copied the exact host-owned C2 seed and CPU span.
        # The old device span still contains source physical-slot indices.
        self.device_spans.pop(destination, None)
        span = self.spans.get(source)
        if span is None:
            self.clear_slot(destination)
            return
        copy = lambda mapping: {layer: tuple(t.clone() for t in pair) for layer, pair in mapping.items()}
        self.spans[destination] = StateSpan(span.start, span.end, span.valid_before,
                                          copy(span.pending_before), copy(span.projected))
