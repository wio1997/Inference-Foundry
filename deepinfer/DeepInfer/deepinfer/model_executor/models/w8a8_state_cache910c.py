"""Original state-cache access over the resident, layer-major W8A8 buffers.

State rows are materialized only for explicit copies/transfers. Numerical cache
addresses, graphs and allocator ownership stay unchanged. Disk rows contain the
stored SWA layers; full rows also preserve native draft/C2/Engram rollback state.
The byte format belongs to this backend and is fingerprinted, not GPU FP8 data.
"""
import hashlib
import json
import struct

import torch
from deepinfer.multimodal_tokens import ImageTokenOffset


class W8A8StateCache:
    VERSION = 2

    def __init__(self, storage):
        from deepinfer.hugging_face.config import _n_prefill_layer
        from deepinfer.worker.cache_engine import get_inference_swa_block_size

        self.storage = storage
        self.model = model = storage.model
        self.device = storage.state_planes[0].cache.device
        self.dtype = torch.uint8
        self.window = get_inference_swa_block_size(model.cfg['sliding_window'])
        prefill_layers = model.cfg.get('n_prefill_layer')
        if prefill_layers is None:
            prefill_layers = _n_prefill_layer(model.cfg)
        self.stored_planes = [plane for layer, plane in sorted(model.cache.swa.items())
                              if layer < prefill_layers]
        self.plane_bytes = [storage._state_view(plane, 0).numel() * plane.cache.element_size()
                            for plane in storage.state_planes]
        if self.window < 1 or any(self.window > p.resident_pages * p.block_size
                                  for p in storage.state_planes):
            raise ValueError('native ring cannot cover the original inference state window')
        self.state_size_to_disk = sum(self.window * p.width * p.cache.element_size()
                                      for p in self.stored_planes)
        self.limit = 1 + (model.dspark.nextn if model.dspark is not None else 0)
        self.c2_widths = {layer.layer_id: (layer.compressor_wkv_weight.shape[0],
                                          layer.compressor_wgate_weight.shape[0])
                          for layer in model.layers
                          if getattr(layer, 'ratio', 0) == 2 and hasattr(layer, 'compressor_wkv_weight')}
        # Integer host history and descriptors; numeric state remains on device.
        self.header_size = 8 * model.max_tokens + 65536
        extra_bytes = 4 * (self.limit + 2) * sum(sum(x) for x in self.c2_widths.values())
        self.shape = (storage.num_state_seqs, self.header_size + sum(self.plane_bytes) + extra_bytes)
        self.transfer_header_size = 65536
        self.transfer_plane_bytes = [self.window * p.width * p.cache.element_size()
                                     for p in storage.state_planes]
        self.total_bytes_inference = self.transfer_header_size + sum(self.transfer_plane_bytes) + extra_bytes
        layout = dict(version=self.VERSION, max_tokens=model.max_tokens, window=self.window,
                      planes=[(tuple(storage._state_view(p, 0).shape), str(p.cache.dtype))
                              for p in storage.state_planes], c2=self.c2_widths, limit=self.limit,
                      stored=len(self.stored_planes))
        self.fingerprint = hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()
        transfer_layout = dict(version=self.VERSION, window=self.window, c2=self.c2_widths,
                               planes=[(p.width, str(p.cache.dtype)) for p in storage.state_planes],
                               stored=len(self.stored_planes), limit=self.limit)
        self.transfer_fingerprint = hashlib.sha256(json.dumps(transfer_layout, sort_keys=True).encode()).hexdigest()

    def _slot(self, state):
        if type(state) is not int or state not in self.storage.state_slots:
            raise ValueError('state ID has no native cache binding')
        return self.storage.state_slots[state]

    def _prepare_read(self):
        self.storage._assert_state_idle()
        work = getattr(self.model, '_original_device_work', None)
        event = getattr(work, 'event', None)
        if event is not None:
            # Preserve do_copy_state's enqueue-only contract and its existing
            # event owner; explicit restores/clones still drain before writes.
            torch.npu.current_stream(self.device).wait_event(event)

    def _bytes(self, value, size):
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.uint8
                or value.device != self.device or value.shape != (size,) or not value.is_contiguous()):
            raise ValueError('state transfer requires a contiguous native uint8 row of the exact size')
        start, end = value.data_ptr(), value.data_ptr() + value.numel()
        if any(start < plane.cache.data_ptr() + plane.cache.numel() * plane.cache.element_size()
               and plane.cache.data_ptr() < end for plane in self.storage.state_planes):
            raise ValueError('state transfer buffer must not overlap live native cache payloads')
        return value

    def _metadata(self, slot, position=None):
        model = self.model
        current = model.next_positions[slot]
        position = current if position is None else position
        history = getattr(model, '_original_state_history', None)
        if type(position) is not int or not 0 <= position <= current:
            raise ValueError('state endpoint is outside committed native history')
        if position != current:
            if history is None:
                raise ValueError('state endpoint has no retained rollback span')
            history.validate_position(slot, position)
        pending = {layer: values[slot] for layer, values in model.pending.items() if slot in values}
        valid = model.yoco_swa_valid_start[slot]
        span = None if history is None else history.spans.get(slot)
        if span is not None and span.end < current:
            # A later non-speculative prefill can advance the cursor without
            # recording another verification span. Its old span is expired.
            if position != current:
                raise ValueError('state endpoint belongs to an expired verification span')
            span = None
        if position != current:
            if position == 0:
                pending, valid, span = {}, 0, None
            else:
                valid = max(span.valid_before, position - 128, 0)
                if position == span.start:
                    pending = span.pending_before
                elif position % 2:
                    index = position - 1 - span.start
                    pending = {layer: tuple(t[index:index + 1] for t in pair)
                               for layer, pair in span.projected.items()}
                else:
                    pending = {}
        restored = getattr(model, '_original_prefix_restore_positions', {}).get(slot)
        tokens = list(model.histories[slot].history[:position]) if model.histories else []
        if model.histories and len(tokens) != position and restored != position:
            raise ValueError('Engram history is incomplete at the exported endpoint')
        meta = dict(version=self.VERSION, fingerprint=self.fingerprint, position=position,
                    valid_start=valid, restored=bool(position and slot in model.restored_prefix_slots),
                    prefix_restore=restored if restored == position else None,
                    history=tokens, pending={}, span=None, tensors=[])
        return meta, pending, span

    def read_state(self, state, position=None, out=None, *, _inference=False):
        """Materialize one full native row, including its unaccepted C2 span."""
        self._prepare_read()
        slot = self._slot(state)
        meta, pending, span = self._metadata(slot, position)
        header_size = self.transfer_header_size if _inference else self.header_size
        row_size = self.total_bytes_inference if _inference else self.shape[1]
        plane_bytes = self.transfer_plane_bytes if _inference else self.plane_bytes
        meta['mode'] = 'inference' if _inference else 'raw'
        meta['plane_end'] = span.end if span is not None else meta['position']
        if _inference:
            # Sequence data, already part of the original request transfer,
            # reconstructs Engram. The wire size must not depend on context
            # admission limits or PE/CE physical ring capacities.
            meta['history'] = []
            meta['prefix_restore'] = meta['position'] if self.model.histories else None
            meta['fingerprint'] = self.transfer_fingerprint
        out = (torch.zeros(row_size, dtype=torch.uint8, device=self.device)
               if out is None else self._bytes(out, row_size).zero_())
        offset = header_size
        for plane, size in zip(self.storage.state_planes, plane_bytes, strict=True):
            src = self.storage._state_view(plane, slot).view(torch.uint8)
            if _inference:
                from deepinfer.model_executor.layers.tilelang_ops.state_cache import select_state_slice
                select_state_slice(out[offset:offset + size].view(1, self.window, -1),
                                   src.view(1, plane.resident_pages * plane.block_size, -1), meta['plane_end'])
            else:
                out[offset:offset + size].copy_(src.flatten())
            offset += size

        def pack(mapping, rows):
            nonlocal offset
            result = {}
            for layer, pair in sorted(mapping.items()):
                widths = self.c2_widths.get(layer)
                if widths is None or len(pair) != 2:
                    raise ValueError('unknown native C2 state payload')
                refs = []
                for tensor, width in zip(pair, widths, strict=True):
                    if (tensor.dtype != torch.float32 or tensor.device != self.device
                            or tensor.ndim != 2 or tensor.shape != (rows, width)):
                        raise ValueError('C2 snapshot shape differs from its resident projection')
                    data = tensor.contiguous().view(torch.uint8).flatten()
                    if offset + data.numel() > out.numel():
                        raise ValueError('native C2 snapshot exceeds its fixed layout')
                    out[offset:offset + data.numel()].copy_(data)
                    refs.append(len(meta['tensors']))
                    meta['tensors'].append(dict(offset=offset, shape=list(tensor.shape)))
                    offset += data.numel()
                result[str(layer)] = refs
            return result

        meta['pending'] = pack(pending, 1)
        if span is not None:
            length = span.end - span.start
            if not 1 <= length <= self.limit:
                raise ValueError('native rollback span exceeds original verification width')
            meta['span'] = dict(start=span.start, end=span.end, valid_before=span.valid_before,
                                pending_before=pack(span.pending_before, 1),
                                projected=pack(span.projected, length))
        encoded = json.dumps(meta, separators=(',', ':'), sort_keys=True).encode()
        if len(encoded) + 4 > header_size:
            raise ValueError('native integer state metadata exceeds its fixed header')
        header = struct.pack('<I', len(encoded)) + encoded
        out[:len(header)].copy_(torch.tensor(list(header), dtype=torch.uint8, device=self.device))
        return out

    def write_state(self, state, value):
        """Restore a full row; validate its complete layout before destination writes."""
        from .w8a8_state_history910c import StateSpan

        self.storage._assert_drained()
        slot = self._slot(state)
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.numel() < 4:
            raise ValueError('native state row must include a complete byte header')
        value = self._bytes(value, value.numel())
        size = struct.unpack('<I', bytes(value[:4].cpu().tolist()))[0]
        if not 0 < size <= min(value.numel() - 4, max(self.header_size, self.transfer_header_size) - 4):
            raise ValueError('invalid native state header length')
        meta = json.loads(bytes(value[4:4 + size].cpu().tolist()))
        if meta.get('mode') not in ('raw', 'inference'):
            raise ValueError('unknown native state layout mode')
        inference = meta['mode'] == 'inference'
        expected_size = self.total_bytes_inference if inference else self.shape[1]
        header_size = self.transfer_header_size if inference else self.header_size
        plane_bytes = self.transfer_plane_bytes if inference else self.plane_bytes
        fingerprint = self.transfer_fingerprint if inference else self.fingerprint
        if (value.numel() != expected_size or size > header_size - 4
                or meta.get('version') != self.VERSION or meta.get('fingerprint') != fingerprint):
            raise ValueError('state belongs to a different native cache layout')
        position, valid, tokens = meta['position'], meta['valid_start'], meta['history']
        prefix = meta['prefix_restore']
        if (type(position) is not int or not 0 <= position <= self.model.max_tokens
                or type(valid) is not int or not 0 <= valid <= position
                or type(meta['restored']) is not bool
                or (prefix is not None and (type(prefix) is not int or prefix != position))
                or not isinstance(tokens, list)
                or any(type(t) is not int or not 0 <= t < self.model.cfg['vocab_size']+ImageTokenOffset.NUM_SPECIAL for t in tokens)
                or (self.model.histories and len(tokens) != position and prefix != position)
                or (not self.model.histories and tokens)):
            raise ValueError('invalid native state cursor or integer Engram history')
        plane_end = meta['plane_end']
        if type(plane_end) is not int or not position <= plane_end <= self.model.max_tokens:
            raise ValueError('invalid native inference window endpoint')
        tensors, cursor = [], header_size + sum(plane_bytes)
        for desc in meta['tensors']:
            offset, shape = desc['offset'], desc['shape']
            if (type(offset) is not int or offset != cursor or not isinstance(shape, list)
                    or len(shape) != 2 or any(type(x) is not int or x < 1 for x in shape)):
                raise ValueError('invalid native C2 tensor descriptor')
            count = shape[0] * shape[1] * 4
            if cursor + count > value.numel():
                raise ValueError('native C2 tensor exceeds its state row')
            tensors.append(value[cursor:cursor + count].view(torch.float32).view(shape))
            cursor += count
        used = set()

        def unpack(mapping, rows):
            result = {}
            for key, refs in mapping.items():
                layer = int(key)
                widths = self.c2_widths.get(layer)
                if widths is None or not isinstance(refs, list) or len(refs) != 2:
                    raise ValueError('unknown native C2 tensor references')
                pair = []
                for index, width in zip(refs, widths, strict=True):
                    if (type(index) is not int or not 0 <= index < len(tensors) or index in used
                            or tensors[index].shape != (rows, width)):
                        raise ValueError('native C2 tensor reference has a different shape or owner')
                    used.add(index)
                    pair.append(tensors[index].clone())
                result[layer] = tuple(pair)
            return result

        pending = unpack(meta['pending'], 1)
        if position % 2 and not self.c2_widths.keys() <= pending.keys():
            raise ValueError('odd native state endpoint lacks C2 pending data')
        span = None
        if meta['span'] is not None:
            s = meta['span']
            if (any(type(s[k]) is not int for k in ('start', 'end', 'valid_before'))
                    or not 0 <= s['valid_before'] <= s['start'] <= position <= s['end'] <= self.model.max_tokens
                    or not 1 <= s['end'] - s['start'] <= self.limit):
                raise ValueError('invalid native rollback span')
            before = unpack(s['pending_before'], 1)
            projected = unpack(s['projected'], s['end'] - s['start'])
            if s['start'] % 2 and not self.c2_widths.keys() <= before.keys():
                raise ValueError('odd native rollback start lacks C2 pending data')
            if not self.c2_widths.keys() <= projected.keys():
                raise ValueError('native rollback span lacks a C2 projection')
            span = StateSpan(s['start'], s['end'], s['valid_before'], before, projected)
        if plane_end != (span.end if span is not None else position):
            raise ValueError('native window endpoint differs from its retained rollback span')
        if used != set(range(len(tensors))):
            raise ValueError('unowned native C2 state payload')
        offset = header_size
        for plane, count in zip(self.storage.state_planes, plane_bytes, strict=True):
            target = self.storage._state_view(plane, slot)
            if inference:
                from deepinfer.model_executor.layers.tilelang_ops.state_cache import load_state_to_cache
                target.zero_()
                load_state_to_cache(value[offset:offset + count].view(1, self.window, -1),
                                    target.view(torch.uint8).view(1, plane.resident_pages * plane.block_size, -1), plane_end)
            else:
                target.copy_(value[offset:offset + count].view(target.dtype).view_as(target))
            offset += count
        model = self.model
        for values in model.pending.values():
            values.pop(slot, None)
        for layer, pair in pending.items():
            model.pending.setdefault(layer, {})[slot] = pair
        if model.histories:
            model.histories[slot].history = tokens
        model.next_positions[slot], model.yoco_swa_valid_start[slot] = position, valid
        model.restored_prefix_slots.discard(slot)
        if meta['restored']:
            model.restored_prefix_slots.add(slot)
        prefix_map = model._original_prefix_restore_positions
        prefix_map.pop(slot, None)
        if prefix is not None:
            prefix_map[slot] = prefix
        history = getattr(model, '_original_state_history', None)
        if history is not None:
            history.clear_slot(slot)
            if span is not None:
                history.spans[slot] = span
        self.storage._metadata_owners.pop(state, None)

    def __getitem__(self, state):
        if isinstance(state, tuple):
            state, columns = state
            return self.read_state(state)[columns]
        return self.read_state(state)

    def __setitem__(self, state, value):
        self.write_state(state, value)

    def zero_(self):
        self.storage._assert_drained()
        for state in self.storage.state_slots:
            self.storage._zero_state(state)
        return self

    def select_state_slice_to_store(self, state_id, last_hit_position, state_tensor=None, TO_CE=False):
        if TO_CE:
            return self.read_state(state_id, last_hit_position, state_tensor, _inference=True)
        self._prepare_read()
        slot = self._slot(state_id)
        current = self.model.next_positions[slot]
        if (type(last_hit_position) is not int or last_hit_position % 128
                or not 0 <= last_hit_position <= current
                or any(max(0, last_hit_position - self.window) < max(0, current - p.resident_pages * p.block_size)
                       for p in self.stored_planes)):
            raise ValueError('disk state endpoint must be an available original 128-token boundary')
        if not self.state_size_to_disk:
            return None
        from deepinfer.model_executor.layers.tilelang_ops.state_cache import select_state_slice

        out = (torch.empty(self.state_size_to_disk, dtype=torch.uint8, device=self.device)
               if state_tensor is None else self._bytes(state_tensor, self.state_size_to_disk))
        offset = 0
        for plane in self.stored_planes:
            source = self.storage._state_view(plane, slot).view(torch.uint8).view(
                1, plane.resident_pages * plane.block_size, -1)
            count = self.window * source.shape[-1]
            select_state_slice(out[offset:offset + count].view(1, self.window, -1), source, last_hit_position)
            offset += count
        return out

    def load_state_from_store_to_prefill_cache(self, state_id, last_hit_position, state_tensor):
        self.storage._assert_drained()
        slot = self._slot(state_id)
        if (type(last_hit_position) is not int or last_hit_position % 128
                or not 0 <= last_hit_position <= self.model.max_tokens):
            raise ValueError('disk restore requires an original 128-token boundary')
        state_tensor = self._bytes(state_tensor, self.state_size_to_disk)
        if not self.state_size_to_disk:
            return
        from deepinfer.model_executor.layers.tilelang_ops.state_cache import load_state_to_cache

        self.storage._zero_state(state_id)
        offset = 0
        for plane in self.stored_planes:
            target = self.storage._state_view(plane, slot).view(torch.uint8).view(
                1, plane.resident_pages * plane.block_size, -1)
            count = self.window * target.shape[-1]
            load_state_to_cache(state_tensor[offset:offset + count].view(1, self.window, -1), target, last_hit_position)
            offset += count
        self.model.next_positions[slot] = last_hit_position
        self.model.yoco_swa_valid_start[slot] = last_hit_position
        self.model.restored_prefix_slots.add(slot)
        self.model._original_prefix_restore_positions[slot] = last_hit_position
