"""Storage operations for original CacheEngine commands on existing W8A8 buffers.

This module owns no allocator, scheduler or eviction policy. The caller supplies
the original-state-ID to physical-slot binding and must drain graph owners before
changing storage. Original metadata supplies block tables; it is not inferred
from request order or from the allocator's free-list policy.
"""
import numpy as np
import torch
from deepinfer.multimodal_tokens import ImageTokenOffset


class W8A8CacheStorage:
    def __init__(self, model, *, state_slots):
        self.model = model
        self.gpu_cache = model.cache
        self.state_slots = dict(state_slots)
        if (not self.state_slots or any(type(k) is not int or k < 0 for k in self.state_slots)
                or any(type(v) is not int or not 0 <= v < model.requests for v in self.state_slots.values())
                or set(self.state_slots) != set(range(model.requests))
                or len(set(self.state_slots.values())) != len(self.state_slots)
                or self.state_slots.get(0) != model.requests - 1):
            raise ValueError('original state bindings require distinct physical cache slots')
        self.state_planes = list(model.cache.swa.values())
        if model.dspark is not None:
            self.state_planes.extend(model.dspark.planes)
        self.block_planes = [plane for name in ('compressed', 'index_key', 'index_scale')
                             for plane in getattr(model.cache, name).values()]
        if not self.state_planes or not self.block_planes:
            raise ValueError('complete main state and compressed block planes are required')
        self.physical_blocks = self.block_planes[0].cache.shape[0]
        # Native C2 verification writes temporary rows in the final request's
        # private pages. Never expose those pages to the original allocator.
        self.scratch_slot = model.requests - 1
        self.num_gpu_blocks = self.scratch_slot * self.block_planes[0].resident_pages
        device = self.state_planes[0].cache.device
        for plane in self.state_planes:
            if (plane.requests != model.requests or plane.cache.shape[0] != model.requests * plane.resident_pages
                    or plane.cache.device != device):
                raise ValueError('state plane layout differs from physical bindings')
        for plane in self.block_planes:
            if (plane.cache.shape[0] != self.physical_blocks or plane.cache.device != device
                    or plane.resident_pages != self.block_planes[0].resident_pages):
                raise ValueError('compressed block planes must share original block coordinates')
        self.num_state_seqs = max(self.state_slots) + 1
        self._metadata_owners = {}
        self._state_cache = None
        if not hasattr(model, '_original_prefix_restore_positions'):
            model._original_prefix_restore_positions = {}

    @property
    def state_cache(self):
        if self._state_cache is None:
            from .w8a8_state_cache910c import W8A8StateCache
            self._state_cache = W8A8StateCache(self)
        return self._state_cache

    def restore_prefix_histories(self, metadata):
        """Rebuild Engram from original Sequence data after a disk-only restore."""
        restored = self.model._original_prefix_restore_positions
        prepared = []
        for row in metadata:
            slot = self.state_slots[row.state_id]
            if slot not in restored:
                continue
            position = restored[slot]
            tokens = row.restored_history
            if (position != row.seqlen_k - row.seqlen_q or not isinstance(tokens, tuple)
                    or len(tokens) != position
                    or any(type(t) is not int or not 0 <= t < self.model.cfg['vocab_size']+ImageTokenOffset.NUM_SPECIAL for t in tokens)):
                raise ValueError('restored state requires the original complete prefix history')
            prepared.append((slot, list(tokens)))
        for slot, tokens in prepared:
            if self.model.histories:
                self.model.histories[slot].history = tokens
            restored.pop(slot)

    @classmethod
    def from_model(cls, model):
        # BlockManager reserves ID zero for both block and state managers.
        # Real states 1..N address real slots 0..N-1; zero is scratch only.
        return cls(model, state_slots={0: model.requests - 1,
                                      **{state: state - 1 for state in range(1, model.requests)}})

    def cache_parameters(self):
        """Describe existing physical payloads for original CacheConfig.

        C2 keeps one FP32 projection/gate pair per state outside the ring
        planes. Count its maximum payload from the real projection shapes;
        request history remains host data, as in the existing executor.
        """
        pending_bytes = 0
        for layer in self.model.layers:
            if getattr(layer, 'ratio', 0) == 2 and hasattr(layer, 'compressor_wkv_weight'):
                pending_bytes += 4 * (layer.compressor_wkv_weight.shape[0]
                                      + layer.compressor_wgate_weight.shape[0])
        state_bytes = sum(plane.resident_pages * plane.cache[0].numel()
                          * plane.cache.element_size() for plane in self.state_planes)
        block_bytes = sum(plane.cache[0].numel() * plane.cache.element_size()
                          for plane in self.block_planes)
        return dict(block_size=128, num_gpu_blocks=self.num_gpu_blocks,
                    num_state_seqs=self.num_state_seqs, bytes_per_block=block_bytes,
                    bytes_of_state_per_seqs=state_bytes + pending_bytes)

    def bind_metadata(self, metadata):
        """Resolve original state IDs, preserving original metadata order.

        This does not admit requests, allocate IDs, reset state or free old
        owners. The original Scheduler controls those operations. A caller
        must drain graph owners before changing these address bindings.
        """
        tables, slots, ids, owners = {}, [], set(), {}
        for row in metadata:
            if row.is_placeholder:
                raise ValueError('placeholder metadata has no original cache owner')
            state, seq_id = row.state_id, row.seq_id
            if (type(state) is not int or not 0 < state < self.num_state_seqs
                    or type(seq_id) is not int or seq_id < 0 or seq_id in ids):
                raise ValueError('invalid or duplicate original request/state ID')
            slot = self.state_slots[state]
            if slot in tables:
                raise ValueError('active original requests still share a state; COW is required')
            q, k = row.seqlen_q, row.seqlen_k
            blocks = tuple(row.block_table)
            if (type(q) is not int or type(k) is not int or not 1 <= q <= k <= self.model.max_tokens
                    or len(blocks) < (k + 127) // 128):
                raise ValueError('original metadata exceeds allocated cache coverage')
            tables[slot] = blocks
            slots.append(slot)
            ids.add(seq_id)
            owners[state] = seq_id
        if not slots:
            return ()
        # Steady decode may reuse address bindings while the previous sample
        # is in flight. No table upload or graph drain is needed unless IDs or
        # request ownership changed. New ownership still requires a drain even
        # when the original allocator reuses exactly the same block IDs.
        if (all(self._metadata_owners.get(state) == seq for state, seq in owners.items())
                and all(plane._bound_tables.get(slot) == blocks
                        for plane in self.block_planes for slot, blocks in tables.items())):
            return tuple(slots)
        self.bind_block_tables(tables)
        self._metadata_owners.update(owners)
        return tuple(slots)

    def bind_block_tables(self, tables):
        """Bind physical request slots to tables supplied by original metadata."""
        self._assert_drained()
        tables = {slot: tuple(blocks) for slot, blocks in tables.items()}
        if any(type(slot) is not int or not 0 <= slot < self.scratch_slot for slot in tables):
            raise ValueError('original request cannot bind the private scratch slot')
        if any(type(block) is not int or not 0 < block < self.num_gpu_blocks
               for blocks in tables.values() for block in blocks):
            raise ValueError('original block table aliases reserved or scratch storage')
        self.model.cache.bind_block_tables(tables)

    @staticmethod
    def _pairs(value, name):
        data = np.asarray(value)
        if data.size == 0:
            return ()
        if data.ndim != 2 or data.shape[1] != 2 or data.dtype != np.int32:
            raise ValueError(f'{name} requires an int32 [N,2] array')
        return tuple(tuple(pair) for pair in data.tolist())

    @staticmethod
    def _zeros(value):
        data = np.asarray(value)
        if data.size == 0:
            return ()
        if data.ndim != 1 or data.dtype != np.int32:
            raise ValueError('zero_states requires an int32 [N] array')
        return tuple(data.tolist())

    def _assert_state_idle(self):
        history = getattr(self.model, '_original_state_history', None)
        if history is not None and history.recording:
            raise RuntimeError('cache commands require a completed target state recording')
        if any(getattr(self.model, name, None) is not None for name in
               ('_zos_device_pipeline', '_zos_prepared_frame', '_zos_commit_frame')):
            raise RuntimeError('cache commands require the current graph pipeline to be drained and closed')

    def _assert_drained(self):
        self._assert_state_idle()
        work = getattr(self.model, '_original_device_work', None)
        if work is not None:
            work.drain()

    def _validate_state_metadata(self, states):
        model = self.model
        for state in states:
            if state not in self.state_slots:
                raise ValueError('original state ID has no physical cache binding')
            slot = self.state_slots[state]
            position = model.next_positions[slot]
            valid = model.yoco_swa_valid_start[slot]
            if type(position) is not int or type(valid) is not int or not 0 <= valid <= position <= model.max_tokens:
                raise ValueError('invalid bound state cursor or YOCO visibility')
            if (model.histories and len(model.histories[slot].history) != position
                    and model._original_prefix_restore_positions.get(slot) != position):
                raise ValueError('bound Engram history differs from committed state cursor')
            for values in model.pending.values():
                if slot in values:
                    pair = values[slot]
                    if (not isinstance(pair, tuple) or len(pair) != 2
                            or any(not isinstance(t, torch.Tensor) or t.dtype != torch.float32
                                   or t.device != self.state_planes[0].cache.device for t in pair)):
                        raise ValueError('invalid C2 state payload')

    def _state_view(self, plane, slot):
        start = slot * plane.resident_pages
        return plane.cache[start:start + plane.resident_pages]

    def _clone_state(self, source, destination):
        src, dst = self.state_slots[source], self.state_slots[destination]
        if src == dst:
            return
        model = self.model
        # Allocate the small pending-pair copies before modifying destination
        # storage. Sequential commands see the preceding command's new state.
        pending = {layer: tuple(t.clone() for t in values[src])
                   for layer, values in model.pending.items() if src in values}
        history = list(model.histories[src].history) if model.histories else None
        for plane in self.state_planes:
            self._state_view(plane, dst).copy_(self._state_view(plane, src))
        for values in model.pending.values():
            values.pop(dst, None)
        for layer, pair in pending.items():
            model.pending[layer][dst] = pair
        if model.histories:
            model.histories[dst].history = history
        model.next_positions[dst] = model.next_positions[src]
        model.yoco_swa_valid_start[dst] = model.yoco_swa_valid_start[src]
        model._original_prefix_restore_positions.pop(dst, None)
        if src in model._original_prefix_restore_positions:
            model._original_prefix_restore_positions[dst] = model._original_prefix_restore_positions[src]
        if src in model.restored_prefix_slots:
            model.restored_prefix_slots.add(dst)
        else:
            model.restored_prefix_slots.discard(dst)
        if getattr(model, '_original_state_history', None) is not None:
            model._original_state_history.clone_slot(src, dst)

    def _zero_state(self, state):
        slot = self.state_slots[state]
        for plane in self.state_planes:
            self._state_view(plane, slot).zero_()
        for values in self.model.pending.values():
            values.pop(slot, None)
        if self.model.histories:
            self.model.histories[slot].history = []
        self.model.next_positions[slot] = 0
        self.model.yoco_swa_valid_start[slot] = 0
        self.model.restored_prefix_slots.discard(slot)
        self.model._original_prefix_restore_positions.pop(slot, None)
        self._metadata_owners.pop(state, None)
        if getattr(self.model, '_original_state_history', None) is not None:
            self.model._original_state_history.clear_slot(slot)

    def clone_and_zero_states_and_blocks(self, clone_states, clone_blocks, zero_states):
        state_pairs = self._pairs(clone_states, 'clone_states')
        block_pairs = self._pairs(clone_blocks, 'clone_blocks')
        zeros = self._zeros(zero_states)
        if not state_pairs and not block_pairs and not zeros:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        self._assert_drained()
        # Validate the entire command before any destination write, including
        # blocks that only release a source pin (destination zero).
        states = {state for pair in state_pairs for state in pair} | set(zeros)
        if 0 in states:
            raise ValueError('state zero is reserved by the original allocator')
        self._validate_state_metadata(states)
        if any(not 0 < source < self.num_gpu_blocks or not 0 <= destination < self.num_gpu_blocks
               for source, destination in block_pairs):
            raise ValueError('original block ID is outside physical cache capacity')
        for source, destination in state_pairs:
            self._clone_state(source, destination)
        for source, destination in block_pairs:
            if destination:
                for plane in self.block_planes:
                    plane.cache[destination].copy_(plane.cache[source])
        for state in zeros:
            self._zero_state(state)
        # Owned arrays remain valid through the original DeferredRelease event;
        # a subsequent call cannot overwrite these source-unpin IDs.
        return (np.asarray([source for source, _ in state_pairs], dtype=np.int32),
                np.asarray([source for source, _ in block_pairs], dtype=np.int32))
