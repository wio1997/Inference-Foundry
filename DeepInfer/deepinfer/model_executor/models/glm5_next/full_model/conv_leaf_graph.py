"""Bounded graphs of the existing GLM convolution; cache pages stay unchanged.

Only decode is replayed. Model metadata is copied into private input buffers on
every call. Capture and warmup use the reserved null page, never a live request.
No mathematical operator, dtype, weight, cache index, or state layout is changed.
"""
from collections import Counter


class ConvLeafGraphs:
    def __init__(self, original, *, max_requests, capture_sizes):
        import torch
        self.torch = torch
        self.original = original
        self.max_requests = max_requests
        self.capture_sizes = tuple(capture_sizes)
        self.enabled = False
        self.entries = {}
        self.warmed_layers = set()
        self.counts = Counter()
        self.stream = None
        self.module = None
        self.observed_contracts = set()

    def _key(self, x, weight, state, starts, indices, batch):
        return (state.data_ptr(), tuple(state.shape), tuple(state.stride()),
                weight.data_ptr(), tuple(weight.shape), weight.dtype,
                x.device, x.dtype, x.shape[-1], starts.dtype, indices.dtype, tuple(indices.shape[1:]), batch)

    def _entry(self, x, weight, state, starts, indices, batch):
        torch = self.torch
        key = self._key(x, weight, state, starts, indices, batch)
        if key in self.entries:
            return self.entries[key]
        if len(self.entries) >= 45 * self.max_requests:
            raise RuntimeError('convolution leaf graph bound exceeded')
        # Scratch holds only this batch. Physical cache storage and page strides
        # are the original SDK views. Null page zero is asserted before enabling.
        static_x = torch.zeros((batch, x.shape[-1]), device=x.device, dtype=x.dtype)
        static_starts = torch.arange(batch + 1, device=starts.device, dtype=starts.dtype)
        static_indices = torch.zeros((batch, *indices.shape[1:]), device=indices.device, dtype=indices.dtype)
        if self.stream is None:
            self.stream = torch.npu.Stream(device=x.device)
        current = torch.npu.current_stream(x.device)
        self.stream.wait_stream(current)
        with torch.npu.stream(self.stream):
            # Compile/warm existing kernels before capture. All dummy rows refer
            # to reserved page zero; none refer to user-owned request state.
            unused = self.original(static_x, weight, state, static_starts,
                                   static_indices, run_mode=1)
        self.stream.synchronize()
        del unused
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=self.stream):
            output = self.original(static_x, weight, state, static_starts,
                                   static_indices, run_mode=1)
        current.wait_stream(self.stream)
        entry = (graph, static_x, static_starts, static_indices, output, state, weight)
        self.entries[key] = entry
        self.counts['captures'] += 1
        return entry

    def __call__(self, x, weight, conv_state, query_start_loc, cache_indices,
                 *, run_mode, initial_state_mode=None, num_accepted_tokens=None):
        torch = self.torch
        if len(self.observed_contracts) < 16:
            self.observed_contracts.add((run_mode, tuple(x.shape), tuple(conv_state.shape), tuple(cache_indices.shape), tuple(query_start_loc.shape), str(x.device), str(weight.device), str(conv_state.device), str(cache_indices.device), str(query_start_loc.device)))
        eligible = (self.enabled and num_accepted_tokens is None and x.ndim == 2
                    and conv_state.ndim == 3 and cache_indices.ndim in (1, 2)
                    and query_start_loc.ndim == 1 and conv_state.shape[0] > 0
                    and all(t.device.type == 'npu' for t in
                            (x, weight, conv_state, query_start_loc, cache_indices)))
        if not eligible or torch.npu.is_current_stream_capturing():
            self.counts['fallback_contract_or_capture'] += 1
            return self.original(x, weight, conv_state, query_start_loc, cache_indices,
                run_mode=run_mode, initial_state_mode=initial_state_mode,
                num_accepted_tokens=num_accepted_tokens)
        layer = self._key(x, weight, conv_state, query_start_loc, cache_indices, 0)
        if layer not in self.warmed_layers:
            for batch in self.capture_sizes:
                self._entry(x, weight, conv_state, query_start_loc, cache_indices, batch)
            self.warmed_layers.add(layer)
        batch = cache_indices.shape[0]
        if (run_mode != 1 or not 1 <= batch <= self.max_requests
                or x.shape[0] != batch or query_start_loc.shape[0] != batch + 1):
            self.counts['fallback_prefill_or_shape'] += 1
            return self.original(x, weight, conv_state, query_start_loc, cache_indices,
                run_mode=run_mode, initial_state_mode=initial_state_mode,
                num_accepted_tokens=num_accepted_tokens)
        graph, static_x, static_starts, static_indices, output, _, _ = self._entry(
            x, weight, conv_state, query_start_loc, cache_indices, batch)
        # Always refresh every data-dependent operand. Replay advances live state
        # exactly once, independent of capture/warmup execution semantics.
        current = torch.npu.current_stream(x.device)
        self.stream.wait_stream(current)
        with torch.npu.stream(self.stream):
            static_x.copy_(x)
            static_starts.copy_(query_start_loc)
            static_indices.copy_(cache_indices)
            graph.replay()
            x.copy_(output)
        current.wait_stream(self.stream)
        self.counts['replays'] += 1
        return x

    def snapshot(self):
        return {'enabled': self.enabled, 'entries': len(self.entries),
                'layers': len(self.warmed_layers), 'counts': dict(self.counts),
                'observed_contracts': [list(x) for x in self.observed_contracts],
                'scope': 'existing convolution graph; exact active batch; reserved-page warmup'}

    def close(self):
        self.torch.npu.synchronize()
        if self.module is not None and self.module.causal_conv1d is self:
            self.module.causal_conv1d = self.original
        for graph, *_ in self.entries.values():
            graph.reset()
        self.entries.clear()


def install_conv_leaf_graphs(*, max_requests, capture_sizes):
    from vllm_ascend.models.glm5next import kda
    if isinstance(kda.causal_conv1d, ConvLeafGraphs):
        raise RuntimeError('convolution leaf graph already installed')
    result = ConvLeafGraphs(kda.causal_conv1d, max_requests=max_requests,
                            capture_sizes=capture_sizes)
    result.module = kda
    kda.causal_conv1d = result
    return result
