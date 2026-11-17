"""Bounded actual-model QLI shadow; never enabled by the optimization flag.

Two eager chunks per worker, using only the existing attention TP group. No
whole-pool snapshots, new groups, model forwards or duplicate sparse attention.
The caller must gate scoring on all workers' terminal ``disabled`` records.
"""
import hashlib
import json
import os
from pathlib import Path

DIRECTORY = os.environ.get('SGLANG_G46_VALIDATE_DIR', '')


def validation_armed():
    # Root creates this sentinel only after service readiness. A startup eager
    # profiling batch must not consume an actual-request validation window.
    return bool(DIRECTORY and (Path(DIRECTORY) / 'ARMED').is_file())


def digest(tensor):
    import torch
    cpu = tensor.detach().contiguous().cpu()
    return dict(shape=list(cpu.shape), dtype=str(cpu.dtype),
                sha256=hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest())


def integer_counts(tensor):
    return dict(elements=tensor.numel(), valid_elements=int((tensor >= 0).sum().item()),
                minimum=int(tensor.min().item()), maximum=int(tensor.max().item()))


def logical_digest(tensor, table, tokens):
    """Only live logical tokens in PA_BBND order, never the allocated pool."""
    import torch
    page_size = tensor.shape[1]
    count = (tokens + page_size - 1) // page_size
    pages = table[0, :count].to(dtype=torch.int64)
    assert pages.numel() == count
    logical = tensor.index_select(0, pages).flatten(0, 1)[:tokens]
    return digest(logical)


def tensor_versions(values):
    """Identity + mutation counters, without copying large sparse KV pools."""
    import torch
    result = {}
    for name, value in values.items():
        if not isinstance(value, torch.Tensor):
            continue
        try:
            version = value._version
        except RuntimeError:
            version = 'unavailable_inference_tensor'
        result[name] = dict(pointer=value.data_ptr(), shape=list(value.shape),
                            dtype=str(value.dtype), version=version)
    return result


class QliShadow:
    def __init__(self, rank, expected_sources, expected_sparse):
        self.rank = rank
        self.path = Path(DIRECTORY) / ('global_tp_rank_%s.json' % rank)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Existing evidence must not silently be confused with this process.
        if self.path.exists():
            raise RuntimeError('G46 shadow evidence already exists: %s' % self.path)
        self.expected_sources = set(expected_sources)
        self.expected_sparse = set(expected_sparse)
        self.windows = []
        self.active = None
        self.disabled = False
        self.write('armed')

    def write(self, status, **extra):
        record = dict(schema=1, status=status, shadow_enabled=not self.disabled,
                      disabled=self.disabled,
                      all_checks_passed=(self.disabled and all(
                          c['passed'] for w in self.windows for c in w['checks'])),
                      pid=os.getpid(), global_tp_rank=self.rank,
                      expected_sources=sorted(self.expected_sources),
                      expected_sparse=sorted(self.expected_sparse),
                      windows=self.windows, **extra)
        temporary = self.path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        os.replace(temporary, self.path)

    def check(self, label, passed, **details):
        entry = dict(label=label, passed=bool(passed), **details)
        self.active['checks'].append(entry)
        if not passed:
            self.write('failed', failure=entry)
            raise AssertionError('G46 actual-model shadow: %s' % label)

    def begin(self, global_end, rows, parallel, request_ids, fallback_seen):
        if self.disabled:
            return False
        if self.active is not None:
            self.check('previous_window_finished_before_metadata_refresh', False)
        if global_end in [w['global_end'] for w in self.windows]:
            return False
        if self.windows and request_ids != self.windows[0]['request_ids']:
            # Distinct requests cannot prove prefix/chunk continuity. Keep the
            # gate pending rather than silently substituting a different one.
            return False
        self.active = dict(global_end=global_end, query_rows=rows,
                           absolute_start=global_end-rows,
                           request_ids=request_ids,
                           naturally_observed_fallback_original_qli=fallback_seen,
                           static_fallback_review=['graph', 'decode', 'verify', 'batch_gt1'],
                           attn_tp_rank=parallel.attn_tp_rank,
                           attn_tp_size=parallel.attn_tp_size,
                           sources=[], sparse=[], checks=[], candidate_source=None,
                           candidate_digest=None)
        self.windows.append(self.active)
        if len(self.windows) == 2:
            self.check('same_request_consecutive_prefill_chunks',
                       global_end == self.windows[0]['global_end'] + rows,
                       first_global_end=self.windows[0]['global_end'],
                       second_global_end=global_end, request_ids=request_ids)
        self.write('active')
        return True

    def rank_equal(self, label, hashes, device):
        import torch
        from sglang.srt.layers.dp_attention import attn_tp_all_gather_into_tensor
        payload = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).digest()
        local = torch.tensor(list(payload), dtype=torch.int32, device=device)
        full = local.new_empty(4 * local.numel())
        attn_tp_all_gather_into_tensor(full, local)
        records = full.view(4, -1).cpu().tolist()
        self.check(label, all(item == records[0] for item in records),
                   rank_sha256=[bytes(item).hex() for item in records], inputs=hashes)

    def wrap(self, partitioned, layer_id, positions, finalize):
        return ShadowCall(self, partitioned, layer_id, positions, finalize)

    def sparse(self, layer_id, index_source, indices, lengths, call):
        """Same sparse operation and same tensor objects; verify reuse/read-only."""
        state = self.active
        if state is None:
            return
        reference = next(item for item in state['sources'] if item['layer_id'] == index_source)
        current = dict(indices=digest(indices.view(*reference['indices']['shape'])),
                       lengths=digest(lengths))
        self.check('sparse_indices_and_lengths_source_reuse',
                   current == {k: reference[k] for k in current},
                   layer_id=layer_id, index_source=index_source, **current)
        state['sparse'].append(layer_id)
        self.write('active')

    def finish_sparse(self, layer_id, before, after):
        self.check('sparse_call_input_identity_and_available_versions_unchanged', before == after,
                   layer_id=layer_id, tensors=before)
        if layer_id != max(self.expected_sparse):
            return
        self.check('all_index_sources_seen',
                   {s['layer_id'] for s in self.active['sources']} == self.expected_sources)
        self.check('all_sparse_layers_seen', set(self.active['sparse']) == self.expected_sparse)
        self.active['status'] = 'passed_exact'
        self.active = None
        if len(self.windows) == 2:
            self.disabled = True
            self.write('disabled', terminal=True,
                       disable_reason='two_distinct_actual_prefill_chunks_completed',
                       sparse_equivalence='identical original call and verified finalized indices/lengths; no repeated attention',
                       score_permitted_only_after_all_16_workers_disabled=True)
        else:
            self.write('armed')


class ShadowCall:
    def __init__(self, shadow, partitioned, layer_id, positions, finalize):
        self.shadow, self.partitioned = shadow, partitioned
        self.layer_id, self.positions, self.finalize = layer_id, positions, finalize

    def __call__(self, q, key, weights, qscale, key_scale, topk, quant_mode, **kwargs):
        import torch
        s = self.shadow
        mode = kwargs['candidate_mode']
        tokens = self.partitioned.plan.global_end // self.partitioned.plan.ratio
        plan = self.partitioned.plan
        expected_positions = torch.arange(plan.global_end-plan.total_rows,
                                           plan.global_end, dtype=self.positions.dtype,
                                           device=self.positions.device)
        s.check('actual_positions_match_partition_causal_geometry',
                torch.equal(self.positions, expected_positions), layer_id=self.layer_id,
                global_end=plan.global_end, first_position=plan.global_end-plan.total_rows,
                last_position=plan.global_end-1)
        del expected_positions

        def fingerprint():
            result = {name: digest(t) for name, t in dict(
                q=q, qscale=qscale, weights=weights, positions=self.positions,
                block_table=kwargs['block_table'], metadata=kwargs['metadata'],
                cu=kwargs['cu_seqlens_q'], seq=kwargs['seqused_k']).items()}
            for name in ('cmp_residual_k', 'candidate_topk_index'):
                if kwargs.get(name) is not None:
                    result[name] = digest(kwargs[name])
            result['logical_key'] = logical_digest(key, kwargs['block_table'], tokens)
            result['logical_scale'] = logical_digest(key_scale, kwargs['block_table'], tokens)
            return result

        before = fingerprint()
        # Physical page numbers and opaque kernel scheduling can differ across
        # workers; compare logical data and semantic lengths across the group.
        replicated = {k: v for k, v in before.items() if k not in ('block_table', 'metadata')}
        s.rank_equal('attention_tp4_real_operands_identical', replicated, q.device)
        if mode == 2:
            s.check('candidate_consumer_uses_published_source',
                    before['candidate_topk_index'] == s.active['candidate_digest'],
                    layer_id=self.layer_id, candidate_source=s.active['candidate_source'])
        # The reference uses the original full QLI metadata and the EXACT same
        # projected/quantized tensors. No second model forward or quantization.
        reference, _, candidate_ref = self.partitioned.bindings.qli(
            q, key, weights, qscale, key_scale, topk, quant_mode, **kwargs)
        selected, scores, published = self.partitioned(
            q, key, weights, qscale, key_scale, topk, quant_mode, **kwargs)
        s.check('full_vs_partitioned_raw_exact', torch.equal(reference, selected),
                layer_id=self.layer_id, mode=mode, ratio=self.partitioned.plan.ratio,
                raw_shape=list(selected.shape), raw_elements=selected.numel(),
                raw_counts=integer_counts(selected),
                local_rows=[self.partitioned.plan.start, self.partitioned.plan.end],
                local_causal_end=self.partitioned.plan.causal_end)
        if mode == 1:
            s.check('full_vs_partitioned_candidates_exact', torch.equal(candidate_ref, published),
                    layer_id=self.layer_id, counts=integer_counts(published))
            s.active['candidate_source'] = self.layer_id
            s.active['candidate_digest'] = digest(published)
        s.check('qli_inputs_and_live_logical_cache_unchanged', before == fingerprint(),
                layer_id=self.layer_id, before=before)
        self.reference = reference
        self.mode = mode
        return selected, scores, published

    def finalize_indices(self, selected, positions, ratio, **kwargs):
        import torch
        lengths = torch.empty_like(kwargs['lengths_output'])
        expected = self.finalize(self.reference.squeeze(1), positions, ratio,
                                 lengths_output=lengths)
        self.reference = None
        actual = self.finalize(selected, positions, ratio, **kwargs)
        s = self.shadow
        s.check('full_vs_partitioned_final_indices_exact', torch.equal(expected, actual),
                layer_id=self.layer_id, elements=actual.numel())
        s.check('full_vs_partitioned_final_lengths_exact',
                torch.equal(lengths, kwargs['lengths_output']), layer_id=self.layer_id,
                elements=lengths.numel())
        s.active['sources'].append(dict(layer_id=self.layer_id, mode=self.mode,
                                        ratio=ratio, indices=digest(actual), lengths=digest(lengths),
                                        index_counts=integer_counts(actual), length_counts=integer_counts(lengths)))
        s.write('active')
        return actual
