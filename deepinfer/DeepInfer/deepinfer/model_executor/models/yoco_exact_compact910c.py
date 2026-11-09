# SPDX-License-Identifier: Apache-2.0
"""Exact suffix compaction at the final shared-KV boundary for 910C prefill.

This is not DeepInfer's approximate bounded replay: every retained output and
per-layer SWA state must remain exact. The current candidate covers only the
unpadded, restored-prefix sequence-parallel/CP path and never edits weights.
"""
import os
import math
import time
import torch

from deepinfer.npu_ops.cp910c import QuerySlice
from deepinfer.npu_ops.dsa910c import DSABatch


def suffix_plan(lengths, keep):
    if type(keep) is not int or keep < 1 or not lengths or any(type(q) is not int or q < 1 for q in lengths):
        raise ValueError('positive integer query lengths and retention bound required')
    selected, compact_lengths, offset = [], [], 0
    for q in lengths:
        kept = min(q, keep)
        compact_lengths.append(kept)
        selected.extend(range(offset + q - kept, offset + q))
        offset += q
    return tuple(compact_lengths), selected


def plan_key(model, state, keep):
    cp = state.context_query
    return (tuple(state.batch.query_lengths), keep, model.sp.world, model.sp.rank,
            cp.start, cp.end, cp.width, str(model.device))


def prepare_compaction_plan(model, state, keep, *, official_selection=None):
    """Prepare the native contiguous gather for a legacy or official suffix."""
    old_batch, old_cp = state.batch, state.context_query
    queries = list(old_batch.query_lengths)
    key = plan_key(model, state, keep)
    if official_selection is None:
        lengths, selected = suffix_plan(queries, keep)
    else:
        lengths, selected = official_selection
        lengths, selected = tuple(lengths), list(selected)
        if len(lengths) != len(queries) or any(
                type(n) is not int or not 1 <= n <= q for n, q in zip(lengths, queries)):
            raise ValueError('invalid official YOCO consumer lengths')
        expected, offset = [], 0
        for q, n in zip(queries, lengths):
            expected.extend(range(offset + q - n, offset + q))
            offset += q
        if selected != expected:
            raise ValueError('official YOCO indices do not match per-request suffixes')
        # Keep plans derived from full-request metadata outside the legacy cache.
        key += (lengths,)
    total = sum(queries)
    if len(selected) == total:
        return None
    local_rows = [i - old_cp.start for i in selected if old_cp.start <= i < old_cp.end]
    counts = []
    for rank in range(model.sp.world):
        start = min(rank * old_cp.width, total)
        end = min((rank + 1) * old_cp.width, total)
        counts.append(sum(start <= i < end for i in selected))
    width = max(counts)
    if not width or counts[model.sp.rank] != len(local_rows) or sum(counts) != len(selected):
        raise ValueError('invalid old CP contribution plan')
    local_index = torch.tensor(local_rows, dtype=torch.int64, device=model.device)
    gathered_index = torch.tensor([rank * width + j for rank, count in enumerate(counts)
                                   for j in range(count)], dtype=torch.int64, device=model.device)
    return (key, lengths, selected, local_rows, width, local_index, gathered_index)


def compact(model, state, trunk, previous, *, bounded_keep=None):
    detail = os.environ.get('DEEPINFER_910C_PREFILL_TIMING_DETAIL') == '1'
    marks = []
    def stamp(label):
        if detail:
            marks.append((label, time.perf_counter()))
    stamp('begin')
    old_batch = state.batch
    old_cp = state.context_query
    if old_cp is None or not state.sequence_parallel or state.prefill_padding_margin:
        return None
    source = model.tail_ffn_source
    if not model.tail_ffn or source != max(model.cfg['kv_source_layer_ids']):
        raise ValueError('compaction requires the validated exact FFN-tail boundary')
    if any(i > source for i in model.cfg.get('engram_layer_ids', [])):
        raise ValueError('downstream Engram rows are not covered by this candidate')
    if bounded_keep is None:
        keep = 1 + (len(model.layers) - 1 - source) * (model.cfg['sliding_window'] - 1)
    else:
        keep=bounded_keep
        if (type(keep) is not int or keep not in (140,256) or model.dspark is None or
                keep < model.cfg['sliding_window']+model.dspark.nextn+1 or
                len(set(old_batch.seq_lengths))!=1 or old_batch.seq_lengths[0]%128 or
                old_batch.seq_lengths[0]<=keep or any(q<keep for q in old_batch.query_lengths)):
            raise ValueError('invalid page-aligned bounded replay contract')
    plan = getattr(state, 'yoco_prepared_plan', None)
    if plan is not None:
        if plan[0] != plan_key(model, state, keep):
            raise ValueError('prepared YOCO plan no longer matches the input layout')
        if state.index_source != source or state.selected is None:
            raise ValueError('the final source index must be available before compaction')
        model.yoco_preplan_consumed += 1
        _, lengths, selected, local_rows, width, local_index, gathered_index = plan
    else:
        lengths, selected = suffix_plan(list(old_batch.query_lengths), keep)
        total = sum(old_batch.query_lengths)
        if len(selected) == total:
            return None
        if state.index_source != source or state.selected is None:
            raise ValueError('the final source index must be available before compaction')
        # Match the official compaction pattern: each old CP rank contributes
        # only retained rows, pads to the common contribution width, gathers,
        # and removes rank padding before the new partition is formed.
        local_rows = [i - old_cp.start for i in selected if old_cp.start <= i < old_cp.end]
        counts = []
        for rank in range(model.sp.world):
            start = min(rank * old_cp.width, total)
            end = min((rank + 1) * old_cp.width, total)
            counts.append(sum(start <= i < end for i in selected))
        width = max(counts)
        if not width or counts[model.sp.rank] != len(local_rows) or sum(counts) != len(selected):
            raise ValueError('invalid old CP contribution plan')
        local_index = torch.tensor(local_rows, dtype=torch.int64, device=model.device)
        gathered_index = torch.tensor([rank * width + j for rank, count in enumerate(counts)
                                       for j in range(count)], dtype=torch.int64, device=model.device)

    def gather_kept(value, fill=0):
        if value is None:
            return None
        expected = old_cp.end - old_cp.start
        if value.shape[0] < expected:
            raise ValueError('query-aligned CP value has fewer rows than its owner slice')
        contribution = value.index_select(0, local_index)
        padded = value.new_full((width, *value.shape[1:]), fill)
        padded[:len(local_rows)].copy_(contribution)
        gathered = model.sp.gather(padded, width * model.sp.world)
        return gathered.index_select(0, gathered_index)

    stamp('metadata_start')
    if os.environ.get('DEEPINFER_910C_YOCO_META_FUSE') == '1':
        sel, cand = state.selected, state.candidates
        if cand is None or sel.dtype != cand.dtype or sel.shape[0] != cand.shape[0]:
            raise ValueError('metadata fusion requires same-dtype query-aligned indices and candidates')
        sel_shape, cand_shape = sel.shape[1:], cand.shape[1:]
        sel_width, cand_width = math.prod(sel_shape), math.prod(cand_shape)
        packed = torch.cat((sel.reshape(sel.shape[0], sel_width),
                            cand.reshape(cand.shape[0], cand_width)), dim=1)
        merged = gather_kept(packed, fill=-1)
        indices = merged[:, :sel_width].reshape(len(selected), *sel_shape)
        candidates = merged[:, sel_width:].reshape(len(selected), *cand_shape)
        model.yoco_metadata_fusions += 1
    else:
        indices = gather_kept(state.selected, fill=-1)
        candidates = gather_kept(state.candidates, fill=-1)
    stamp('metadata_done')
    compact_trunk = gather_kept(trunk)
    stamp('trunk_done')
    compact_previous = gather_kept(previous)
    stamp('previous_done')
    batch = DSABatch.build(lengths, old_batch.seq_lengths, model.device)
    new = type(state)(batch, state.slots, batch.positions(),
                      model.cache.swa_plan(0, batch, state.slots),
                      model.cache.swa[0].table(state.slots))
    new.valid_query_lengths = lengths
    new.prefill_padding_margin = 0
    if bounded_keep is not None:
        # Original KV table is shifted only at attention consumption. Writes,
        # compressed KV, positions and index selection remain absolute.
        new.bounded_swa_base=old_batch.seq_lengths[0]-keep
        new.bounded_swa_keep=keep
    new.source_metadata = state.source_metadata
    new.mc2_global_bs = state.mc2_global_bs
    new.sequence_parallel = True
    new.bind_row_controls(model.sp,state.full_images_seq_mask,
                          state.full_gate_force_random,selected=selected)
    cp = QuerySlice.build(batch.query_lengths, batch.seq_lengths, model.sp.world, model.sp.rank)
    new.context_query = cp
    new.query_slots = [state.slots[i] for i in cp.request_indices]
    if cp.query_lengths:
        new.query_batch = DSABatch.build(cp.query_lengths, cp.seq_lengths, model.device)
        new.query_swa_table = model.cache.swa[0].table(new.query_slots)
    new.selected = indices[cp.start:cp.end].contiguous()
    new.candidates = candidates[cp.start:cp.end].contiguous() if candidates is not None else None
    new.index_source = state.index_source
    new.tail_ffn_indices = {}
    for layer in range(source + 1, len(model.layers)):
        required = 1 + (len(model.layers) - 1 - layer) * (model.cfg['sliding_window'] - 1)
        rows, offset = [], 0
        for q in batch.query_lengths:
            left = max(offset + max(0, q - required), cp.start)
            right = min(offset + q, cp.end)
            rows.extend(range(left - cp.start, max(left, right) - cp.start))
            offset += q
        new.tail_ffn_indices[layer] = torch.tensor(rows, dtype=torch.int64, device=model.device)
    if model.dspark is not None:
        model.dspark.prepare_capture(new, model.sp)
    result = model.sp.split(compact_trunk), model.sp.split(compact_previous), new
    stamp('done')
    if detail:
        model._prefill_compaction_detail = {marks[i][0] + '_to_' + marks[i+1][0] + '_seconds': marks[i+1][1]-marks[i][1] for i in range(len(marks)-1)}
    return result
