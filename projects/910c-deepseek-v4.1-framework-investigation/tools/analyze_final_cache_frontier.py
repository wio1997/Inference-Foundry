#!/usr/bin/env python3
"""Source-bound physical-cache/row domains and a retained-cache C8 sensitivity.

No candidate kernel, model import, device work, or new timing is performed.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_final_producer_entry import latest_dp_change
from price_vocab_selection_fusion import records, cached, BODY

OUT = Path('audit/execution-search')

def read(p):
    return json.loads(Path(p).read_bytes())

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    identity = read(OUT/'final-cache-execution-identity.json')
    for r in identity['sources']:
        assert sha(r['local_path']) == r['runtime_sha256']
    src = {r['runtime_path'].split('/vllm-workspace/')[1]: Path(r['local_path']) for r in identity['sources']}
    # Execute only the original integer retention method on dummy scalars.
    # This is source interpretation evidence, not a replacement test.
    p = src['vllm/vllm/v1/core/single_type_kv_cache_manager.py']
    tree = ast.parse(p.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SlidingWindowManager')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_num_skipped_tokens')
    ns = {}
    exec(compile('from __future__ import annotations\n'+ast.get_source_segment(p.read_text(), method), str(p), 'exec'), ns)
    dummy = type('Retention', (), dict(sliding_window=128, extra_retained_tokens=0))()
    examples = []
    for committed in [0, 3200, 3328, 3334]:
        skipped = ns['get_num_skipped_tokens'](dummy, committed)
        examples.append(dict(committed_prefix=committed, skipped_tokens=skipped,
                             freed_full_blocks=skipped//128, retained_block_start=(skipped//128)*128))
    assert examples[2]['retained_block_start'] == 3200
    # Re-evaluate the allocator's original topology-only helper with inert
    # marker specs. Byte capacities below are dummy units, not size evidence.
    cp = src['vllm-ascend/vllm_ascend/models/deepseek_v41/cache_config.py']
    cp_tree = ast.parse(cp.read_text())
    class MLA: pass
    class SWA: pass
    class State: pass
    cpns = dict(AscendMLAAttentionSpec=MLA, AscendSlidingWindowMLASpec=SWA, CircularBufferSpec=State)
    for name in ['_layer_number','_draft_layer_number','get_layer_tuples']:
        n = next(n for n in cp_tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile('from __future__ import annotations\n'+ast.get_source_segment(cp.read_text(),n),str(cp),'exec'),cpns)
    specs = {}
    plan = read('audit/code-led/target-layer-plan.json')
    def marker(cls):
        v = cls(); v.unpadded_page_size_bytes = 1; return v
    for role in plan['layers']:
        prefix = f"model.layers.{role['layer_idx']}.self_attn"
        specs[prefix+'.swa_cache'] = marker(SWA)
        if role['is_kv_source']:
            specs[prefix+'.long_kv_cache'] = marker(MLA)
            specs[prefix+'.indexer.k_cache'] = marker(MLA)
        if role['is_kv_source'] and role['compress_ratio']==2:
            specs[prefix+'.compressor.state_cache'] = marker(State)
    for i in range(3):
        specs[f'model.mtp.{i}.self_attn.swa_cache'] = marker(SWA)
    _, tuples = cpns['get_layer_tuples'](specs)
    slot, = [i for i,t in enumerate(tuples) if 'model.layers.39.self_attn.swa_cache' in t]
    assert slot==3 and 'model.layers.20.self_attn.long_kv_cache' in tuples[slot]
    column = [name for name in tuples[slot] if isinstance(specs[name],SWA) and '.mtp.' not in '.'+name]
    group = 2+column.index('model.layers.39.self_attn.swa_cache')
    assert group==11
    # Logical prefix counterexample: a legal hit/replay at boundary256 needs
    # original rows129..256. Global tail at original end3328 has no such values.
    replay_q = 256
    replay_window = list(range(replay_q-127, replay_q+1))
    tail_start = 3328-128
    assert max(replay_window) < tail_start

    original = read(OUT/'final-producer-entry.json')
    waves, hist, _, _, _ = records()
    bodies = cached(BODY)['profiles']
    windows = []
    # Preserve six original serial cache-producing primitives at their measured
    # durations. Ignore all required AG/wait/new-implementation costs: favorable
    # same-primitive sensitivity, NOT a hardware lower bound or universal cap.
    for w in original['C8_first_eager_price']['windows']:
        rank = w['rank']; p = bodies[rank]; b = p['bodies'][w['ordinal']]
        by_id = {r[0]: r for r in b['rows']}
        start = w['last_Attn_HcPre_start_ns']
        stop = by_id[b['HcPre_TASK_ROWS'][79]][1]
        rows = [r for r in b['rows'] if start <= r[1] < stop]
        def one(pred):
            result = [r for r in rows if pred(p['regimes'][r[8]])]
            assert len(result) == 1, (rank, w['ordinal'], [r[0] for r in result])
            return result[0]
        first = by_id[b['HcPre_TASK_ROWS'][78]]
        norm = one(lambda g: g[1] == 'RmsNorm' and g[2].strip('"').split(';')[1] == '5120')
        kv = one(lambda g: g[1] == 'MatMulV2' and g[2].strip('"').split(';')[1] == '512,5120')
        kvnorm = one(lambda g: g[1] == 'RmsNorm' and g[2].strip('"').split(';')[1] == '512')
        rope = one(lambda g: g[1] == 'InplacePartialRotaryMul' and ',1,1,512;' in g[2])
        scatter = one(lambda g: g[1] == 'ScatterNdUpdateSk')
        chosen = [first, norm, kv, kvnorm, rope, scatter]
        assert all(a[2] <= z[1] for a,z in zip(chosen, chosen[1:])), (rank,w['ordinal'])
        mandatory = sum(r[2]-r[1] for r in chosen)
        windows.append(dict(wave=w['wave'],rank=rank,ordinal=w['ordinal'],scheduled=w['scheduled'],
            original_parent_ms=w['original_parent_ms'],start_ns=start,head_start_ns=w['head_matmul_start_ns'],
            retained_serial_cache_tasks=[dict(task=r[0],type=p['regimes'][r[8]][1],
                input_shapes=p['regimes'][r[8]][2],duration_ns=r[2]-r[1]) for r in chosen],
            mandatory_same_primitive_duration_ms=mandatory/1e6,
            adjusted_free_end_ns=w['head_matmul_start_ns']-mandatory,
            native_head_rows=w['head_rows']))
    by = {(w['wave'],w['rank']):w for w in windows}
    offsets = hist['clock_contract']['offset_upper_ns']; dp = {}
    for wave in sorted(set(w['wave'] for w in windows)):
        for d in [0,1]:
            ix = list(range(d*8,d*8+8))
            if not all((wave,i) in by for i in ix):
                continue
            g = [by[wave,i] for i in ix]
            dp[wave,d] = latest_dp_change([r['adjusted_free_end_ns'] for r in g],
                [r['start_ns'] for r in g], [[offsets[i][j] for j in ix] for i in ix])
    upper = {w:max(v[1] for (wave,d),v in dp.items() if wave==w) for w in sorted({w for w,d in dp})}
    requests = original['C8_first_eager_price']['requests']
    base = max(r['TTFT_ms'] for r in requests)
    gross = [sum(v for w,v in upper.items() if w<=r['wave']) for r in requests]
    p90 = base-max(r['TTFT_ms']-v for r,v in zip(requests,gross))
    scenarios = []
    for rho in [0,.25,.5,.75]:
        for h in [.1,.5,1]:
            gains = [rho*v-h*sum(w<=r['wave'] for w in upper) for r,v in zip(requests,gross)]
            net = base-max(r['TTFT_ms']-v for r,v in zip(requests,gains))
            scenarios.append(dict(removable_exposed_fraction=rho,added_ms_per_wave=h,
                P90_signed_net_ms=net,P90_signed_net_percent=100*net/base))
    # Count-level qualification only. Scheduled dict insertion order is not
    # evidence of actual InputBatch/QSL row order after runner reordering.
    row_counts = []
    for w in windows:
        if w['rank'] % 8:
            continue
        counts = list(w['scheduled'].values())
        assert all(n==6 or n>=3200 for n in counts)
        wanted = sum(6 if n==6 else 1 for n in counts)
        assert wanted == w['native_head_rows']
        row_counts.append(dict(wave=w['wave'],DP=w['rank']//8,scheduled_total=sum(counts),
            prefill_requests=sum(n>=3200 for n in counts),all_live_verify_rows=sum(n for n in counts if n==6),
            selected_count=wanted,native_head_rows=w['native_head_rows'],
            exact_multi_request_row_order='NOT_DERIVED_FROM_SCHEDULER_DICT'))
    result = dict(status='PARTIAL_PHYSICAL_CACHE_AND_QUERY_ROW_REVIEW',whole_chain_complete=False,
        precision='EVERY_REPLACEMENT_PRIMITIVE_BITWISE_IDENTICAL',
        source_identity='final-cache-execution-identity.json',
        physical_chain=[
            dict(entry='cache_config.py:147-227,248-274',finding='Four layer-outermost physical slot tensors; body39 SWA belongs to slot3 with source20/other SWAs. SWA groups transpose slots and allocate distinct global block IDs. Sharing a backing tensor does not authorize simultaneous same-block alias writes.'),
            dict(entry='model_runner_v1.py:5389-5462;dsa_v41.py:227-242,297-308',finding='Per-layer views carry physical padded page stride; A3 scatter uses builder [block,row] coordinates and skips[-1,-1]. Original PA read uses the same group table; no modulo128 ring for target SWA.'),
            dict(entry='compute_slot_mapping.py:43-90;dsa_v41.py:974-1035',finding='Noncircular DCP1 mapping is table[request,position//block_size]*block_size+position%block_size, then split by storage_block_size. A nonnegative null block ID is not automatically a PAD skip.'),
            dict(entry='kv_cache_manager.py:525-546;single_type_kv_cache_manager.py:681-718,944-1061,1146-1182',finding='Retire full blocks from committed/processed prefix excluding in-flight speculation. Prefix-hit needs contiguous window-tail blocks plus eagle drop block; full chunks remain allocated while current queries run.'),
            dict(entry='single_type_kv_cache_manager.py:470-524,1063-1144;kv_cache_coordinator.py:335-360',finding='SWA cached block tails may serve later sibling replay at reachable prefix boundaries. Dense/sparse retention, extra retained tokens and CoW ownership cannot be inferred from window length alone.')],
        original_retention_integer_examples=examples,
        original_allocator_marker_replay=dict(body39_slot=slot,body39_framework_group=group,
            aliased_slot_names=list(tuples[slot]),
            qualification='Executed original topology helper with inert marker specs. No Torch import; dummy capacities discarded. Group11 follows original transpose after full/state groups; not a live data_ptr/block-ID dump.'),
        tail_only_counterexample=dict(original_length=3328,naive_kept_rows=[tail_start,3327],
            later_query_position=replay_q,needed_window=[replay_window[0],replay_window[-1]],
            existing_cache_needed=[replay_window[0],replay_q-1],shared_prefix_at_least=384,
            possible_eagle_peek_blocks=[1,2],later_private_query_row=replay_q,
            qualification='Legal logical prefix replay counterexample; not an observed C8 hit. Tail-only writer with unchanged cache publication is stopped. Effective prefix setting is not needed to retain the safe full-writer alternative.'),
        selected_row_contract=dict(
            equation='For original global query row g: old CP owner=floor(g/ceil(N/8)); old local row=g-owner*ceil(N/8). Gather topK36 by that old local row. Restore selected global IDs explicitly through TP head exchange and SP return.',
            single_request_3328=dict(global_selected_row=3327,CP_owner=7,old_CP_local_topK_row=415,
                wrong_new_prefix_row=0,all_eight_TP_participants_required=True),
            contiguous_suffix='For each original request/CP intersection, selecting its tail k and keeping authoritative device seq_len preserves logical causal position L-k+j. Arbitrary holes need explicit positions/custom metadata or dense fallback. Logical set equality does not prove SMLA primitive bits.',
            numerical_geometry='Keep old M/tile indices, KV reduction/softmax/P rounding, GEMM and HC reduction trees, EP expert order and original collective reduction arithmetic for every retained output. Masking tiles is a design direction; no native implementation or admission.',
            cases=row_counts,
            outstanding=['Actual multi-request InputBatch/QSL and post-reject seq_len mapping rather than scheduler dict order',
                'SMLA metadata/task partition and selected suffix can change K-panel/partial softmax order; prove bit-preserving old geometry',
                'Zero-query ranks must retain full cache writes and participate in TP exchange; retain original SP residual/pre_mix ownership',
                'Original FFN expert grouping/tiling changes under compaction; previous failed compact recipe stays stopped']),
        C8_retained_cache_sensitivity=dict(windows=windows,DP_wave_bounds_ms={f'{w}:{d}':v for (w,d),v in dp.items()},
            favorable_wave_bounds_ms={str(w):v for w,v in upper.items()},
            P90_optimistic_saved_ms=p90,P90_optimistic_percent=100*p90/base,paid_scenarios=scenarios,
            qualification='56 matching original eager windows, retaining measured six serial cache primitives. All gather, waits, surviving verification rows, selected compute, communication and replacement costs still relaxed. All real eager waves includes decode-only and mixed waves; no credit for deleting live verify rows. Same-primitive original-schedule sensitivity, not a hardware bound, actual gain, full alternative limit or additive terminal budget.'),
        candidates=[dict(name='Tail128-only cache writes with unchanged prefix publication',state='已停止',reason='Missing reachable cache rows under chunk/replay/rollback ownership; no modulo ring guarantee'),
                    dict(name='Selected rows with topK new prefix and unchanged CP restore',state='已停止',reason='Old row415 becomes new0 in the unambiguous3328 single-request case; collective return also loses global IDs'),
                    dict(name='Full original cache producer plus query/O/FFN demand scheduling',state='待确认',reason='Worth deeper implementation judgment; concrete row maps and old numerical geometry/native/net costs still required')],
        next_action='Continue remaining source-derived input/embedding/host-context deliveries; retain last39 full-cache selected-query candidate with explicit geometry/row-map/net gates. No tail-only cache patch or retry of failed compact FFN.',
        inputs=[dict(path=str(p),sha256=sha(p)) for p in [OUT/'final-cache-execution-identity.json',OUT/'final-producer-entry.json',BODY,Path(__file__)]],
        new_NPU_work=0,new_profile_collection=0,service_requests=0,remote_writes=[])
    (OUT/'final-cache-frontier.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(windows=len(windows),row_count_cases=len(row_counts),
        P90_same_cache_optimistic_ms=p90,P90_same_cache_optimistic_percent=100*p90/base,
        middle_scenario=next(s for s in scenarios if s['removable_exposed_fraction']==.5 and s['added_ms_per_wave']==.5),
        admitted_native=0,new_NPU_work=0)))

if __name__ == '__main__':
    main()
