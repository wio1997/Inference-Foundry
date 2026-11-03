"""Join retained summaries by raw file hash; no cross-capture timestamp joins."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import statistics
import hashlib
import json

R = Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
OUT = Path(__file__).parent / 'profile-first-20261031'


def read(rel):
    return json.loads((R/rel).read_text())


def dist(values):
    return {'n':len(values), 'min':min(values), 'median':statistics.median(values), 'max':max(values)} if values else None


def main():
    paths = ['publication-draft/official-operators.json','continuation-20261030/graph-executions.json',
             'continuation-20261030/iteration-gaps.json','continuation-20261030/prepare-input-analysis.json',
             'continuation-20261030/worker-validation.json','agents/fusion/qr-quant-window-coverage.json']
    operators, graphs, gaps, prepare, worker, qr = (read(p) for p in paths)
    graph_by_hash = {x['kernel_sha256']:x for x in graphs['rows']}
    graph_by_dir = {x['source_dir']:x for x in graphs['rows']}
    assert len(graph_by_hash) == len(graph_by_dir) == len(graphs['rows']) == 32
    bindings = []
    for sample in operators['samples']:
        digest = sample['sha256']['kernel_details.csv']
        graph = graph_by_hash[digest]
        assert sample['window_ordinal'] == graph['window']
        bindings.append({'sample_id':sample['sample_id'], 'window':graph['window'], 'rank':graph['rank'],
                         'kernel_sha256':digest, 'source_dir':graph['source_dir']})
    assert len({(x['rank'],x['window']) for x in bindings}) == len(bindings) == 32
    window_counts = Counter(x['window'] for x in bindings)
    assert window_counts == {1:16,2:16}
    main_counts = Counter()
    for row in graphs['rows']:
        main_counts[(row['rank'],row['window'])] = sum(
            bool(ex['main_shape_compatible_40_moe']) for stream in row['model_streams'] for ex in stream['executions'])
    gap_rows = []
    for row in gaps['rows']:
        graph = graph_by_hash[row['kernel_sha256']]
        assert graph['rank'] == row['rank'] and graph['window'] == row['window'] and graph['source_dir'] == row['source_dir']
        for gap in row['gaps']:
            assert 0 <= gap['all_record_union_ns'] <= gap['span_ns']
            gap_rows.append({'rank':row['rank'], 'window':row['window'], 'ordinal':gap['ordinal'],
                             'span_ns':gap['span_ns'], 'all_record_union_ns':gap['all_record_union_ns'],
                             'record_union_fraction':gap['all_record_union_ns']/gap['span_ns'],
                             'type_union_ns':gap['type_union_ns']})
    types = sorted(set(k for x in gap_rows for k in x['type_union_ns']))
    gap_types = [{'type':t, 'gaps_with_positive_exported_interval':sum(x['type_union_ns'].get(t,0)>0 for x in gap_rows),
                  'per_gap_union_ns':dist([x['type_union_ns'].get(t,0) for x in gap_rows]),
                  'rank_count_with_positive_interval':len({x['rank'] for x in gap_rows if x['type_union_ns'].get(t,0)>0})} for t in types]
    prepare_rows = []
    for row in prepare['rows']:
        graph = graph_by_dir[row['source_dir']]
        assert graph['rank'] == row['rank']
        scopes = row['prepare_input_scopes']
        for scope in scopes:
            assert scope['root_exclusive_recorded_scope_ns'] + scope['descendant_union_ns'] == scope['duration_ns']
        prepare_rows.append({'rank':row['rank'], 'window':graph['window'], 'scopes':len(scopes),
                             'duration_ns':dist([s['duration_ns'] for s in scopes]),
                             'recorded_exclusive_ns':dist([s['root_exclusive_recorded_scope_ns'] for s in scopes]),
                             'containment_violations':sum(s['containment_violations'] for s in scopes),
                             'interpretation':'Exclusive of nested recorded scopes, not measured CPU active work or removable host time.'})
    common = {'generated_utc':datetime.now(timezone.utc).isoformat(), 'overall_complete':False,
              'whole_modules_closed':0,'new_target_source_read':False,'new_device_work':False}
    result = {**common, 'capture':operators['run_id'], 'sample_rank_window_bindings':bindings,
              'binding_rule':'Exact unique retained raw kernel CSV SHA-256 and matching recorded window. Not scheduler/request/tensor identity.',
              'shape_compatible_main_segment_counts':[{'rank':r,'window':w,'count':c} for (r,w),c in sorted(main_counts.items())],
              'main_segment_caution':'Recurring graph signatures and40MoE are role-compatible segments, not independent proof of complete iterations.',
              'gap_scope':{'rank_count':len({x['rank'] for x in gap_rows}), 'gap_count':len(gap_rows),
                           'span_ns':dist([x['span_ns'] for x in gap_rows]),
                           'exported_record_union_ns':dist([x['all_record_union_ns'] for x in gap_rows]),
                           'union_fraction':dist([x['record_union_fraction'] for x in gap_rows])},
              'gap_types':gap_types,'worker_prepare_scopes':prepare_rows,
              'separate_worker_sampler':{'run_id':worker['run_id'], 'stack_files_present':worker['stack_files_present'],
                                        'sampler_exit_codes':{k:v.get('exit_code') for k,v in worker['samplers'].items()},
                                        'interpretation':'No Python worker stack artifact in this input; process counters are separate-run evidence, not a replacement.'},
              'qr_selected_summary':{k:qr[k] for k in ['scope','special8_layers','window_shape_count_distributions','special8_filtered_sequences','limitations']},
              'provenance':[{'path':str(R/p),'sha256':hashlib.sha256((R/p).read_bytes()).hexdigest()} for p in paths],
              'limits':['All joins use retained summary provenance, not a fresh raw export parse.',
                        'An inter-main-segment interval contains many exported operations and is not a measurement of idle time.',
                        'Types and streams can overlap; per-type unions must not be added for an end-to-end gain.',
                        'No source semantics, tensor dependency, valid row count, cross-rank clock or critical-path gain is proved by this join.']}
    (OUT/'cohort-and-gap-bindings.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    ledger=json.loads((OUT/'candidate-profile-ledger.json').read_text())
    for c in ledger['candidates']:
        if c['id'] in ['F05','F09']:
            c['profile_review_state']='SELECTED_PROFILE_SUMMARY_REVIEWED_IDENTITY_UNRESOLVED'
            c['scoped_detail']={'summary':'qr-quant-window-coverage.json','windows':1920,
                               'D_Q_D_I_windows':384,'D_Q_windows':1536,'two_inputs_equal_proved':False,
                               'meaning':'Repeated matching quantization signatures support a reuse candidate; they do not establish equal input snapshots or parameters.'}
            c['next_profile_or_contract_action']='复用已存EP3双量化host-flow关联与局部接口合同；源码共同qr变量不能补成历史tensor值相等，保留输入版本和实际参数缺口。'
            c['target_source_gate']='REUSE_EXISTING_TARGETED_QUANT_CONTRACT; runtime value/alias gap remains'
        if c['id'] in ['F30','F31']:
            wanted=['rejection_greedy_sample_triton','ArgMaxV2'] if c['id']=='F30' else ['prepare_inputs_padded_kernel','copy_and_expand_dflash_and_dspark_inputs_kernel','_build_dspark_swa_indices_kernel','SparseFlashMlaMetadata']
            c['profile_review_state']='BOUND_MAIN_SEGMENT_GAP_TYPES_REVIEWED_ROLE_PARTIAL'
            c['scoped_detail']={'source':'cohort-and-gap-bindings.json','gap_count':len(gap_rows),
                               'types':[t for t in gap_types if t['type'] in wanted],
                               'same_call_value_or_full_branch_proof':False}
            c['next_profile_or_contract_action']='按已绑定rank/window检查对应提交栈和copy/metadata消费者；显式greedy kernel只是选定区间观察，不作全部概率分支排除。'
    ledger.update(common)
    (OUT/'candidate-profile-ledger-v2.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n')
    modules=json.loads((OUT/'module-profile-ledger.json').read_text())
    for m in modules['modules']:
        if m['id'] in ['M03','M04','M16','M17','M18','M19','M20','M21','M22','M24','M25']:
            m['profile_entry_points'].append('cohort-and-gap-bindings.json')
            m['review_state']='PARTIAL_BOUND_RANK_WINDOW_GAP_OR_HOST_SCOPE_EVIDENCE; no full module closure'
    modules.update(common)
    (OUT/'module-profile-ledger-v2.json').write_text(json.dumps(modules,ensure_ascii=False,indent=2)+'\n')
    summary={**common,'official_samples_bound':len(bindings),'rank_count':16,'window_counts':dict(window_counts),
             'gap_scope':result['gap_scope'],'prepare_scope_groups':len(prepare_rows),
             'candidate_review_states':dict(Counter(c['profile_review_state'] for c in ledger['candidates'])),
             'greedy_rejection_gap_evidence':[t for t in gap_types if t['type']=='rejection_greedy_sample_triton'],
             'next_action':'按v2台账继续21项仅family入口的scope/stack核查；并复用已存精度/alias合同，实际历史身份缺口继续单列。'}
    (OUT/'binding-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    main()
