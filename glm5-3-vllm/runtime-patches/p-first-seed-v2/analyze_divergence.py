#!/usr/bin/env python3
"""Compare the bounded actual-target snapshots, without loading model weights."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshots',type=Path,required=True)
    parser.add_argument('--client',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists()
    records=[json.loads(line) for path in args.snapshots.glob('*.jsonl')
             for line in path.read_text().splitlines() if line.strip()]
    snapshots=[row for row in records if row.get('kind')=='target_tensor_snapshot']
    errors=[row for row in records if row.get('kind') in ('probe_capture_refused','probe_dump_error','shared_overlap_refused')]
    policies=[row for row in records if row.get('kind')=='shared_overlap_actual_calls']
    decisions=[row for row in records if row.get('kind')=='unused_draft_decision']
    skips=[row for row in records if row.get('kind')=='native_skip_drafting_executed']
    clients=json.loads(args.client.read_text())
    assert clients['completed']
    cache={}

    def tensor(info):
        name=info['path']
        assert Path(name).name==name
        if name not in cache:
            path=args.snapshots/name
            assert path.is_file() and not path.is_symlink()
            raw=path.read_bytes()
            assert len(raw)==info['bytes'] and hashlib.sha256(raw).hexdigest()==info['sha256']
            dtype=info['dtype'].removeprefix('torch.')
            if dtype=='bfloat16':
                data=(np.frombuffer(raw,dtype='<u2').astype('<u4')<<16).view('<f4')
            else:
                dtype={'int64':'<i8','int32':'<i4','float32':'<f4','float16':'<f2'}[dtype]
                data=np.frombuffer(raw,dtype=dtype)
            cache[name]=data.reshape(info['shape'])
        return cache[name]

    rows=[]
    vectors={}
    for client in clients['rows']:
        if not client['focus']:continue
        public=client['response']['id']
        matched=[row for row in snapshots if row['request_id'].startswith(public+'-0-')]
        ranks={row['tp_rank']:row for row in matched}
        row={'case_id':client['case_id'],'cohort':client['cohort'],
             'phase':client.get('phase'),
             'diagnostic_mode':matched[0].get('diagnostic_mode','native') if matched else client.get('diagnostic_mode','unknown'),
             'history_sha256':client['history_sha256'],'public_request_id':public,
             'API_token_id':client['response']['choices'][0]['token_ids'][0],
             'ranks':sorted(ranks),'all16':len(ranks)==len(matched)==16}
        rows.append(row)
        if not row['all16']:continue
        hidden=[]
        policy_calls=[]
        for rank in range(16):
            snapshot=ranks[rank]
            ids=tensor(snapshot['input_snapshots']['input_ids'])
            positions=tensor(snapshot['input_snapshots']['positions'])
            start,end=snapshot['query_start_loc'][snapshot['batch_index']:snapshot['batch_index']+2]
            assert snapshot['computed_before']==0 and snapshot['scheduled_tokens']==len(client['request']['prompt'])
            assert ids[start:end].tolist()==client['request']['prompt']==snapshot['prompt_ids']
            assert positions[start:end].tolist()==list(range(len(client['request']['prompt'])))
            assert snapshot['history_sha256']==client['history_sha256']
            assert snapshot['diagnostic_copy_event_ready'] and snapshot['native_event_ready']
            assert snapshot['same_constructor_stream']
            if snapshot.get('diagnostic_mode') in ('shared-on','shared-off'):
                expected=snapshot['diagnostic_mode']=='shared-on'
                assert snapshot.get('actual_shared_overlap_enabled') is expected
                assert client['diagnostic_mode']==snapshot['diagnostic_mode']
                selected=[policy for policy in policies if policy['tp_rank']==rank and snapshot['request_id'] in policy['request_ids']]
                assert len(selected)==1 and selected[0]['enabled'] is expected
                counts=selected[0]['calls'];key='target_on' if expected else 'target_off'
                assert counts.get(key,0)>0 and counts.get('target_off' if expected else 'target_on',0)==0
                policy_calls.append(counts)
            if snapshot.get('diagnostic_mode') in ('draft-native','draft-skip'):
                expected=snapshot['diagnostic_mode']=='draft-skip'
                assert snapshot.get('actual_shared_overlap_enabled') is True
                assert client['diagnostic_mode']==snapshot['diagnostic_mode']
                chosen=[decision for decision in decisions if decision['tp_rank']==rank and snapshot['request_id'] in decision['request_ids']]
                native_skips=[skip for skip in skips if skip['tp_rank']==rank and snapshot['request_id'] in skip['request_ids']]
                assert len(chosen)==1 and chosen[0]['native_fits'] and chosen[0]['eligible_all_max1_D_TP16_DP1_PP1']
                assert chosen[0]['selected_skip'] is expected
                assert len(native_skips)==(1 if expected else 0)
                selected=[policy for policy in policies if policy['tp_rank']==rank and snapshot['request_id'] in policy['request_ids']]
                assert len(selected)==1 and selected[0]['enabled'] is True
                counts=selected[0]['calls'];assert counts.get('target_on')==75 and not counts.get('target_off')
                assert counts.get('after_target_on',0)==(0 if expected else 5)
                policy_calls.append(counts)
            output=snapshot['native_output']
            index=output['req_ids'].index(snapshot['request_id'])
            assert output['sampled_token_ids'][index]==[row['API_token_id']]
            hidden.append(tensor(snapshot['hidden']))
        rank0=ranks[0]
        start,end=rank0['query_start_loc'][rank0['batch_index']:rank0['batch_index']+2]
        logits=tensor(rank0['logits'])
        assert logits.ndim==1 and logits.size==rank0['global_vocab_size']
        argmax=np.flatnonzero(logits==np.max(logits)).tolist()
        assert row['API_token_id'] in argmax
        vectors[public]={'hidden':hidden,'logits':logits}
        alternatives=(22,99419) if len(client['request']['prompt'])==258 else (98886,16)
        row.update(actual_query_ids_positions_all16_match=True,
                   original_output_event_and_copy_ready_all16=True,
                   native_sample_equals_API_all16=True,
                   actual_shared_expert_calls_all16=policy_calls,
                   actual_batch_size=len(rank0['batch_req_ids']),
                   actual_batch_req_ids=rank0['batch_req_ids'],
                   actual_input_shape=rank0['input_snapshots']['input_ids']['shape'],
                   batch_query_lengths=np.diff(rank0['query_start_loc']).tolist(),
                   batch_index=rank0['batch_index'],
                   input_file=rank0['input_snapshots']['input_ids']['path'],
                   query_range=[start,end],
                   hidden_dtype=rank0['hidden']['dtype'],hidden_shape=rank0['hidden']['shape'],
                   logits_dtype=rank0['logits']['dtype'],logits_shape=rank0['logits']['shape'],
                   hidden_distinct_rank_hashes=len({ranks[i]['hidden']['sha256'] for i in range(16)}),
                   logits_distinct_rank_hashes=len({ranks[i]['logits']['sha256'] for i in range(16)}),
                   lmhead=rank0['lmhead'],raw_argmax_ids=argmax,
                   SFA_attention=rank0.get('attention'),
                   candidate_logits={str(token):float(logits[token]) for token in alternatives},
                   candidate_margin=float(logits[alternatives[0]])-float(logits[alternatives[1]]))

    grouped=defaultdict(list)
    for row in rows:grouped[row['case_id']+' / '+row['diagnostic_mode']].append(row)
    summary={}
    for case,group in grouped.items():
        valid=[row for row in group if row['all16']]
        baseline=next((row for row in valid if row['cohort'].startswith('C1')),None)
        if baseline is None:continue
        b=vectors[baseline['public_request_id']]
        for row in valid:
            current=vectors[row['public_request_id']]
            hidden_equal=[np.array_equal(x,y) for x,y in zip(b['hidden'],current['hidden'])]
            differences=[x.astype(np.float32)-y.astype(np.float32) for x,y in zip(b['hidden'],current['hidden'])]
            ld=b['logits'].astype(np.float32)-current['logits'].astype(np.float32)
            row.update(hidden_equal_to_first_C1_all16=all(hidden_equal),
                       hidden_equal_ranks_to_first_C1=[i for i,equal in enumerate(hidden_equal) if equal],
                       hidden_max_abs_difference_to_first_C1=max(float(np.max(np.abs(x))) for x in differences),
                       hidden_mean_abs_difference_to_first_C1=float(np.mean([np.mean(np.abs(x)) for x in differences])),
                       logits_equal_to_first_C1=np.array_equal(b['logits'],current['logits']),
                       logits_max_abs_difference_to_first_C1=float(np.max(np.abs(ld))),
                       logits_mean_abs_difference_to_first_C1=float(np.mean(np.abs(ld))))
        summary[case]={'captured':len(valid),'missing':len(group)-len(valid),
                       'C1_count':sum(row['cohort'].startswith('C1') for row in valid),
                       'C4_focus_count':sum(row['cohort'].startswith('C4') for row in valid),
                       'choices':dict(Counter(str(row['API_token_id']) for row in valid)),
                       'actual_batch_sizes':dict(Counter(str(row['actual_batch_size']) for row in valid)),
                       'C1_hidden_equal_all16':all(row['hidden_equal_to_first_C1_all16'] for row in valid if row['cohort'].startswith('C1')),
                       'C1_logits_equal':all(row['logits_equal_to_first_C1'] for row in valid if row['cohort'].startswith('C1')),
                       'C4_hidden_changed_requests':sum(not row['hidden_equal_to_first_C1_all16'] for row in valid if row['cohort'].startswith('C4')),
                       'C4_logits_changed_requests':sum(not row['logits_equal_to_first_C1'] for row in valid if row['cohort'].startswith('C4'))}
    c1_pairs=[]
    for after in rows:
        if not after['all16'] or after['cohort']!='C1-after':continue
        before=next(row for row in rows if row['all16'] and row['case_id']==after['case_id']
                    and row['diagnostic_mode']==after['diagnostic_mode']
                    and row['phase']==after['phase'] and row['cohort']=='C1-before')
        b=vectors[before['public_request_id']];a=vectors[after['public_request_id']]
        c1_pairs.append({'case_id':after['case_id'],'mode':after['diagnostic_mode'],'phase':after['phase'],
                         'actual_input_shapes':[before['actual_input_shape'],after['actual_input_shape']],
                         'actual_batch_sizes':[before['actual_batch_size'],after['actual_batch_size']],
                         'hidden_equal_all16':all(np.array_equal(x,y) for x,y in zip(a['hidden'],b['hidden'])),
                         'logits_equal':np.array_equal(a['logits'],b['logits']),
                         'before_API_token':before['API_token_id'],'after_API_token':after['API_token_id']})
    cross_modes=[]
    for case in {row['case_id'] for row in rows}:
        on=next((row for row in rows if row['all16'] and row['case_id']==case and row['diagnostic_mode']=='shared-on' and row['cohort']=='C1-before'),None)
        off=next((row for row in rows if row['all16'] and row['case_id']==case and row['diagnostic_mode']=='shared-off' and row['cohort']=='C1-before'),None)
        if on is None or off is None:continue
        a=vectors[on['public_request_id']];b=vectors[off['public_request_id']]
        cross_modes.append({'case_id':case,'comparison':'First C1-before in each mode; not isolated stream-order proof',
                            'hidden_equal_all16':all(np.array_equal(x,y) for x,y in zip(a['hidden'],b['hidden'])),
                            'logits_equal':np.array_equal(a['logits'],b['logits']),
                            'hidden_max_abs_difference':max(float(np.max(np.abs(x.astype(np.float32)-y.astype(np.float32)))) for x,y in zip(a['hidden'],b['hidden'])),
                            'on_API_token':on['API_token_id'],'off_API_token':off['API_token_id']})
    report={'schema':'glm53-target-boundary-comparison/v1','focus_requests':len(rows),
            'complete_all16_requests':sum(row['all16'] for row in rows),
            'snapshot_records':len(snapshots),'probe_errors':errors,
            'cases':summary,'C1_before_after_pairs':c1_pairs,'first_C1_cross_mode':cross_modes,'rows':rows,
            'scope':'Actual final input/positions, final sample hidden and full gathered logits at two cold native full-prefill histories. Differences locate the first observed boundary only; no full KV/intermediate equality, harmless-rounding conclusion, quality/performance acceptance, or proof about the original speculative decode state.'}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ['focus_requests','complete_all16_requests','snapshot_records','cases']}))


if __name__=='__main__':main()
