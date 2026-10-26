#!/usr/bin/env python3
"""Fail-closed per-arm admission of full48 light stage packets; diagnostic only."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT=Path('/data/wio/Inference_Foundry')
DATASET=Path('/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl')
LABELS=('cycle_begin','target_before','target_after','proposer_before','proposer_after')


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def need(ok,why):
    if not ok: raise AssertionError(why)


def client_text_hash(path):
    row=read(path)
    parts=[]
    for event in row['events']:
        raw=base64.b64decode(event['payload_b64'],validate=True)
        if raw==b'[DONE]':continue
        data=json.loads(raw)
        for choice in data.get('choices',()):
            delta=choice.get('delta') or {}
            parts.extend(str(delta[key]) for key in ('reasoning_content','reasoning','content')
                         if delta.get(key))
    return hashlib.sha256(''.join(parts).encode()).hexdigest()


def arm(path:Path,name:str,on:bool):
    admission=read(path/'client_admission.json')
    need(admission['status']=='client_two_phase_admitted',f'{name}: client admission')
    recheck=path/'client_admission_recheck.json'
    subprocess.run([sys.executable,
                    str(ROOT/'scripts/loop080_frontier_phase_validate.py'),
                    '--dataset',str(DATASET),
                    '--warmup-dir',str(path/'warmup_client'),
                    '--measured-dir',str(path/'measured_client'),
                    '--output',str(recheck)],
                   check=True,capture_output=True,text=True)
    need(read(recheck)==admission,f'{name}: independent client raw/SSE recheck')
    measured=admission['measured_summary']
    need(measured['n']==48 and measured['success']==48 and measured['fail']==0 and
         measured['concurrency']==12 and measured['max_tokens']==1024,
         f'{name}: frozen client contract')
    client_rows=sorted((path/'measured_client').glob('request_*.json'))
    need(len(client_rows)==48 and all(p.name==f'request_{i:03d}.json'
                                      for i,p in enumerate(client_rows)),
         f'{name}: exact measured raw')
    by_response={}
    for index,p in enumerate(client_rows):
        row=read(p)['row']
        need(row['phase']=='measured' and row['i']==index and
             row['http_status']==200 and row['error'] is None and
             row['output_tokens']==1024, f'{name}: client output {index}')
        need(row['response_id'] not in by_response,f'{name}: duplicate client response')
        by_response[row['response_id']]=index
    need(len(by_response)==48,f'{name}: client response cardinality')
    expected={f'rank{r}_cohort{c}.json' for r in range(8) for c in range(5,9)}
    need({p.name for p in (path/'ledger').iterdir()}==expected|{'arm'},
         f'{name}: exact ledger')
    need({p.name for p in (path/'packet').iterdir()}==(expected if on else set()),
         f'{name}: exact packet set')
    all_runtime={f'rank{r}_cohort{c}.json' for r in range(8) for c in range(1,9)}
    need({p.name for p in (path/'runtime').iterdir()}==all_runtime,
         f'{name}: warm+measured Runtime')
    need((path/'server_post_count.txt').read_text().strip()=='96' and
         (path/'server_post_count_after_stop.txt').read_text().strip()=='96',
         f'{name}: exact server POSTs')
    source_hashes={str(p):sha(p) for p in (
        path/'client_admission.json',recheck,path/'warmup_client_admission.json')}
    for section in ('warmup_client','measured_client'):
        files=list((path/section).iterdir())
        want={f'request_{i:03d}.json' for i in range(48)}|{'summary.json'}
        need({p.name for p in files}==want,f'{name}: exact {section} source set')
        for p in files:source_hashes[str(p)]=sha(p)
    for section in ('ledger','packet','runtime'):
        for p in (path/section).glob('*.json'):source_hashes[str(p)]=sha(p)
    cycles_by_cohort={}
    effective_by_cohort={}
    request_indexes=[]
    all_ids=set()
    all_stage=[]
    all_cycle=[]
    stage_by_class={}
    host_export_bound=[]
    server_ns=set()
    for rank in range(8):
        generation=[1]*5126
        previous_parked=0
        for cohort in range(5,9):
            filename=f'rank{rank}_cohort{cohort}.json'
            ledger=read(path/'ledger'/filename)
            rt=read(path/'runtime'/filename)
            need(ledger['rank']==rank and ledger['cohort']==cohort and
                 ledger['run_ts']==f'LOOP081-RUN656-{name.upper()}',
                 f'{name}: ledger run/rank/cohort')
            need(rt['rank']==rank and rt['cohort']==cohort and rt['pass'] is True and
                 rt['target_graph_requested'] is True and rt['target_graph_mode']=='FULL',
                 f'{name}: actual Runtime FULL/correct')
            need(ledger['target_graph_requested'] is True and
                 ledger['target_graph_mode']=='FULL' and
                 ledger['req_ids']==rt['req_ids'] and
                 ledger['cycles']==rt['cycles'] and
                 ledger['generated_output_counts']==rt['generated_output_counts']==[1024]*12,
                 f'{name}: ledger/Runtime identity')
            C=ledger['cycles']
            need(type(C) is int and 1<=C<=1025 and
                 type(ledger['runtime_scoped_wall_s']) in (int,float) and
                 math.isfinite(ledger['runtime_scoped_wall_s']) and
                 ledger['runtime_scoped_wall_s']>0 and
                 ledger['runtime_scoped_wall_s']==rt['wall_seconds'],
                 f'{name}: Runtime wall/cycles')
            counts=ledger['accepted_counts']
            need(len(counts)==C and all(len(row)==12 and
                 all(type(v) is int and 0<=v<=8 for v in row) for row in counts),
                 f'{name}: accepted count history')
            for key in ('canonical_effective_staged_trajectory_sha256',
                        'count_history_sha256','raw_padded_token_history_sha256',
                        'runtime_bulk_output_sha256'):
                value=ledger[key]
                need(isinstance(value,str) and len(value)==64 and
                     all(ch in '0123456789abcdef' for ch in value),
                     f'{name}: {key} digest')
            server_ns.add(ledger['time_namespace'])
            identity=(ledger['req_ids'],C,counts,
                      ledger['canonical_effective_staged_trajectory_sha256'],
                      ledger['count_history_sha256'],
                      ledger['runtime_bulk_output_sha256'])
            if rank==0:
                cycles_by_cohort[cohort]=C
                effective_by_cohort[cohort]=ledger['canonical_effective_staged_trajectory_sha256']
                for req_id in ledger['req_ids']:
                    matches=[response for response in by_response
                             if req_id.startswith(response+'-')]
                    need(len(matches)==1,f'{name}: client/server request join')
                    response=matches[0]
                    need(response not in all_ids,f'{name}: duplicate joined request')
                    all_ids.add(response)
                    request_indexes.append(by_response[response])
            else:
                ref=read(path/'ledger'/f'rank0_cohort{cohort}.json')
                need(identity==(ref['req_ids'],ref['cycles'],ref['accepted_counts'],
                                ref['canonical_effective_staged_trajectory_sha256'],
                                ref['count_history_sha256'],
                                ref['runtime_bulk_output_sha256']),
                     f'{name}: all8 trajectory/output divergence')
            if on:
                packet=read(path/'packet'/filename)
                need(packet['status']=='instrumented_current_stream_full_cohort' and
                     packet['identity']==ledger and
                     packet['event_capacity']==5126 and
                     packet['event_used']==5*C+1,
                     f'{name}: packet identity/coverage')
                need(packet['pre_product_warm_record_end_ns'] <
                     measured['wall_start_monotonic_ns'],
                     f'{name}: Event pool not prewarmed before measured Product')
                need(packet['stream_key'][0].startswith('npu:') and
                     type(packet['stream_key'][1]) is int,
                     f'{name}: stream identity')
                rows=packet['events']; classes=packet['class_rows']
                need(len(rows)==5*C+1 and len(classes)==C,
                     f'{name}: event/class count')
                last_host=-1;last_event=-1.0
                for i,event in enumerate(rows):
                    want_cycle=i//5 if i<5*C else C-1
                    want_label=LABELS[i%5] if i<5*C else 'serve_terminal'
                    event_index=i if i<5*C else 5125
                    generation[event_index]+=1
                    need(event['cycle']==want_cycle and event['label']==want_label and
                         event['generation']==generation[event_index] and
                         type(event['host_submit_ns']) is int and
                         event['host_submit_ns']>=last_host and
                         isinstance(event['elapsed_ms'],(int,float)) and
                         math.isfinite(event['elapsed_ms']) and
                         event['elapsed_ms']>=last_event,
                         f'{name}: event order/generation/time')
                    last_host=event['host_submit_ns'];last_event=event['elapsed_ms']
                for i,row in enumerate(classes):
                    need(row['cycle']==i and
                         0<=row['parked_before']<=row['parked_after']<=12,
                         f'{name}: park class')
                    if i:need(row['parked_before']==classes[i-1]['parked_after'],
                              f'{name}: park continuity')
                need(classes[0]['parked_before']==0 and
                     classes[-1]['parked_after']==12,
                     f'{name}: park start/end')
                for i in range(C):
                    e=rows[5*i:5*i+5]
                    target=e[2]['elapsed_ms']-e[1]['elapsed_ms']
                    draft=e[4]['elapsed_ms']-e[3]['elapsed_ms']
                    end=rows[5*(i+1)]['elapsed_ms'] if i+1<C else rows[-1]['elapsed_ms']
                    cycle=end-e[0]['elapsed_ms']
                    need(target>=0 and draft>=0 and cycle>=target+draft-1e-6,
                         f'{name}: stage/cycle interval')
                    all_stage.append((target,draft))
                    all_cycle.append(cycle)
                    key=(cohort,classes[i]['parked_before'],classes[i]['parked_after'])
                    stage_by_class.setdefault(key,[]).append((target,draft,cycle))
                host_export_bound.append(packet['pre_product_setup_host_ns']/1e6)
            previous_parked=0
    need(len(all_ids)==48 and len(server_ns)==1 and
         next(iter(server_ns))==measured['clock']['time_namespace'],
         f'{name}: all Product identities/time namespace')
    need(sorted(request_indexes)==list(range(48)),f'{name}: complete 48 join')
    def quant(rows,index):
        v=sorted(x[index] for x in rows)
        return {'min':v[0],'median':statistics.median(v),
                'p95':v[min(len(v)-1,math.ceil(.95*len(v))-1)],'max':v[-1]}
    summary={'name':name,'status':'per_arm_admitted',
             'scope':'instrumented_diagnostic_only_not_formal_TPS_or_bound',
             'client_wall_s':measured['duration_s'],
             'client_output_tps_diagnostic_only':measured['output_tps_diagnostic_only'],
             'cycles_by_cohort':cycles_by_cohort,
             'effective_by_cohort':effective_by_cohort,
             'total_cycles':sum(cycles_by_cohort.values()),
             'joined_requests':len(all_ids),
             'request_indexes_by_cohort':request_indexes,
             'client_text_sha256_by_index':[client_text_hash(p) for p in client_rows],
             'raw_sha256':source_hashes}
    if on:
        summary['stage_rank_cycle_count']=len(all_stage)
        summary['target_ms']=quant(all_stage,0)
        summary['proposer_ms']=quant(all_stage,1)
        summary['cycle_ms']=quant([(x,) for x in all_cycle],0)
        summary['class_count']=len(stage_by_class)
        summary['pre_product_setup_host_ms']=quant([(x,) for x in host_export_bound],0)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    arms={n:arm(a.root/n,n,n=='on') for n in ('off_a','on','off_b')}
    cycles={n:tuple(v['cycles_by_cohort'].values()) for n,v in arms.items()}
    output={n:v['client_text_sha256_by_index'] for n,v in arms.items()}
    effective={n:tuple(v['effective_by_cohort'].values()) for n,v in arms.items()}
    indexes={n:tuple(v['request_indexes_by_cohort']) for n,v in arms.items()}
    same_work=(len(set(cycles.values()))==1 and
               len(set(effective.values()))==1 and
               len(set(indexes.values()))==1 and
               len(set(tuple(x) for x in output.values()))==1)
    result={'status':'all_arms_individually_admitted',
            'cross_arm_fixed_w0':same_work,
            'cross_arm_timing_transfer_admitted':False,
            'observer_effect_quantified':False,
            'arms':arms,
            'limits':'Per-arm current-stream stage intervals include queue/wait; OFF/ON walls cannot be subtracted if trajectory/output mismatch. No ordinary preparation, all-stream ready, HCCL intrinsic service, legal Best Schedule or formal Product TPS bound.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'cross_arm_fixed_w0':same_work,
                      'cycles':cycles,'on_stage_rank_cycles':arms['on']['stage_rank_cycle_count']}))


if __name__=='__main__':main()
