#!/usr/bin/env python3
"""Attribute copied Run580 Level0 task timelines to two complete Graph replays."""
from __future__ import annotations
import argparse,csv,hashlib,json,statistics
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT=Path('/data/wio/Inference_Foundry')
RUN=ROOT/'evidence/20260928_loop080_bound/run580'
LEDGER=ROOT/'evidence/20260927_loop077_bound/run378/ledger.json'
LEDGER_SHA='c5bbe55c0853b011188cd0f2e3e6e0a673951b8e95c5b6fd625ca80e45827211'

def need(ok,why):
    if not ok:raise ValueError(why)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def D(x):return Decimal(str(x).strip())
def start(row):return D(row['task_start(us)'])
def end(row):return D(row['task_stop(us)'])
def ms(x):return float(x/1000)

def hname(row):
    kind=row['kind'];dtype=row['dtype']
    if kind=='hcom_reduceScatter':return 'aiv_reduce_scatter_bfloat16_t'
    if kind=='hcom_alltoall':return 'aiv_all_to_all_bfloat16_t'
    if kind=='hcom_allGather':
        return 'aiv_all_gather_float' if dtype=='FP32' else 'aiv_all_gather_bfloat16_t'
    raise ValueError(f'bad ledger kind {kind}')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    final_path=RUN/'final_admission.json'
    final=json.loads(final_path.read_text())
    manifest_path=RUN/'offline_parse/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    need(manifest['status']=='offline_copied_parse_complete' and
         manifest['raw_source_immutable'] is True and
         manifest['final_admission_sha256']==sha(final_path) and
         len(manifest['parsed'])==8,'Run580 parse manifest admission')
    for path,expected in final['raw_profile_provenance'].items():
        need(sha(ROOT/path)==expected,f'Run580 raw profile changed {path}')
    need(sha(LEDGER)==LEDGER_SHA,'Run378 ledger SHA drift')
    ordered=json.loads(LEDGER.read_text())['ordered_tasks']
    need(len(ordered)==265,'ordered ledger length')
    rank_rows=[]
    provenance={str(final_path.relative_to(ROOT)):sha(final_path),
                str(manifest_path.relative_to(ROOT)):sha(manifest_path),
                str(LEDGER.relative_to(ROOT)):sha(LEDGER)}
    for rank in range(8):
        item=next((x for x in manifest['parsed'] if x['rank']==rank),None)
        need(item is not None,f'rank{rank} parsed entry')
        task_path=ROOT/item['task_time_csv']
        trace_path=ROOT/item['trace_json']
        need(sha(task_path)==item['task_time_sha256'],f'rank{rank} task SHA')
        provenance[str(task_path.relative_to(ROOT))]=sha(task_path)
        provenance[str(trace_path.relative_to(ROOT))]=sha(trace_path)
        with task_path.open(newline='') as stream:
            reader=csv.DictReader(stream)
            need({'Device_id','kernel_name','kernel_type','stream_id','task_id',
                  'task_time(us)','task_start(us)','task_stop(us)'}<=set(reader.fieldnames or ()),
                 f'rank{rank} task CSV schema')
            tasks=list(reader)
        need(len(tasks)==4984 and all(int(t['Device_id'])==rank for t in tasks),
             f'rank{rank} complete Level0 task census')
        g=[x for x in tasks if x['kernel_type']=='KERNEL_MIX_AIC' and
           x['kernel_name'].startswith('GroupedMatmul')]
        h=[x for x in tasks if x['kernel_type']=='KERNEL_AIVEC' and
           x['kernel_name'].startswith(('aiv_all_gather','aiv_reduce_scatter','aiv_all_to_all'))]
        barrier=[x for x in tasks if x['kernel_name']=='aiv_all_reduce_float']
        models=[x for x in tasks if x['kernel_type']=='MODEL_EXECUTE']
        need(len(g)==172 and len(h)==530 and len(barrier)==2 and len(models)==4,
             f'rank{rank} two active Graphs plus barrier count')
        need(len({x['stream_id'] for x in g})==len({x['stream_id'] for x in h})==1 and
             g[0]['stream_id']!=h[0]['stream_id'] and
             all(int(x['task_id'])==i%86 for i,x in enumerate(g)) and
             all(int(x['task_id'])==1+4*(i%265) for i,x in enumerate(h)),
             f'rank{rank} separate complete Graph task identity')
        for i,x in enumerate(g):
            expected='GroupedMatmulSwigluQuantV2' if i%2==0 else 'GroupedMatmul'
            need(x['kernel_name'].startswith(expected),f'rank{rank} GMM order task{i}')
        for i,x in enumerate(h):
            need(x['kernel_name']==hname(ordered[i%265]),
                 f'rank{rank} HCCL ledger order task{i}')
        events=json.loads(trace_path.read_text())
        g_events=[x for x in events if x.get('ph')=='X' and
                  x.get('args',{}).get('Model Id')==33 and
                  x.get('args',{}).get('Task Type')=='KERNEL_MIX_AIC']
        h_events=[x for x in events if x.get('ph')=='X' and
                  x.get('args',{}).get('Model Id')==32 and
                  x.get('args',{}).get('Task Type')=='KERNEL_AIVEC']
        need(len(g_events)==172 and len(h_events)==530 and
             all(x['name'].startswith('aclnnGroupedMatmul') for x in g_events) and
             all(x['name'].startswith('hcom_') for x in h_events),
             f'rank{rank} trace Graph model identity')
        def csv_keys(rows):
            return {(int(x['task_id']),int(x['stream_id']),start(x)) for x in rows}
        def trace_keys(rows):
            return {(int(x['args']['Task Id']),int(x['args']['Physic Stream Id']),D(x['ts']))
                    for x in rows}
        need(csv_keys(g)==trace_keys(g_events) and
             csv_keys(h)==trace_keys(h_events),
             f'rank{rank} copied CSV/trace native task join')
        markers={name:[x for x in events if x.get('name')==name and x.get('ph')=='X']
                 for name in ('run580_active_serial','run580_active_concurrent',
                              'run580_gmm_graph_replay_submit','run580_hccl_graph_replay_submit')}
        need(len(markers['run580_active_serial'])==len(markers['run580_active_concurrent'])==1 and
             len(markers['run580_gmm_graph_replay_submit'])==
             len(markers['run580_hccl_graph_replay_submit'])==2,
             f'rank{rank} Host marker cardinality')
        bench=json.loads((RUN/'live/b/bench'/f'rank{rank}.json').read_text())['mixed']
        bench_path=RUN/'live/b/bench'/f'rank{rank}.json'
        provenance[str(bench_path.relative_to(ROOT))]=sha(bench_path)
        observations=[]
        for i,arm in enumerate(('serial','concurrent')):
            gg=g[i*86:(i+1)*86];hh=h[i*265:(i+1)*265]
            need(all(start(x)<end(x) and D(x['task_time(us)'])>0 for x in gg+hh),
                 f'rank{rank} {arm} positive tasks')
            gs,ge=min(map(start,gg)),max(map(end,gg))
            hs,he=min(map(start,hh)),max(map(end,hh))
            scope=markers['run580_active_'+arm][0]
            scope_start=D(scope['ts']);scope_end=scope_start+D(scope['dur'])
            gm=markers['run580_gmm_graph_replay_submit'][i]
            hm=markers['run580_hccl_graph_replay_submit'][i]
            need(scope_start<D(gm['ts'])<D(hm['ts'])<scope_end and
                 scope_start<=gs<ge<=scope_end and
                 scope_start<=hs<he<=scope_end,
                 f'rank{rank} {arm} Host/device attribution')
            during=[x for x in hh if start(x)<ge]
            after=[x for x in hh if start(x)>=ge]
            overlap=max(Decimal(0),min(ge,he)-max(gs,hs))
            observations.append({
                'arm':arm,'gmm_task_count':86,'hccl_task_count':265,
                'gmm_stream_id':gg[0]['stream_id'],'hccl_stream_id':hh[0]['stream_id'],
                'gmm_span_ms':ms(ge-gs),'hccl_span_ms':ms(he-hs),
                'hccl_first_minus_gmm_first_ms':ms(hs-gs),
                'native_interval_overlap_ms':ms(overlap),
                'combined_native_envelope_ms':ms(max(ge,he)-min(gs,hs)),
                'hccl_kernel_duration_sum_ms':ms(sum((D(x['task_time(us)']) for x in hh),Decimal(0))),
                'hccl_before_gmm_end_count':len(during),
                'hccl_before_gmm_end_duration_sum_ms':ms(sum((D(x['task_time(us)']) for x in during),Decimal(0))),
                'hccl_before_gmm_end_median_us':float(statistics.median(D(x['task_time(us)']) for x in during)) if during else None,
                'hccl_after_gmm_end_count':len(after),
                'hccl_after_gmm_end_duration_sum_ms':ms(sum((D(x['task_time(us)']) for x in after),Decimal(0))),
                'hccl_after_gmm_end_median_us':float(statistics.median(D(x['task_time(us)']) for x in after)) if after else None,
                'host_gmm_submit_to_hccl_submit_ms':ms(D(hm['ts'])-D(gm['ts'])),
                'profiled_event_joint_ms':bench['profile_steps'][i+1]['event_sample']['joint_ms'],
            })
        need(observations[0]['native_interval_overlap_ms']==0 and
             0.2<observations[1]['hccl_first_minus_gmm_first_ms']<0.6 and
             observations[1]['native_interval_overlap_ms']>12 and
             observations[1]['hccl_span_ms']>observations[0]['hccl_span_ms']*3,
             f'rank{rank} observed mechanism drift')
        prefix=observations[1]['hccl_before_gmm_end_count']
        concurrent_gmm_end=max(map(end,g[86:]))
        crossing=sum(start(x)<concurrent_gmm_end<end(x) for x in h[265:])
        need(prefix in (84,86) and crossing==1,
             f'rank{rank} overlap boundary identity')
        matched={
            'prefix_call_count':prefix,
            'serial_prefix_median_task_us':float(statistics.median(
                D(x['task_time(us)']) for x in h[:prefix])),
            'concurrent_prefix_median_task_us':float(statistics.median(
                D(x['task_time(us)']) for x in h[265:265+prefix])),
            'serial_tail_median_task_us':float(statistics.median(
                D(x['task_time(us)']) for x in h[prefix:265])),
            'concurrent_tail_median_task_us':float(statistics.median(
                D(x['task_time(us)']) for x in h[265+prefix:])),
            'tasks_crossing_gmm_end':crossing,
        }
        rank_rows.append({'rank':rank,'task_csv':item['task_time_csv'],
                          'trace_json':item['trace_json'],'barrier_task_count':2,
                          'gmm_model_id':33,'hccl_model_id':32,
                          'serial':observations[0],'concurrent':observations[1],
                          'matched_ordinal_hccl_task_duration':matched})
    result={'status':'scoped_all8_attributed_level0_timeline',
            'run_tag':'LOOP080-RUN580-B',
            'provenance':provenance,
            'rank_rows':rank_rows,
            'claim':'HCCL AIV tasks begin 0.37-0.42ms after GMM starts under concurrent replay; their exported task intervals overlap, and HCCL task durations inflate during the GMM task envelope. Late initial HCCL Graph launch alone does not explain this slowdown.',
            'limits':['Level0 AIV task duration includes waiting/scheduling; it is not physical link bandwidth, AIV occupancy or HBM payload.',
                      'Cannot separate rank-arrival wait, HCCL internal scheduling, AIV/HBM contention or graph-runtime effects from these task intervals alone.',
                      'Rank-local relative times only; no direct cross-rank clock subtraction.',
                      'Profiled intervals explain mechanism; Run579/580 unprofiled four-arm events are service-time evidence.',
                      'Independent ready inputs do not establish legal production DAG overlap or Product E2E gain.'],
            'strict_capacity_upper':None,
            'resource_hardware_endpoint_s':None,
            'scheduling_execution_endpoint_s':None,
            'product_e2e_tps_interval':None,
            'numeric_current_to_limit_distance':None}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'ranks':len(rank_rows),
                      'concurrent_hccl_span_ms':[round(x['concurrent']['hccl_span_ms'],3) for x in rank_rows]}))

if __name__=='__main__':main()
