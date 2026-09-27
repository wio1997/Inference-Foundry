#!/usr/bin/env python3
"""Post-stop salvage of Run578 raw MemoryAccess counters only; no service time."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,statistics
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT=Path('/data/wio/Inference_Foundry')
LIVE=ROOT/'evidence/20260928_loop080_bound/run578/live'
ARM=LIVE/'b'
METRICS=tuple(f'{core}_{metric}(KB)' for core in ('aic','aiv') for metric in (
    'read_main_memory_datas','write_main_memory_datas','GM_to_L1_datas',
    'L0C_to_L1_datas','L0C_to_GM_datas','GM_to_UB_datas','UB_to_GM_datas'))
TYPES=('GroupedMatmulSwigluQuantV2','GroupedMatmul')

def need(ok,msg):
    if not ok: raise ValueError(msg)

def read(path,provenance):
    raw=path.read_bytes()
    provenance[str(path.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
    return raw

def analyse_csv(path,rank,provenance):
    raw=read(path,provenance)
    with path.open(newline='') as stream:
        reader=csv.DictReader(stream)
        need({'Device_id','Model ID','Task ID','Stream ID','Type','Name',
              'Start Time(us)','Duration(us)'}|set(METRICS)
             <=set(reader.fieldnames or ()),'CSV fields')
        rows=list(reader)
    need(len(rows)==172,f'rank{rank} not exactly two complete Graph replays: {len(rows)}')
    need([int(row['Task ID']) for row in rows]==list(range(86))*2,
         f'rank{rank} task sequence')
    need(len({row['Model ID'] for row in rows})==1 and
         len({row['Stream ID'] for row in rows})==1 and
         all(int(row['Device_id'])==rank for row in rows),
         f'rank{rank} model/stream/device')
    starts=[];ends=[]
    for i,row in enumerate(rows):
        try:
            start=Decimal(row['Start Time(us)'].strip())*1000
            duration=Decimal(row['Duration(us)'].strip())*1000
        except InvalidOperation as error:
            raise ValueError(f'rank{rank} bad device time row{i}') from error
        need(start.is_finite() and duration.is_finite() and duration>0,
             f'rank{rank} invalid device time row{i}')
        starts.append(start);ends.append(start+duration)
    need(all(starts[i]>starts[i-1] for i in range(1,172)) and
         starts[86]>ends[85] and len(set(starts))==172,
         f'rank{rank} replay order/unique task instances')
    need(all(row['Type']==TYPES[i%2] for i,row in enumerate(rows)),
         f'rank{rank} GMM alternation')
    need(all(('GroupedMatmulSwigluQuant' in row['Name'] if i%2==0 else
              'GroupedMatmulWeightNz' in row['Name'])
             for i,row in enumerate(rows)),f'rank{rank} native name')
    need(all(float(row['aic_read_main_memory_datas(KB)'])+
                 float(row['aiv_read_main_memory_datas(KB)'])>0 for row in rows),
         f'rank{rank} nonpositive GMM main-memory read')
    replays=[]
    for repeat in range(2):
        pair=rows[86*repeat:86*(repeat+1)]
        entry={}
        for parity,name in enumerate(('gmm1','gmm2')):
            selected=pair[parity::2]
            counters={}
            for metric in METRICS:
                vals=[float(row[metric]) for row in selected]
                need(all(math.isfinite(v) and v>=0 for v in vals),
                     f'rank{rank} invalid {metric}')
                counters[metric]=sum(vals)
            entry[name]={'task_count':43,'counter_KB':counters}
        read_kb=sum(entry[name]['counter_KB'][f'{core}_read_main_memory_datas(KB)']
                    for name in ('gmm1','gmm2') for core in ('aic','aiv'))
        write_kb=sum(entry[name]['counter_KB'][f'{core}_write_main_memory_datas(KB)']
                     for name in ('gmm1','gmm2') for core in ('aic','aiv'))
        entry['counter_reported_main_memory_read_GB']=read_kb*1024/1e9
        entry['counter_reported_main_memory_write_GB']=write_kb*1024/1e9
        replays.append(entry)
    return {'csv':str(path.relative_to(ROOT)),
            'model_id':rows[0]['Model ID'],'stream_id':rows[0]['Stream ID'],
            'complete_task_count':172,'complete_replays':replays,
            'replay_gap_ns':str(starts[86]-ends[85]),
            'first_start_ns':str(starts[0]),'last_end_ns':str(ends[-1])}


def validate_profiler_identity(info,meta,rank):
    config=info['config']
    need(info['rank_id']==rank and info['cann_version']=='9.1.0' and
         info['torch_npu_version']=='2.10.0.post4' and
         config['experimental_config']['_profiler_level']=='Level1' and
         config['experimental_config']['_aic_metrics']=='ACL_AICORE_MEMORY_ACCESS' and
         config['common_config']['schedule']==
         {'wait':0,'active':2,'warmup':1,'repeat':1,'skip_first':0,'skip_first_wait':0} and
         meta['parallel_group_info']['group_name_3']['group_rank']==rank and
         meta['parallel_group_info']['group_name_3']['global_ranks']==list(range(8)),
         f'rank{rank} profiler identity/config')


def build():
    provenance={}
    client=json.loads(read(ARM/'client_admission.json',provenance))
    need(client['status']=='client_two_phase_admitted' and client['request_count']==96 and
         client['warmup_summary']['success']==client['measured_summary']['success']==48 and
         client['warmup_summary']['fail']==client['measured_summary']['fail']==0 and
         client['warmup_summary']['concurrency']==client['measured_summary']['concurrency']==12 and
         client['warmup_summary']['max_tokens']==client['measured_summary']['max_tokens']==1024,
         'client contract')
    need(len(client['request_index'])==96 and
         [x['phase'] for x in client['request_index']]==['warmup']*48+['measured']*48,
         'client phase')
    need(int((ARM/'server_post_count.txt').read_text())==96,'POST count before stop')
    log=ROOT/'logs/serve_dsv4f-w4a8_8npu_dp1tp8_mlen1M_nomooncake_LOOP080-RUN578-B.log'
    read(log,provenance)
    need(log.read_text(errors='replace').count('POST /v1/chat/completions')==96,
         'POST count after stop')
    cleanup=dict(line.split('=',1) for line in read(LIVE/'cleanup_status.txt',provenance).decode().splitlines())
    need(cleanup['run_exit']==cleanup['final_exit']=='1' and
         all(cleanup[k]=='0' for k in ('stop_exit','stop_verify_exit','restore_exit',
             'source_sha_exit','source_compare_exit','script_sha_exit','script_compare_exit')),
         'failed-run cleanup/restore')
    need(read(LIVE/'source_before.sha256',provenance)==read(LIVE/'source_after.sha256',provenance),
         'source restore')
    need(read(LIVE/'scripts_before.sha256',provenance)==read(LIVE/'scripts_after.sha256',provenance),
         'scripts stable')
    for name in ('final_stop_verify.log','stop_npu_probe.txt','restore.log',
                 'source_before.sha256','source_after.sha256','scripts_before.sha256',
                 'scripts_after.sha256'):
        read(LIVE/name,provenance)
    read(ARM/'service_validate.log',provenance)
    need((LIVE/'final_stop_verify.log').read_text().strip().splitlines()[-1].startswith('idle HBM ['),
         'all8 idle')
    census_path=ROOT/'evidence/20260928_loop080_bound/run576/operand_census.json'
    census=json.loads(read(census_path,provenance))
    need(provenance[str(census_path.relative_to(ROOT))]==
         '615aeb42086b035903867261e45642a6390a13ba396d91ec31d7cb62320ca1e8',
         'Run576 census SHA')
    read(LIVE/'install.json',provenance)
    read(LIVE/'patch_check.json',provenance)
    runtimes=[]
    for cohort in range(1,9):
        ids=None
        for rank in range(8):
            path=ARM/'runtime'/f'rank{rank}_cohort{cohort}.json'
            row=json.loads(read(path,provenance))
            need(row['rank']==rank and row['cohort']==cohort and row['pass'] is True and
                 row['generated_output_counts']==[1024]*12,
                 f'rank{rank} cohort{cohort} Runtime')
            if ids is None: ids=row['req_ids']
            need(row['req_ids']==ids,f'cohort{cohort} all8 IDs')
        need(len(ids)==len(set(ids))==12,f'cohort{cohort} unique IDs')
        matched=[]
        for req in ids:
            options=[item for item in client['request_index']
                     if req==item['response_id'] or req.startswith(item['response_id']+'-')]
            need(len(options)==1,f'cohort{cohort} client ID join')
            matched.append(options[0])
        phase='warmup' if cohort<=4 else 'measured'
        offset=((cohort-1)%4)*12
        need({item['phase'] for item in matched}=={phase} and
             {item['dataset_index'] for item in matched}==set(range(offset,offset+12)),
             f'cohort{cohort} phase/dataset')
        runtimes.append(ids)
    need(len(list((ARM/'runtime').glob('rank*_cohort*.json')))==64,
         'all64 Runtime records')
    for name in ('loop080_realweight_profile_run578.py','loop080_realweight_patch_run578.py',
                 'loop080_realweight_reduce_run578.py','run_loop080_realweight_run578.sh'):
        read(ROOT/'scripts'/name,provenance)
    profiles=[]
    for rank in range(8):
        fixture=ROOT/f'evidence/20260928_loop080_bound/run576/live/b/capture/rank{rank}_cohort5.json'
        read(fixture,provenance)
        need(provenance[str(fixture.relative_to(ROOT))]==
             census['provenance'][str(fixture.relative_to(ROOT))],
             f'rank{rank} Run576 fixture SHA')
        paths=list((ARM/'bench'/'profile'/f'rank{rank}').glob('**/kernel_details.csv'))
        need(len(paths)==1,f'rank{rank} unique offline CSV')
        profiles.append(analyse_csv(paths[0],rank,provenance))
        profile_dir=ARM/'bench'/'profile'/f'rank{rank}'
        info_path=list(profile_dir.glob(f'**/profiler_info_{rank}.json'))
        meta_path=list(profile_dir.glob('**/profiler_metadata.json'))
        need(len(info_path)==len(meta_path)==1,f'rank{rank} unique profiler metadata')
        info=json.loads(read(info_path[0],provenance))
        meta=json.loads(read(meta_path[0],provenance))
        validate_profiler_identity(info,meta,rank)
        offline=profile_dir/'offline_parse.log'
        offline_text=read(offline,provenance).decode(errors='replace')
        need('All profiling data parsed in a total time' in offline_text,
             f'rank{rank} offline parse completion')
        raw_files=sorted(path for path in profile_dir.glob('**/*') if path.is_file() and
                         any(part.startswith('PROF_') or part=='FRAMEWORK'
                             for part in path.relative_to(profile_dir).parts))
        need(len(raw_files)>0,f'rank{rank} raw profile absent')
        for path in raw_files:read(path,provenance)
        profiles[-1]['raw_profile_file_count']=len(raw_files)
    reads=[statistics.median(x['counter_reported_main_memory_read_GB']
             for x in row['complete_replays']) for row in profiles]
    writes=[statistics.median(x['counter_reported_main_memory_write_GB']
              for x in row['complete_replays']) for row in profiles]
    return {'status':'partial_counter_diagnostic_after_online_parse_failure',
            'run_tag':'LOOP080-RUN578-B',
            'measurement_class':'terminal_real_weight_synthetic_activation_GMM_MemoryAccess_only',
            'contract':'48 warmup + 48 measured c12/1024, terminal cohort8',
            'scope':'all8 two complete 43 GMM1 + 43 GMM2 private Graph replay CSV windows',
            'profile_per_rank':profiles,
            'rank_counter_reported_main_memory_read_GB':reads,
            'rank_counter_reported_main_memory_write_GB':writes,
            'counter_semantics':'CANN9.1 Level1 MemoryAccess aic+aiv read/write main-memory KB times 1024; not certified physical HBM payload',
            'timing_status':'A0 samples were not persisted; A1 was not executed; no paired service time',
            'source_restore_exact':True,'all8_idle':True,'post_count':96,
            'provenance':provenance,
            'strict_capacity_upper':None,'resource_hardware_endpoint_s':None,
            'scheduling_execution_endpoint_s':None,'product_e2e_tps_interval':None,
            'numeric_current_to_limit_distance':None,
            'limits':['Run578 controller exited1; only raw counter window recovered offline.',
                      'A0 not persisted and A1 not executed; no paired service time or formal TPS is admitted.',
                      'Counter-reported main-memory bytes are current Graph traffic, not compulsory work or physical HBM-controller payload.',
                      'Different diagnostic route/activation from Run99 formal W0 and no mixed Target/DSpark/HCCL/KV contention.']}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    result=build();a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],
                      'rank_read_GB':result['rank_counter_reported_main_memory_read_GB']}))
if __name__=='__main__':main()
