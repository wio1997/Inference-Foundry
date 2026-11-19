"""Bind an actual G1178 P/D pair and the measured warmup boundary to AISBench."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
import urllib.request

MODEL = 'ds41flash'
ROLE_BASES = {'prefill':'http://<ROLE_HOST>:19093', 'decode':'http://<ROLE_HOST>:19094'}
PROXY = 'http://<ROLE_HOST>:18090'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    raw = path.read_bytes()
    return json.loads(raw), digest(raw)


def request(base, path):
    with urllib.request.urlopen(base+path, timeout=12) as response:
        assert response.status == 200
        return response.read()


def proof_age(now, proof):
    return (now - datetime.datetime.fromisoformat(proof['at'])).total_seconds()


def main(args):
    assert socket.gethostname() == '<ROLE_HOSTNAME>'
    root = args.root.resolve(strict=True)
    client = root/'client_ds41_g1122'
    run = root/'sg_pd_g1178'
    output = args.output.resolve()
    assert output.is_relative_to(client) and not output.exists()
    assert args.run_id == Path(args.run_id).name and args.run_id not in ('.','..')
    now = datetime.datetime.now(datetime.timezone.utc)
    p, psha = load(args.prefill)
    d, dsha = load(args.decode)
    pd, pdsha = load(args.runtime)
    assert -60 <= proof_age(now,p) < 600 and 0 <= proof_age(now,d) < 600
    assert p['ready'] == d['ready'] == 200
    assert p['all8_ownership_and_FDs_verified'] and d['all8_ownership_and_FDs_verified']
    assert p['guard_FD_verified'] and p['guard_PID']==2227468
    assert d['graph_capture_all8_verified']
    assert pd['terminal_verified'] and pd['raw_files_verified'] and not (pd['outer_live'] or pd['inner_live'])
    router = pd['handle']['router']
    guard = pd['fresh_live_guard']
    assert p['guard_PID'] == d['handle']['guard_PID'] == router['guard_PID'] == guard['pid'] == 2227468
    assert guard['phase'] == 'holding'
    guard_remaining = guard['bound_s'] - proof_age(now,guard)
    assert guard_remaining > args.minimum_remaining, guard_remaining
    result = pd['result']
    assert result['math1']=='91' and result['completed']==3
    for key in ('transfer_count_delta_by_P_rank','transfer_mb_delta_by_P_rank','D_full_graph_counter_delta_by_rank'):
        assert set(map(int,result[key]))==set(range(8)) and min(result[key].values())>0
    assert {row['rank'] for row in pd['P_all8_runtime_query_policy']}==set(range(8))
    assert p['container']['Id']==router['P_container_ID'] and d['container_ID']==router['D_container_ID']
    assert p['handle']['pid']==router['P_controller_PID'] and d['handle']['pid']==router['D_controller_PID']
    state, _=load(run/'state.json')
    assert state['generation']==1178 and state['pid']==d['handle']['pid'] and state['container_id']==d['container_ID']
    assert str(run/'node_controller.py').encode() in Path('/proc',str(state['pid']),'cmdline').read_bytes().split(b'\0')
    model_info=json.loads(subprocess.check_output(['docker','inspect',d['container_ID']],timeout=10))[0]
    assert model_info['State']['Running'] and model_info['Id']==d['container_ID']
    workers=json.loads(request(PROXY,'/workers'))['workers']
    assert len(workers)==2 and {(w['worker_type'],w['url']) for w in workers
        if w['model_id']==MODEL and w['is_healthy'] and w['load']==0}==set(ROLE_BASES.items())
    assert json.loads(request(PROXY,'/v1/models'))['data'][0]['id']==MODEL

    sys.path.insert(0,str(client/'benchmark_pipeline_SG_g1183'))
    from metric_selectors import select_counter
    roles={}
    metric='sglang:prefill_effective_tokens_total'
    for role, receipt, cid, chunk in [('prefill',p,p['container']['Id'],8192),('decode',d,d['container_ID'],1024)]:
        base=ROLE_BASES[role]
        request(base,'/ready')
        opts=receipt['server_args']
        assert (opts['tp_size'],opts['ep_size'],opts['pp_size'])==(8,8,1)
        assert opts['context_length']==263296 and opts['chunked_prefill_size']==chunk
        assert opts['speculative_algorithm']=='DSPARK' and opts['speculative_num_draft_tokens']==6
        assert not opts['disable_overlap_schedule'] and not opts['disable_radix_cache']
        values=request(base,'/metrics').decode()
        for name in ('num_running_reqs','num_queue_reqs'):
            counts=[float(v) for v in re.findall(r'^sglang:'+name+r'(?:\{[^\n]*\})?\s+([0-9.eE+\-]+)',values,re.M)]
            assert counts and sum(counts)==0,(role,name,counts)
        labels={'engine_type':role,'mode':'device_hit','model_name':MODEL,'tp_rank':'0',
                'moe_ep_rank':'0','pp_rank':'0'}
        counter=select_counter(values,metric,labels)
        roles[role]={'ready':True,'controller_pid':receipt['handle']['pid'],'container_id':cid,
            'started_at':receipt['handle']['at'],'base_url':base,'ready_status':200,'readiness_path':'/ready',
            'metrics_url':base+'/metrics','prefix_hit_counter':metric,'prefix_hit_counter_labels':labels,
            'device_hit_count_at_gate':counter,'verified_args':opts}
    assets, asset_sha=load(client/'aisbench_assets_state_g1183.json')
    assert assets['phase']=='complete' and assets['profiles']==24 and assets['prefix_count']==448
    assert (client/'cold_prefix_audit.json').is_file()
    dataset, _=load(client/'dataset_manifest.json')
    formal=[row['sha256'] for row in dataset['profiles']
            if row['phase']=='formal' and row['input_tokens']==args.length]
    assert len(formal)==1
    for ledger in (client/'formal_submissions_sglang.json',
                   root/'client_ds41_g503/formal_submissions_sglang.json'):
        if ledger.is_file():
            records, _=load(ledger)
            assert all(row['dataset_sha256']!=formal[0] for row in records['submissions']), ledger
    prior=client/'runs'/'sglang'/args.run_id
    prefill_compile=None
    if args.stage in {'remaining','formal'}:
        warm, _=load(prior/'state.json')
        if args.stage=='remaining':
            assert warm['phase']=='warmup_paused_for_P_snapshot' and warm['completed_warmups']==3
        else:
            assert warm['phase']=='warmup_diagnostics_complete_pending_compile_audit' and warm['readiness_passed']
            assert args.prefill_compile is not None
            prefill_compile, compare_sha=load(args.prefill_compile)
            assert prefill_compile['same_actual_P_instance'] and prefill_compile['P_container_ID']==roles['prefill']['container_id']
            assert prefill_compile['persistent_compiler_artifacts_unchanged']
            assert prefill_compile['before_label']==f'after_warmup3_{args.length}'
            assert prefill_compile['after_label']==f'after_warmup5_{args.length}'
            prefill_compile['comparison_SHA256']=compare_sha
    elif args.stage=='initial':
        assert not prior.exists()
    out={'schema':1,'task_id':root.name,'engine':'sglang','profile_epoch':1178,'model_id':MODEL,
        'validated_at':now.isoformat(),'proxy_url':PROXY,'ready':True,'pd_transfer_verified':True,
        'workload_guard_actual_verified':True,'configuration_verified':True,'resources_verified':True,
        'max_model_len':263296,'cold_mode':'novel_prefixes','cold_prefix_audit':'client_ds41_g1122/cold_prefix_audit.json',
        'formal_not_previously_submitted':True,'warmup_dataset_phases':[f'warmup{i}' for i in range(1,6)],
        'warmup_only':args.stage!='formal','P_compiler_cache_verified_after_warmups':prefill_compile is not None,
        'guard_remaining_seconds':guard_remaining,'roles':roles,'actual_decode_compile_target':{
            'host_run_root':str(run),'container_id':d['container_ID'],'container_name':d['container_ID']},
        'source_receipts_sha256':{'P_readiness':psha,'D_readiness':dsha,'PD_runtime':pdsha,'assets':asset_sha},
        'prefill_compiler_comparison':prefill_compile,'run_id':args.run_id,'stage':args.stage,'input_tokens':args.length,
        'novel_formal_dataset_sha256':formal[0],
        'D_prefix_counter_scope':'Observed decode prefill-effective device series. P transfer is recorded separately.'}
    if args.stage=='initial':
        out['warmup_pause_after']=3
    with output.open('x') as file:
        json.dump(out,file,indent=2)
        file.write('\n')
    print('G1229_SG_AISBENCH_GATE_JSON '+json.dumps({'at':now.isoformat(),'stage':args.stage,
        'gate':str(output),'sha256':digest(output.read_bytes()),'requests_submitted':False}),flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for key in ('root','prefill','decode','runtime','output'):
        cli.add_argument('--'+key,type=Path,required=True)
    cli.add_argument('--prefill-compile',type=Path)
    cli.add_argument('--run-id',required=True)
    cli.add_argument('--stage',choices=('diagnostic','initial','remaining','formal'),required=True)
    cli.add_argument('--length',type=int,choices=(32768,65536,131072,262144),required=True)
    cli.add_argument('--minimum-remaining',type=int,default=900)
    main(cli.parse_args())
