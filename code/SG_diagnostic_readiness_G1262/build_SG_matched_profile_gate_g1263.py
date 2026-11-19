"""Bind a verified G1178 32K result to the separate matched capture inputs."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import urllib.request

MODEL='ds41flash'
ROLE_BASES={'prefill':'http://<ROLE_HOST>:19093','decode':'http://<ROLE_HOST>:19094'}
PROXY='http://<ROLE_HOST>:18090'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def request(base,path):
    with urllib.request.urlopen(base+path,timeout=12) as response:
        assert response.status==200
        return response.read()


def dataset_identity(manifest):
    return sorted((row['phase'],row['input_tokens'],row['requests'],row['sha256']) for row in manifest['profiles'])


def main(args):
    assert socket.gethostname()=='<ROLE_HOSTNAME>'
    root=args.root.resolve(strict=True)
    client=root/'client_ds41_g1122'
    assert args.run_id==Path(args.run_id).name and args.run_id not in ('.','..')
    output=args.output.resolve()
    assert output.is_relative_to(client) and not output.exists()
    now=datetime.datetime.now(datetime.timezone.utc)
    p,d,runtime=[read(path) for path in (args.prefill,args.decode,args.runtime)]
    for role in (p,d):
        age=(now-datetime.datetime.fromisoformat(role['at'])).total_seconds()
        assert -60<=age<600 and role['ready']==200
        assert role['all8_ownership_and_FDs_verified']
    assert p['guard_FD_verified'] and p['guard_PID']==d['handle']['guard_PID']==2227468
    assert d['graph_capture_all8_verified'] and runtime['terminal_verified'] and runtime['raw_files_verified']
    router=runtime['handle']['router']
    assert router['P_container_ID']==p['container']['Id'] and router['D_container_ID']==d['container_ID']
    result=runtime['result']
    assert result['math1']=='91' and result['completed']==3
    for name in ('transfer_count_delta_by_P_rank','D_full_graph_counter_delta_by_rank'):
        assert set(map(int,result[name]))==set(range(8)) and min(result[name].values())>0
    lease=runtime['fresh_live_guard']
    assert lease['pid']==router['guard_PID']==2227468 and lease['phase']=='holding'
    remaining=lease['bound_s']-(now-datetime.datetime.fromisoformat(lease['at'])).total_seconds()
    assert remaining>args.minimum_remaining
    assert read(args.prefill_export)['P_CID']==p['container']['Id']
    run=root/'sg_pd_g1178'
    state=read(run/'state.json')
    assert state['pid']==d['handle']['pid'] and state['generation']==1178 and state['container_id']==d['container_ID']
    info=json.loads(subprocess.check_output(['docker','inspect',state['container_id']],timeout=10))[0]
    assert info['State']['Running'] and any(m['Source']==str(run) and m['Destination']=='/task/sg_pd_g1178' and m['RW']
                                          for m in info['Mounts'])
    assert str(run/'node_controller.py').encode() in Path('/proc',str(state['pid']),'cmdline').read_bytes().split(b'\0')
    assert not (client/'profile_submitted_sglang_epoch1178.json').exists()

    reference=client/'runs'/'sglang'/args.run_id
    assert read(reference/'state.json')['phase']=='all_phases_terminal'
    run_manifest=read(reference/'run_manifest.json')
    assert run_manifest['engine']=='sglang' and run_manifest['lengths']==[32768] and run_manifest['warmup_passes']==5
    assert run_manifest['gate']['roles']['prefill']['container_id']==p['container']['Id']
    assert run_manifest['gate']['roles']['decode']['container_id']==d['container_ID']
    audit=read(reference/'audit_results.json')
    assert audit['all_required_raw_metrics_and_usage_validated'] and len(audit['profiles'])==1
    profile=audit['profiles'][0]
    assert profile['status']=='cold_hot_raw_metrics_validated' and profile['cold_cache_counter_verified']
    assert profile['cold']['fully_validated'] and profile['hot_replay']['fully_validated']
    assert all(profile[name]['requests']==32 and profile[name]['successful_requests']==32 for name in ('cold','hot_replay'))
    warm=read(reference/'warmup_stability_32768.json')
    assert warm['passed'] and not any(warm['compiler_artifact_changes'].values())
    resumes=[read(path) for path in reference.glob('resume_manifest_*.json')]
    formal=[row for row in resumes if row['refreshed_gate']['stage']=='formal']
    assert len(formal)==1
    fg=formal[0]['refreshed_gate']
    assert fg['P_compiler_cache_verified_after_warmups'] and fg['prefill_compiler_comparison']['persistent_compiler_artifacts_unchanged']
    assert fg['prefill_compiler_comparison']['P_container_ID']==p['container']['Id']

    usage=Path(run_manifest['usage_contract']).resolve(strict=True)
    assert usage.is_relative_to(client) and read(usage)['reviewed']
    inputs=args.inputs.resolve(strict=True)
    assert inputs.is_relative_to(root)
    assert sha(inputs)=='308d46f5ae05d7d30b8af9b4711b24582fe4b8a0c6fdc76f32473720f7acf4f4'
    frozen=read(inputs)
    assert frozen['original_reserved_prefixes_checked']==448 and frozen['new_prefixes_unique']==16
    assert all(row['input_tokens']==32768 for row in frozen['records'])
    assert len({row['first128_token_ids_sha256'] for row in frozen['records']})==16
    assets=read(client/'dataset_manifest.json')
    assert len(assets['profiles'])==24 and sum(row['requests'] for row in assets['profiles'])==448
    assert assets['tokenizer_sha256']==frozen['tokenizer_sha256']
    assert frozen['dataset_sha256'] not in {row['sha256'] for row in assets['profiles']}
    native_assets=root/'client_ds41_g503/dataset_manifest.json'
    assert sha(native_assets)==frozen['dataset_manifest_sha256']
    assert dataset_identity(read(native_assets))==dataset_identity(assets)
    for base in ROLE_BASES.values():
        request(base,'/ready')
        assert json.loads(request(base,'/v1/models'))['data'][0]['id']==MODEL
    workers=json.loads(request(PROXY,'/workers'))['workers']
    assert len(workers)==2 and {(w['worker_type'],w['url']) for w in workers
           if w['is_healthy'] and w['model_id']==MODEL and w['load']==0}==set(ROLE_BASES.items())
    roles={k:{'base_url':ROLE_BASES[k],'controller_pid':x['handle']['pid'],'container_id':cid}
           for k,x,cid in [('prefill',p,p['container']['Id']),('decode',d,d['container_ID'])]}
    gate={'engine':'sglang','profile_epoch':1178,'verified_at':now.isoformat(),'model_id':MODEL,
        'proxy_url':PROXY,'roles':roles,'ready':True,'workload_guard_actual_verified':True,'pd_transfer_verified':True,
        'all8_D_target_and_draft_graph_captures_verified':True,'P_D_last_two_warmups_compile_verified':True,
        'profile_cohort_distinct_from_formal_and_warmup':True,'profiler_export_roots_checked':True,
        'PD_runtime_proof':result,'verified_32k_run_id':args.run_id,'usage_contract':str(usage),
        'service_lease_remaining_s':remaining,'native_matching_profile_inputs_SHA256':sha(inputs),
        'SGLang_formal_audit_SHA256':sha(reference/'audit_results.json'),
        'SGLang_formal_gate_resume_SHA256':hashlib.sha256(json.dumps(formal[0],sort_keys=True).encode()).hexdigest(),
        'scope':'Current preflight evidence for a distinct P/D profile. No profiler or generation request from this program.'}
    with output.open('x') as file:
        json.dump(gate,file,indent=2)
        file.write('\n')
    print('G1263_SG_PROFILE_GATE_JSON '+json.dumps({
        'gate':str(output),'SHA256':sha(output),'actual_profile_launched':False}),flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for key in ('root','prefill','decode','runtime','prefill-export','inputs','output'):
        cli.add_argument('--'+key,type=Path,required=True)
    cli.add_argument('--run-id',required=True)
    cli.add_argument('--minimum-remaining',type=int,default=2500)
    main(cli.parse_args())
