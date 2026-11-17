"""G133/G134负载的便携AISBench入口，默认只列命令；执行需现有独占服务和共享benchmark锁。"""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,importlib.util,json,os,re,subprocess,time,urllib.request,uuid
ROOT=Path(__file__).resolve().parents[1];CURRENT=ROOT/'reproduction/20261116'
CLIENT='sha256:05358d58d23f2679c71e48d93797d5d2a28d72eedc7fb4900479c89955465d76'
IMAGES={'sglang':'sha256:08367522c4f9803fb876808c2cd4afe459e7e2b0adaab5a0daf36061edf8e517',
        'vllm':'sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'}
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--engine',choices=['sglang','vllm'],required=True)
    p.add_argument('--runtime-root',type=Path,default=ROOT/'.work/current');p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--profiles',type=int,nargs='+',default=[32768,65536,131072,262144],choices=[32768,65536,131072,262144])
    p.add_argument('--server-container');p.add_argument('--benchmark-lock',type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
    runtime=a.runtime_root.resolve();model=a.model_path.resolve();engine=a.engine
    name=a.server_container or f'comparison-{engine}-910c';port=18167 if engine=='sglang' else 18168
    assert len(a.profiles)==len(set(a.profiles)) and a.profiles==sorted(a.profiles)
    phases=[(n,phase,count) for n in a.profiles for phase,count in [('warmup',16),('formal',32)]]
    def command(client,n,phase,count,run):
        return ['docker','exec',client,'timeout','--signal=TERM','--kill-after=30','1200','ais_bench',
            f'/task/configs/aisbench_{engine}_{phase}_{n}.py','--mode','perf','--num-warmups','0','--num-prompts',str(count),
            '--max-num-workers','1','--work-dir',f'/task/results/{engine}/{run}/{phase}_{n}']
    if not a.execute:
        print(json.dumps({'engine':engine,'client_image_id':CLIENT,'request_concurrency':16,'outputs':1024,'stage_timeout_s':1200,'total_timeout_s':3600,
            'commands':[command('<CLIENT_CONTAINER>',n,phase,count,'<RUN_ID>') for n,phase,count in phases]},ensure_ascii=False,indent=2));return
    assert a.benchmark_lock and a.benchmark_lock.parent.is_dir(),'Use the fleet-wide benchmark lock, not a new per-task bypass'
    lock=a.benchmark_lock.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert (runtime/'stage.json').is_file() and load(runtime/'stage.json')['datasets_staged']
    def inspect(x):return json.loads(subprocess.check_output(['docker','inspect',x],text=True,timeout=20))[0]
    server=inspect(name);assert server['State']['Running'] and server['RestartCount']==0 and server['Image']==IMAGES[engine]
    assert server['Config']['Labels']['archive.branch']=='sglang-dsv4.1flash1017-910c'
    mounts={m['Destination']:m for m in server['Mounts']}
    assert Path(mounts['/task']['Source']).resolve()==runtime and Path(mounts['/model']['Source']).resolve()==model
    for rel,item in load(CURRENT/'model-control-hashes.json').items():assert sha(model/rel)==item['expected']
    for item in load(CURRENT/'dataset-manifest.json')['profiles']:assert sha(runtime/item['path'].removeprefix('/task/'))==item['sha256']
    for config in (CURRENT/'configs').glob('*.py'):assert sha(runtime/'configs'/config.name)==sha(config)
    image=json.loads(subprocess.check_output(['docker','image','inspect',CLIENT],text=True))[0]
    assert image['Id']==CLIENT and image['Architecture']=='arm64'
    def api(path,method='GET'):
        with urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}'+path,method=method),timeout=60) as f:
            assert f.status==200;return f.read()
    raw=api('/server_info' if engine=='sglang' else '/server_info?config_format=json')
    def sg_gate(after_warm=False):
        if engine!='sglang':return
        cfg=json.loads(api('/server_info'))
        assert cfg['tp_size']==cfg['ep_size']==16 and cfg['attn_dp_size']==4
        assert cfg['max_running_requests']==16 and cfg['chunked_prefill_size']==8192 and cfg['moe_a2a_backend']=='ascend_fuseep'
        assert cfg['engram_config']=={'placement':'cpu','replication':'per_node','lookup':'all_reduce'}
        assert cfg['speculative_algorithm']=='DSPARK' and cfg['ep_num_redundant_experts']==16 and not cfg['enable_eplb']
        s=runtime/'sglang_penalty133';d=runtime/'continuation_G133_fixed_sampling_256K_20261116'
        assert not (s/'integration/ARMED').exists() and not (d/'capture1/route_histograms/ARMED').exists()
        spec=importlib.util.spec_from_file_location('mapping',runtime/'continuation_G95_replica_model_fix_20261115/validate_receipts.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        audit=mod.validate(s/'mapping_audit',d/'candidate/audit_plan.json');pids=set(audit['container_worker_pids'])
        for group in ['projection_hits','smallHC_hits']:
            if not after_warm:assert not (s/group/'FROZEN.json').exists(),'Fresh generation required; do not reuse old gate state'
        if not after_warm:return
        for group,source in [('vendor_hits','sglang_mhc_vendor112.py'),('projection_hits','sglang_mhc_projection119.py'),
                             ('smallHC_hits','sglang_mhc_small122.py'),('empty_sparse_hits','empty_sparse_mask.py')]:
            hits=[load(s/group/f'rank{rank}.json') for rank in range(16)]
            assert {h['pid'] for h in hits}==pids and all(h['rank']==rank and h['source_sha256']==sha(d/'candidate'/source) for rank,h in enumerate(hits))
            if group=='vendor_hits':assert all(h['rows']>=2048 and h['actual_input_strict_passed'] and h['shadow_disabled_after_record'] and h['atol_rtol']==[2e-5,2e-5] and h['sinkhorn_iters']==20 for h in hits)
            elif group=='projection_hits':
                assert all(h['verified_parameter_count']==80 and h['cached_parameter_count']==86 and h['all_checks_strict_passed'] and h['known_parameter_shadow_disabled_after_record'] for h in hits)
                assert all(e['actual_input_public_coefficients_passed'] and e['atol_rtol']==[2e-5,2e-5] and e['rows']>=2048 for h in hits for e in h['entries'])
            elif group=='smallHC_hits':
                assert all(h['qualified_parameter_count']==h['captured_parameter_count']==h['cached_parameter_count']==86 and h['matmul_allow_hf32'] is False and h['host_checks_outside_capture'] for h in hits)
                assert all(e['outside_capture'] and e['startup_public_passed'] and e['nonzero_probe_public_passed'] and e['atol_rtol']==[2e-5,2e-5] for h in hits for e in h['qualified'])
                assert all({e['parameter_ptr'] for e in h['qualified']}=={e['parameter_ptr'] for e in h['captured']} for h in hits)
            else:assert all(h['empty_rows']>0 and h['empty_output_exact_zero'] and h['outside_capture'] and h['recorded_once_before_formal'] for h in hits)
            if group in {'projection_hits','smallHC_hits'}:
                frozen={'parameters_per_rank':[80 if group=='projection_hits' else 86]*16,'source_sha256':sha(d/'candidate'/source),'public_atol_rtol':[2e-5,2e-5]}
                freeze=s/group/'FROZEN.json'
                if freeze.exists():assert load(freeze)==frozen
                else:freeze.write_text(json.dumps(frozen,indent=2)+'\n')
        loads=json.loads(api('/v1/loads'))['loads'];assert len(loads)==4 and all(x['num_running_reqs']==x['num_waiting_reqs']==0 for x in loads)
    sg_gate()
    run=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+engine
    out=runtime/'results'/engine/run;out.mkdir(parents=True,exist_ok=False);(out/'server_info_before.json').write_bytes(raw)
    (out/'clock_anchor.json').write_text(json.dumps({'wall_epoch_seconds':time.time(),'monotonic_seconds':time.monotonic()}))
    client='comparison-aisbench-'+uuid.uuid4().hex[:8];label='archive.run='+run
    subprocess.run(['docker','run','-d','--name',client,'--label',label,'--network','host','--runtime','runc','--cpus','8','--memory','32g',
        '-v',str(runtime)+':/task','-v',str(model)+':/task/tokenizer:ro','-e','TMPDIR=/task/client-tmp','--entrypoint','sleep',CLIENT,'9000'],check=True)
    def ownership(tag):
        text=subprocess.check_output(['npu-smi','info'],text=True);own=set();foreign=[]
        for line in text.splitlines():
            fields=[x.strip() for x in line.split('|')[1:-1]]
            if len(fields)<4 or not re.fullmatch(r'\d+\s+\d+',fields[0]) or not fields[1].isdigit():continue
            try:cg=(Path('/proc')/fields[1]/'cgroup').read_text()
            except FileNotFoundError:continue
            if server['Id'] in cg:own.add(int(fields[1]))
            else:foreign.append(int(fields[1]))
        with (out/'ownership.jsonl').open('a') as f:f.write(json.dumps({'tag':tag,'own':len(own),'foreign':foreign})+'\n')
        assert not foreign and len(own)>=16,'NPU overlap or missing model ranks; reject measurement'
    process=None;start=time.monotonic()
    try:
        (runtime/'client-tmp').mkdir(exist_ok=True)
        for n,phase,count in phases:
            assert time.monotonic()-start<3600
            ownership(f'{n}_{phase}_before')
            if phase=='formal':sg_gate(after_warm=True)
            flush=api('/flush_cache?timeout=60' if engine=='sglang' else '/reset_prefix_cache','POST')
            assert b'Cache flushed' in flush if engine=='sglang' else flush.strip().lower()!=b'false'
            (out/f'{n}_{phase}_flush.txt').write_bytes(flush)
            cmd=command(client,n,phase,count,run);(out/f'{n}_{phase}_command.json').write_text(json.dumps(cmd,indent=2)+'\n')
            with (out/f'{n}_{phase}_cli.log').open('w') as log:
                process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);deadline=min(start+3600,time.monotonic()+1240);last=time.monotonic()
                while process.poll() is None:
                    assert time.monotonic()<deadline,'Bound exceeded; preserve output and review'
                    if time.monotonic()-last>=30:ownership(f'{n}_{phase}_during');last=time.monotonic()
                    time.sleep(2)
                assert process.returncode==0,'Inspect original output; no automatic retry'
            ownership(f'{n}_{phase}_after')
            files=list((out/f'{phase}_{n}').rglob('*_details.jsonl'));assert len(files)==1
            rows=[json.loads(x) for x in files[0].read_text().splitlines() if x.strip()]
            assert len(rows)==count and all(r['success'] is True and r['input_tokens']==n and r['output_tokens']==1024 for r in rows)
        (out/'server_info_after.json').write_bytes(api('/server_info' if engine=='sglang' else '/server_info?config_format=json'))
        (out/'runner_terminal.json').write_text(json.dumps({'validated':False,'status':'CLI_and_token_checks_passed_pending_raw_SQLite_source_and_scoring_acceptance'}))
    finally:
        obj=inspect(client);assert obj['Config']['Labels']['archive.run']==run
        subprocess.run(['docker','stop','--time','20',client],check=True,timeout=60)
        if process is not None and process.poll() is None:
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait()
if __name__=='__main__':main()
