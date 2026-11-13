"""166 G31 head16 + G32 mHC4096 + G33 Engram1024, globalprefill49152, HCCL1024, fullKV/context."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,re,shlex,signal,socket,subprocess,time,traceback
R=Path('/data/sglang-dsv41')
P=R/'sglang_ascops37';MODEL=Path('/data/models/DeepSeek-V4.1-Flash-1017-w8a8')
IMAGE='sha256:08367522c4f9803fb876808c2cd4afe459e7e2b0adaab5a0daf36061edf8e517'
NAME='yxy-v41-align-sglang-166-ascops37';PORTS=[18167,18280,14699,*range(18400,18407)]
CONTROL=Path('/data/shared-npu-pool/.control')
def main():
    assert socket.gethostname()=='SERVER_HOSTNAME_REDACTED'
    P.mkdir(exist_ok=False);locks=[];proc=None;started=False;stop=False
    def stamp(phase,**kw):
        d={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':phase,'pid':os.getpid(),'container':NAME,**kw}
        f=P/'state.json';tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(f)
        print(json.dumps(d,ensure_ascii=False),flush=True)
    def policy():
        d=json.loads((CONTROL/'npu-pool-policy.json').read_text())
        if d.get('leases'):raise RuntimeError('Foreign leases remain; no device takeover: '+json.dumps(d['leases']))
        return d
    def inspect():
        q=subprocess.run(['docker','inspect',NAME],capture_output=True,text=True,timeout=20)
        if q.returncode:return None
        return json.loads(q.stdout)[0]
    def on_signal(*_):nonlocal stop;stop=True
    signal.signal(signal.SIGINT,on_signal);signal.signal(signal.SIGTERM,on_signal)
    try:
        suite=json.loads((R/'probes/sglang_G32_suite/stage.json').read_text())
        assert suite['phase']=='all_three_component_contracts_passed' and suite['target']=='166'
        for rel,sha in suite['files'].items():assert hashlib.sha256((R/rel).read_bytes()).hexdigest()==sha,rel
        state=json.loads((R/'probes/sglang_G32_suite/state.json').read_text())
        assert state['phase']=='own_container_terminal_locks_releasing' and state['exit_code']==0
        cmd=Path('/proc',str(int((R/'sglang_G32suite_guard.pid').read_text())),'cmdline')
        assert not cmd.exists() or b'sglang_G32suite_guard.py' not in cmd.read_bytes()
        engram=json.loads((R/'probes/engram_gate_g33/result.json').read_text())
        assert engram['status']=='passed_engram_FP32_row_chunk_contract' and all(engram['checks'].values())
        assert all(c['BF16_output_bitwise_exact'] for c in engram['cases'])
        assert engram['sources']['candidate']==hashlib.sha256((R/'compat/runtime/g33/engram_npu_uva.py').read_bytes()).hexdigest()
        prior=json.loads((R/'probes/engram_gate_g33/state.json').read_text())
        assert prior['phase']=='own_container_terminal_locks_releasing' and prior['exit_code']==0
        cmd=Path('/proc',str(int((R/'engram_G33_probe_guard.pid').read_text())),'cmdline')
        assert not cmd.exists() or b'engram_G33_probe_guard.py' not in cmd.read_bytes()
        proof_root=R/'probes/ascend_ops_g35'
        proof=json.loads((proof_root/'state.json').read_text())
        assert proof['phase']=='own_container_terminal_locks_releasing' and proof['exit_code']==0
        rank_proofs=[json.loads(p.read_text()) for p in proof_root.glob('rank_*.json')]
        assert len(rank_proofs)==16 and {x['rank'] for x in rank_proofs}==set(range(16))
        assert all(x['status']=='passed_stock_AscendOps_transport_and_graph_contract' and all(x['checks'].values()) and x['configuration']['dispatch_padding']==0 for x in rank_proofs)
        for pidfile,needle in [('ascend_ops_G35_probe_guard.pid',b'ascend_ops_G35_probe_guard.py'),('sglang_G33model_guard.pid',b'sglang_G33model_guard.py')]:
            cmd=Path('/proc',str(int((R/pidfile).read_text())),'cmdline')
            assert not cmd.exists() or needle not in cmd.read_bytes()
        previous=json.loads(subprocess.check_output(['docker','inspect','yxy-v41-align-sglang-166-ascops35'],text=True))[0]
        assert previous['Id']=='4c831a86a074d7412fcadeb9c920deaf55019bf223480c74d107cd56a529bccd' and not previous['State']['Running'] and previous['Config']['Labels']['codex.task']==R.name
        cmd=Path('/proc',str(int((R/'sglang_G35model_guard.pid').read_text())),'cmdline')
        assert not cmd.exists() or b'sglang_G35model_guard.py' not in cmd.read_bytes()
        run=json.loads((R/'aisbench_run_state_sglang_ascops35_full.json').read_text())
        assert run['phase']=='stopped_require_review'
        cmd=Path('/proc',str(int((R/'aisbench_sglang_ascops35_full_runner.pid').read_text())),'cmdline')
        assert not cmd.exists() or b'run_sglang_ascops35_profiles.py' not in cmd.read_bytes()
        markov=json.loads((R/'probes/markov_g37/result.json').read_text())
        assert markov['status']=='passed_real_checkpoint_mapping_and_fail_closed_CPU_contract' and markov['actual_index_keys']==143324
        for rel,sha in markov['sources'].items():assert hashlib.sha256((R/rel).read_bytes()).hexdigest()==sha,rel
        previous=json.loads(subprocess.check_output(['docker','inspect','yxy-v41-align-sglang-166-ascops36'],text=True))[0]
        assert previous['Id']=='f8765ed1f2342e401a32e690848f38af3b006eff24a4b96dc39eb345fb7f2f3f' and not previous['State']['Running'] and previous['Config']['Labels']['codex.task']==R.name
        for pidfile,needle in [('sglang_G36model_guard.pid',b'sglang_G36model_guard.py'),('aisbench_sglang_ascops36_full_runner.pid',b'run_sglang_ascops36_profiles.py')]:
            cmd=Path('/proc',str(int((R/pidfile).read_text())),'cmdline')
            assert not cmd.exists() or needle not in cmd.read_bytes()
        policy()
        for root,values,pattern in [(CONTROL/'pool-device-locks',range(16),'{}'),(CONTROL/'locks',range(16),'device-{}.lock'),(Path('/data/shared-npu-pool/port-locks'),PORTS,'sglang-port-{}.lock')]:
            assert root.is_dir(),str(root)
            for i in values:
                f=(root/pattern.format(i)).open('a')
                try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BaseException:f.close();raise
                locks.append(f)
        pol=policy();assert inspect() is None,'Existing own name requires review; never overwrite'
        image=json.loads(subprocess.check_output(['docker','image','inspect',IMAGE]))[0]
        assert image['Id']==IMAGE and image['Architecture']=='arm64'
        n=subprocess.check_output(['npu-smi','info'],text=True,timeout=30);(P/'npu_before.txt').write_text(n)
        assert all(f'No running processes found in NPU {i}' in n for i in range(8)),'Foreign computation'
        rows=[]
        for line in n.splitlines():
            f=[v.strip() for v in line.split('|')[1:-1]]
            if len(f)==3 and re.fullmatch(r'\d+\s+\d+',f[0]) and ':' in f[1]:
                vals=re.findall(r'(\d+)\s*/\s*(\d+)',f[2])
                if vals:rows.append(tuple(map(int,vals[-1])))
        assert len(rows)==16 and all(total-used>=53*1024 for used,total in rows),'HBM preflight insufficient'
        expected=json.loads((R/'configs/weight_control_hashes.json').read_text())
        for name,item in expected.items():assert hashlib.sha256((MODEL/name).read_bytes()).hexdigest()==item['expected'],name
        for port in PORTS:
            with socket.socket() as s:s.bind(('127.0.0.1',port))
        proof_root=R/'probes/deepep_n3k_g2'
        ranks=[json.loads(p.read_text()) for p in proof_root.glob('rank_*.json')]
        assert len(ranks)==16 and all(x['status']=='passed_identity_roundtrip_and_graph' and x['configuration']['capacity']==32 and x['configuration']['tokens']==24 and x['configuration']['HCCL_BUFFSIZE']=='1024' and x['checks']['normal_T3072_INT8_dispatch_BF16_combine_identity_exact'] and all(x['checks'].values()) for x in ranks)
        mhc=json.loads((R/'probes/mhc_chunk/result.json').read_text())
        assert mhc['status']=='passed_mhc_row_chunk_component_contract' and all(mhc['checks'].values())
        assert mhc['sources']['candidate']=='cb64efc1c76453d94f80186144dee5e666fb148369b3c05173e36d823d29be15'
        head=json.loads((R/'probes/attention_heads16_g3/result.json').read_text())
        assert head['status']=='six_head16_eager_graph_and_changed_input_contracts_passed'
        manifest=json.loads((R/'configs/sglang_G37_runtime_manifest.json').read_text())
        for rel,sha in manifest['files'].items():assert hashlib.sha256((R/rel).read_bytes()).hexdigest()==sha,rel
        raw=json.loads((MODEL/'quant_model_description.json').read_text())
        pairs={k:v for k,v in raw.items() if k.startswith('layers.') and (k.endswith('.attn.wq_a.weight') or k.endswith('.attn.wkv.weight'))}
        assert len(pairs)==80 and set(pairs.values())=={'FLOAT'},'Native QKV fusion requires80FLOAT target projections'
        for key in ['home','tmp','logs','ascend','cache']:(R/'serve_ascops37'/key).mkdir(parents=True,exist_ok=True)
        (R/'compat/probe_g5/home/.cache/torch_extensions/py312_cpu').mkdir(parents=True,exist_ok=True)
        argv=json.loads((R/'configs/sglang_G37model_docker_template.json').read_text())
        oldroot='/data/sglang-dsv41'
        oldmodel='/data/models/DeepSeek-V4.1-Flash-1017-w8a8'
        for i,value in enumerate(argv):
            if i and argv[i-1]=='--name':argv[i]=NAME
            elif i and argv[i-1]=='--label':argv[i]='codex.task='+R.name
            elif i and argv[i-1]=='-v':
                host,rest=value.split(':',1)
                if host.startswith(oldroot):host=str(R)+host[len(oldroot):]
                elif host==oldmodel:host=str(MODEL)
                assert Path(host).exists(),host
                argv[i]=host+':'+rest
        assert IMAGE in argv
        (P/'docker_command.json').write_text(json.dumps(argv,indent=2)+'\n');(P/'policy_at_claim.json').write_text(json.dumps(pol,indent=2)+'\n')
        stamp('all16devices_claimed_launching',image_id=IMAGE,resource_lock_count=len(locks),serving_ready=False)
        with (P/'server.log').open('x') as log:
            proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT);started=True;begin=time.monotonic()
            stamp('adapted_sglang_running_initialization_unverified',docker_client_pid=proc.pid,serving_ready=False,lifetime_seconds=21600)
            while proc.poll() is None and not stop and time.monotonic()-begin<21600:time.sleep(2)
            if proc.poll() is None:
                d=inspect();assert d and d['Config']['Labels']['codex.task']==R.name
                subprocess.run(['docker','stop','--time','30',NAME],check=True,timeout=65);proc.wait(timeout=65)
            stamp('own_container_terminal_locks_releasing',exit_code=proc.returncode,stopped_by_signal=stop,bound_reached=time.monotonic()-begin>=21600)
    except BaseException:
        stamp('failed_require_review',error=traceback.format_exc(),serving_ready=False)
        if started and proc and proc.poll() is None:
            d=inspect();assert d and d['Config']['Labels']['codex.task']==R.name
            subprocess.run(['docker','stop','--time','30',NAME],check=True,timeout=65);proc.wait(timeout=65)
        raise
    finally:
        for f in reversed(locks):f.close()
if __name__=='__main__':main()
