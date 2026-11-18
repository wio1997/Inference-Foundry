"""One original strict full quality run after live pair and diagnostic drain gates."""
from pathlib import Path
import datetime,hashlib,json,re,socket,subprocess,time,urllib.request
assert socket.gethostname()=='<ROLE_HOSTNAME>'
ROOT=Path('/PRIVATE_TASK_ROOT');RUN=ROOT/'sg_pd_g941';OUT=ROOT/'client_ds41_g503/quality_sg941_g958'
assert OUT.is_dir() and not (OUT/'actual_launch_handle.json').exists() and not (OUT/'sglang941_full_test').exists()
parent=json.loads((OUT/'parent_functional_assessment.json').read_text())
assert parent['all8_target_and_draft_native_load_passed'] and parent['all8_target_full_graph_capture'] and parent['all8_draft_full_graph_capture']
assert parent['all8_actual_device_paged_kernel_and_selectedK_mapper_verified'] and parent['SSE_usage']['completion_tokens']==1024 and parent['SSE_DONE']
assert parent['old_nonpaged_FP4_kernels_still_present'],'Retain observed remaining C1 path; no global elimination claim'
guard=json.loads((OUT/'P_guard_proof.json').read_text());assert guard['guard_actual_live'] and guard['canonical_fd_verified'] and not guard['functional_actual_live']
g=guard['guard'];assert g['phase']=='holding' and g['pid']==2195941 and g['D_container_id']=='230668e294845331c75d7209317a87f774c717e289b4ad8bea3f10994be310c6'
age=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(guard['at'])).total_seconds();assert 0<=age<1200
s=json.loads((RUN/'state.json').read_text());assert s['generation']==941 and s['pid']==2519329 and s['container_id']==g['D_container_id'];assert str(RUN/'node_controller.py').encode() in Path('/proc',str(s['pid']),'cmdline').read_bytes()
for i in range(8):
    v=json.loads(Path('/root/.codex/coord',f'<ROLE_HOSTNAME>-npu{i}.owner.json').read_text());assert v['state']=='claimed' and v['pid']==s['pid'] and v['run_directory']==str(RUN)
model=json.loads(subprocess.check_output(['docker','inspect',s['container_id']]))[0];assert model['State']['Running'] and model['Image']=='sha256:7da963869ee54984dfeea2445befdf181710ef846448466db2361247759904ec'
top=subprocess.check_output(['docker','top',s['container_id'],'-eo','pid,args'],text=True);assert not any(k in top for k in ['msprof','parse.py','analyse('])
freeze=RUN/'functional_mechanism_frozen_g957.json';assert hashlib.sha256(freeze.read_bytes()).hexdigest()==parent['profile_sha_freeze']['sha256']
for f in json.loads(freeze.read_text())['files']:
    st=(RUN/'profiles/functional_mechanism_g946'/f['relative']).stat();assert st.st_size==f['bytes'] and st.st_mtime_ns==f['mtime_ns']
c=json.loads(subprocess.check_output(['docker','inspect','yxy-ds41-aisbench-g503']))[0]
assert c['Id']=='900c904de00e543644df6916e089e8b892d9bc78c36f61c438571831feb3c26a' and c['State']['Running'] and not c['HostConfig']['Devices']
assert c['HostConfig']['CpusetCpus']=='184-187' and c['Image']=='sha256:05358d58d23f2679c71e48d93797d5d2a28d72eedc7fb4900479c89955465d76'
client_age=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(c['State']['StartedAt'][:26]+'+00:00')).total_seconds();assert 72000-client_age>8500,'CPU client lease insufficient'
client_top=subprocess.check_output(['docker','top',c['Id'],'-eo','pid,args'],text=True);assert not any(k in client_top for k in ['ais_bench','run_full_gsm8k.py'])
source=OUT/'run_full_gsm8k.py';assert hashlib.sha256(source.read_bytes()).hexdigest()=='7585665108280260a325bfab4b945cdea9c809815266fac5d3ef01b3b5b6e54c'
for role,base in [('p','http://<ROLE_HOST>:19093'),('d','http://127.0.0.1:19094')]:
    for path in ['/ready','/v1/models','/metrics']:
        with urllib.request.urlopen(base+path,timeout=15) as f:raw=f.read();assert f.status==200
        (OUT/(role+'_before_'+path.strip('/').replace('/','_')+'.txt')).write_bytes(raw)
        if path=='/v1/models':assert json.loads(raw)['data'][0]['id']=='ds41flash'
        if path=='/metrics':
            for key in ['sglang:num_running_reqs','sglang:num_queue_reqs']:
                vals=[float(v) for v in re.findall(r'^'+re.escape(key)+r'(?:\{[^\n]*\})?\s+([0-9.eE+\-]+)',raw.decode(),re.M)];assert vals and sum(vals)==0
argv=['docker','exec',c['Id'],'timeout','--signal=TERM','--kill-after=30','8040','python3','-u','/task/client_ds41_g503/quality_sg941_g958/run_full_gsm8k.py']
with (OUT/'quality.log').open('x') as log:child=subprocess.Popen(argv,cwd=OUT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
h={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':child.pid,'argv':argv,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'client_container_id':c['Id'],'SG_D_container_id':s['container_id'],'P_guard_pid':g['pid'],'inner_hard_bound_s':8070,'suite_alarm_s':7200,'scope':'Fullsameofficial1319question/prompt/grader/T0/seed0/max2048/C16/thinkingfalse/600srequesttimeout/no retry. ActualC2FullTopKpagedcandidateonly; C1oldnonpagedpathremains. NoAISBench/finalmatchedprofile/E2Egainclaim.'}
(OUT/'actual_launch_handle.json').write_text(json.dumps(h,indent=2)+'\n');time.sleep(2);assert child.poll() is None,('Immediate quality exit',child.returncode)
print('G958_MATCHED_SG941_FULL_QUALITY_START_JSON '+json.dumps(h),flush=True)
