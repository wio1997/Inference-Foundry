python3 - <<'PY'
from pathlib import Path
import json,socket,subprocess,datetime,hashlib,tarfile,io,base64
assert socket.gethostname()=='<ROLE_NODE>'
root=Path('<TASK_ROOT>');r=root/'sg_pd_g316';c=root/'client_vllm_g42';run=c/'runs/sglang/SG_G316_C16_256K_g404'
h=json.loads((c/'SG_G316_256K_resume_g445_handle.json').read_text());p=Path('/proc',str(h['pid']),'cmdline');assert not p.exists() or b'run_aisbench_cold_hot.py' not in p.read_bytes(),'Actual warm runner still active; preserve it'
state=json.loads((run/'state.json').read_text());assert state['phase']=='warmup_diagnostics_complete_pending_compile_audit'
assert not list(run.glob('cold_*')) and not list(run.glob('hot_*'))
info=json.loads(subprocess.check_output(['docker','inspect','yxy-glm53-sg-d-g316']))[0];assert info['Id']=='e0d00beb0583f3b1afaf77d3a019a7c680c4474101fbe062d5010fd1244069ba' and info['State']['Running']
env=dict(x.split('=',1) for x in info['Config']['Env'] if '=' in x)
for key,suffix in [('HOME','home'),('TMPDIR','tmp'),('ASCEND_CACHE_PATH','cache/ascend'),('TRITON_CACHE_DIR','cache/triton')]:assert env[key]=='/task/sg_pd_g316/'+suffix
assert any(x['Source']==str(r) and x['Destination']=='/task/sg_pd_g316' and x['RW'] for x in info['Mounts'])
paths=['cache/triton','cache/ascend','cache/torch_extensions','home/.cache/vllm','home/.cache/torch','tmp/torchinductor_root']
files={str(p.relative_to(r)):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for q in paths for p in (r/q).rglob('*') if p.is_file() and not p.name.endswith('.lock')}
baseline=r/'D_cache_during_256K_warm3_g483.json';before=json.loads(baseline.read_text());assert before['container_id']==info['Id'];assert before['current_files']==files,'Active D compilation changed during complete warm4+5'
start4=json.loads((run/'warmup4_262144_command.json').read_text())['at'];assert datetime.datetime.fromisoformat(before['at'])<datetime.datetime.fromisoformat(start4)
audits={}
for i in range(1,6):
 d=json.loads((run/f'warmup{i}_262144_audit.json').read_text());assert d['fully_validated'] and d['requests']==d['successful_requests']==16 and d['input_tokens']==262144 and d['output_tokens']==1024;audits[str(i)]=d
stability={k:abs(audits['5'][k]-audits['4'][k])/max(abs(audits['4'][k]),1e-12) for k in ['duration_s','ttft_mean_ms','tpot_mean_ms']};assert all(stability[k]<=.05 for k in ['duration_s','ttft_mean_ms'])
final=r/'D_cache_after_256K_warm5_g496.json';assert not final.exists();d={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'container_id':info['Id'],'current_files':files,'actual_compile_environment':{k:env[k] for k in ['HOME','TMPDIR','ASCEND_CACHE_PATH','TRITON_CACHE_DIR']},'actual_mounts':[m for m in info['Mounts'] if m['Destination'].startswith('/task')],'state':state,'scope':'Actual current-D independent final snapshot after all five warmups, compared with G483 before full warm4+5. No formal request submitted.'};final.write_text(json.dumps(d,indent=2)+'\n')
after=run/'after_warmup5_262144_server_snapshot.json';v=json.loads(after.read_text());latest=max(q['at'] for role,x in v.items() if not role.startswith('_') for q in x.values() if isinstance(q,dict) and 'at' in q)
proof={'run_id':'SG_G316_C16_256K_g404','length':262144,'original_manifest_sha256':hashlib.sha256((run/'run_manifest.json').read_bytes()).hexdigest(),'active_D_container_id':info['Id'],'baseline_path':str(baseline),'baseline_sha256':hashlib.sha256(baseline.read_bytes()).hexdigest(),'final_path':str(final),'final_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),'after_warm5_snapshot_sha256':hashlib.sha256(after.read_bytes()).hexdigest(),'warm5_after_at':latest,'timing_relative_changes':stability,'full_warm4_and5_active_D_cache_equal':True,'formal_submitted':False}
proofpath=r/'D_256K_warm_current_cache_proof_g496.json';assert not proofpath.exists();proofpath.write_text(json.dumps(proof,indent=2)+'\n')
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w:gz') as t:
 for p in run.rglob('*'):
  if p.is_file():t.add(p,arcname='run/'+str(p.relative_to(run)),recursive=False)
 for p in [baseline,final,proofpath]:t.add(p,arcname='corrected_current_D/'+p.name,recursive=False)
 for p in (root/'benchmark_pipeline_sg_g488').iterdir():
  if p.is_file():t.add(p,arcname='corrected_runner/'+p.name,recursive=False)
raw=b.getvalue();print('G496_ALLWARM_CURRENT_D_JSON '+json.dumps({'at':d['at'],'state':state,'audits':audits,'current_D_proof':proof,'current_D_proof_sha256':hashlib.sha256(proofpath.read_bytes()).hexdigest(),'archive_sha256':hashlib.sha256(raw).hexdigest(),'archive_b64':base64.b64encode(raw).decode()}),flush=True)
PY
