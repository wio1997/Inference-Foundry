python3 - <<'PY'
from pathlib import Path
import datetime,hashlib,json,socket,subprocess
assert socket.gethostname()=='<ROLE_NODE>'
root=Path('<TASK_ROOT>');r=root/'sg_pd_g316';old=root/'sg_pd_g181'
h=json.loads((r/'actual_launch_handle.json').read_text());assert h['pid']==3434089 and str(r/'node_controller.py').encode() in Path('/proc',str(h['pid']),'cmdline').read_bytes()
info=json.loads(subprocess.check_output(['docker','inspect','yxy-glm53-sg-d-g316']))[0];assert info['Id']=='e0d00beb0583f3b1afaf77d3a019a7c680c4474101fbe062d5010fd1244069ba' and info['State']['Running'];assert info['Config']['Labels']['codex.run']==r.name
env=dict(x.split('=',1) for x in info['Config']['Env'] if '=' in x)
paths=['cache/triton','cache/ascend','cache/torch_extensions','home/.cache/vllm','home/.cache/torch','tmp/torchinductor_root']
def snapshot(base):
 return {str(p.relative_to(base)):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for q in paths for p in (base/q).rglob('*') if p.is_file() and not p.name.endswith('.lock')}
c=root/'client_vllm_g42';run=c/'runs/sglang/SG_G316_C16_256K_g404';rh=json.loads((c/'SG_G316_256K_resume_g445_handle.json').read_text());proc=Path('/proc',str(rh['pid']),'cmdline');assert proc.exists() and b'run_aisbench_cold_hot.py' in proc.read_bytes();state=json.loads((run/'state.json').read_text());assert state['subphase']=='warmup3' and not (run/'warmup4_262144_command.json').exists(),'Must precede full warm4 and5'
runner=root/'benchmark_pipeline_sg_g199/run_aisbench_cold_hot.py';text=runner.read_text();assert "base=Path('<TASK_ROOT>/sg_pd_g181')" in text
d={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller_handle':h,'container_id':info['Id'],'actual_runner_pid':rh['pid'],'actual_runner_live':True,'state':state,'actual_runner_sha256':hashlib.sha256(runner.read_bytes()).hexdigest(),'actual_compile_environment':{k:v for k,v in env.items() if any(t in k for t in ['CACHE','TMP','HOME'])},'actual_mounts':[x for x in info['Mounts'] if x['Destination'].startswith('/task')],'current_root':str(r),'old_root':str(old),'root_aliases':{q:{'old_resolved':str((old/q).resolve()),'current_resolved':str((r/q).resolve()),'same_path':(old/q).resolve()==(r/q).resolve()} for q in paths},'current_files':snapshot(r),'old_files':snapshot(old),'scope':'Actual current D independent compiler-cache baseline during warm3 and strictly before full warm4+5; original runner old-root check preserved, not valid current-D compilation acceptance unless aliases prove same.'}
p=r/'D_cache_during_256K_warm3_g483.json';assert not p.exists();p.write_text(json.dumps(d,indent=2)+'\n');print('G483_CURRENT_D_CACHE_JSON '+json.dumps(d),flush=True)
PY
