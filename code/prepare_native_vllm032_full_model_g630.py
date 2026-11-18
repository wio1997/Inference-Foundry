from pathlib import Path
import ast,base64,hashlib,json,re,shutil
src=Path('vllm_pd_g624_preparation');dst=Path('vllm_pd_g630');assert not dst.exists();shutil.copytree(src,dst)
IMAGE='sha256:d5492ecfdb3071ff2d4bae3f849245014c54b63035d9eefb35243869461ddae1'
for port in [2203,2204]:
 d=json.loads(Path(f'G628_actual_vllm032_image_audit_{port}_parsed.json').read_text());assert d['result']['scope_passed'] and d['CPU_owner_released']
for role in ['p','d']:
 p=dst/f'candidate_{role}.json';d=json.loads(p.read_text());d['image_id']=IMAGE;d['status']='NATIVE032_PREPARED_FULL_MODEL_NOT_YET_STARTED';d['revision']='G630 actualarm64 d5492ec source113/HEADmatches; native deployment only, allruntime/fullgraph/PD/quality/performance unverified';d['environment']['MOONCAKE_CONFIG_PATH']=f'/task/vllm_pd_g630/mooncake_{role}.json';a=d['argv'];i=a.index('--kv-transfer-config')+1;x=json.loads(a[i]);x['engine_id']='ds41-vllm-g630-'+role;a[i]=json.dumps(x,separators=(',',':'))
 i=a.index('--additional-config')+1;x=json.loads(a[i]);x['enable_cpu_binding']=True
 if role=='d':x.update(ascend_compilation_config={'enable_npugraph_ex':True,'enable_static_kernel':False},multistream_overlap_shared_expert=True)
 a[i]=json.dumps(x,separators=(',',':'));p.write_text(json.dumps(d,indent=2)+'\n')
source=json.loads(Path('image_refresh_g612/user_reference/adaptation/source-manifest.json').read_text());expected=[{k:x[k] for k in ['path','sha256','bytes']} for x in source['files']];(dst/'actual_image_source_prerequisite.json').write_text(json.dumps({'image_id':IMAGE,'vllm_HEAD':'5df63a7203cdb9328d824a27b682e77c7b9928be','ascend_HEAD':'24487f4f48ee99cf45d86d1c504e645549b8c1ef','selected_files':expected,'source113_actual_both_hosts_verified':True,'scope':'Image/sourceidentityonly; noengine/model/PD/quality/performancepass'},indent=2)+'\n')
launch='''from pathlib import Path
import datetime,hashlib,importlib.metadata,json,os,subprocess
R=Path(__file__).resolve().parent;os.chdir(R)
c=json.loads((R/'candidate.json').read_text());p=json.loads((R/'actual_image_source_prerequisite.json').read_text());assert c['image_id']==p['image_id']
for e in p['selected_files']:
 f=Path('/vllm-workspace/vllm-ascend')/e['path'];assert f.stat().st_size==e['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==e['sha256'],str(f)
assert hashlib.sha256(Path('/models/ds41/config.json').read_bytes()).hexdigest()=='43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7'
versions={n:importlib.metadata.version(n) for n in ['vllm','vllm-ascend','torch','torch-npu']};assert versions['vllm-ascend']=='0.32.0rc1' and versions['torch-npu']=='2.10.0.post7'
os.environ.update(c['environment']);os.environ['PYTORCH_NPU_ALLOC_CONF']='expandable_segments:True';os.environ['TORCHINDUCTOR_CACHE_DIR']=str(R/'cache/inductor');os.environ['TORCHAIR_CACHE_HOME']=str(R/'cache/torchair')
(R/'actual_native_launch.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':c['argv'],'environment':c['environment'],'versions':versions,'all113_actual_source_pins_match':True,'native_source_overlay':False,'runtime_validated':False},indent=2)+'\\n')
with (R/'native_cli_help.log').open('x') as log:r=subprocess.run(['vllm','serve','--help=all'],cwd=R,stdout=log,stderr=subprocess.STDOUT,timeout=180)
assert r.returncode==0,('NativeCLIhelp/importfailed beforemodelstartup',r.returncode)
os.execvpe(c['argv'][0],c['argv'],os.environ)
''';ast.parse(launch);(dst/'launch_pd.py').write_text(launch);(dst/'run_pd.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd /task/vllm_pd_g630\nexec python3 -u /task/vllm_pd_g630/launch_pd.py\n')
s=Path('sg_pd_g603/node_controller.py').read_text().replace("NAME='yxy-ds41-sg-'+ROLE+'-g603'","NAME='yxy-ds41-vllm-'+ROLE+'-g630'").replace('generation=603','generation=630').replace('sg_pd_g603','vllm_pd_g630').replace('sha256:7da963869ee54984dfeea2445befdf181710ef846448466db2361247759904ec',IMAGE).replace("HOST!='<ROLE_HOSTNAME>'","HOST not in ['<ROLE_HOSTNAME>','<ROLE_HOSTNAME>']").replace("PORT=19093 if ROLE=='p' else 19094","PORT=19103 if ROLE=='p' else 19104")
s=s.replace("['run_pd.sh','launch_pd.py','candidate.json','adapter.py','source_uva.py','ds41_native_gate/__init__.py','ds41_native_gate/entry.py']","['run_pd.sh','launch_pd.py','candidate.json','actual_image_source_prerequisite.json']")
s=s.replace("[PORT]+([18998,18678] if ROLE=='p' else [])","[PORT,30200 if ROLE=='p' else 30300]")
# CLI helper uses the same native package with mapped devices under all8 lifecycle locks, never on the client VM.
ast.parse(s);(dst/'node_controller.py').write_text(s)
pins={str(p.relative_to(dst)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dst.iterdir() if p.is_file()};(dst/'deployment_source_sha256.json').write_text(json.dumps(pins,indent=2)+'\n')
files={str(p.relative_to(dst)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'b64':base64.b64encode(p.read_bytes()).decode()} for p in dst.iterdir() if p.is_file()}
s='''python3 - <<'REMOTE'
from pathlib import Path
import base64,datetime,hashlib,json,shutil,socket,subprocess,time
host=socket.gethostname();assert host in ['<ROLE_HOSTNAME>','<ROLE_HOSTNAME>'];role='p' if host.endswith('0003') else 'd'
free={p:shutil.disk_usage(p).free for p in ['/data','/data1']};assert max(free,key=free.get)=='/data';root=Path('/PRIVATE_TASK_ROOT');r=root/'vllm_pd_g630';assert not r.exists()
prior=root/('sg_pd_g603' if role=='p' else 'sg_pd_g501');s=json.loads((prior/'state.json').read_text());assert s['phase'] in ['failed_released','stopped_released'] and s['cleanup']['all8_process_lists_clear'];p=Path('/proc',str(s['pid']),'cmdline');assert not p.exists() or str(prior/'node_controller.py').encode() not in p.read_bytes()
img=json.loads(subprocess.check_output(['docker','image','inspect','IMAGE']))[0];assert img['Id']=='IMAGE' and img['Architecture']=='arm64' and any('sha256:ff6d1ea57df08dd9382c51d202d0dd2d9acc35d53a65bcb0dfd906886e674912' in x for x in img['RepoDigests'])
audit=root/'vllm032_image_audit_g628/result.json';d=json.loads(audit.read_text());assert d['scope_passed'] and d['all113_source_bytes_match']
FILES
r.mkdir()
for name,item in files.items():
 raw=base64.b64decode(item['b64']);assert hashlib.sha256(raw).hexdigest()==item['sha256'];(r/name).write_bytes(raw)
(r/'candidate.json').write_bytes((r/f'candidate_{role}.json').read_bytes());(r/'storage_preflight.json').write_text(json.dumps(free,indent=2)+'\\n')
with (r/'controller.log').open('x') as f:p=subprocess.Popen(['python3','-u',str(r/'node_controller.py')],cwd=r,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(2);h={'pid':p.pid,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bound_s':72000,'role':role,'controller_sha256':hashlib.sha256((r/'node_controller.py').read_bytes()).hexdigest(),'image_id':'IMAGE','scope':'Native0.32 fullmodel candidate, all113actualsourcepins; no inherited0.30codeoverlay; fullruntime/PD/graphs/quality/performanceunverified'};(r/'actual_launch_handle.json').write_text(json.dumps(h,indent=2)+'\\n');print('G630_VLLM032_FULL_MODEL_START_JSON '+json.dumps({'handle':h,'actual_live':p.poll() is None,'state':json.loads((r/'state.json').read_text()) if (r/'state.json').exists() else None}),flush=True)
REMOTE
'''.replace('IMAGE',IMAGE).replace('FILES','files='+repr(files));Path('start_native_vllm032_full_model_g630.sh').write_text(s)
print('Preparednative0.32 only/nooverlay,commonTP8EP8PCP1DP1/P.88D.80/chunks8192/1024/DSPARK5/C16/fullDtargetdraft intent')
