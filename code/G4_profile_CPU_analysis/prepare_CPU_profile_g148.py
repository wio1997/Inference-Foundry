"""Run via stdin on either current G4 serving host; immutable raw copied for CPU analysis."""
from pathlib import Path
import base64,datetime,hashlib,json,os,re,socket,subprocess,time,shutil
host=socket.gethostname();assert host in ['<ROLE_NODE>','<ROLE_NODE>']
role='p' if host.endswith('0001') else 'd'
r=Path('/data' if role=='p' else '/data1')/'tiankuan/yxy/glm53_pd_align_910b_20261116'
q=r/'vllm_G4_CPU_profile_g148';q.mkdir(exist_ok=False);profile=r/f'vllm_g4/profiles/{role}'
original_profile=profile
selected=[p for p in original_profile.glob('*ascend_pt') if p.name not in PRIOR_NAMES[role]]
assert len(selected)==8,'Need exactly8new native captures beyond preserved G118'
profile=q/'frozen_raw';profile.mkdir()
for p in selected:shutil.copytree(p,profile/p.name)
name=f'yxy-glm53-vllm-{role}-g4';i=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
expected={'p':'3dfb97a128c0a7a8cdaa92e468dd771dd72412120ea5d0223a506a361a277eac','d':'ec3db29490528584b428ff7894f0496f41bb44ec5cff5dd45f04d93649a8a106'}[role]
assert i['State']['Running'] and i['Id']==expected
assert i['Image']=='sha256:4d683043e4bdb7056bf69c2064acbd535d7c9587ccf76c14f70c467e4c24e823'
top=subprocess.check_output(['docker','top',name,'-eo','pid,args'],text=True)
rows={int(line.split()[0]):line for line in top.splitlines()[1:]}
server=[pid for pid,line in rows.items() if 'vllm serve /models/glm53' in line];assert len(server)==1
argv=Path('/proc',str(server[0]),'cmdline').read_bytes().decode().rstrip('\0').split('\0')
workers=re.findall(r'\|\s*(\d+)\s+0\s*\|\s*(\d+)\s*\|\s*VLLMWorker',subprocess.check_output(['npu-smi','info'],text=True))
assert len(workers)==8;mapping=[]
for physical,pid in workers:
 pid=int(pid);m=re.search(r'Worker_TP(\d+)_EP(\d+)',rows[pid]);assert m
 status=Path('/proc',str(pid),'status').read_text();nspids=next([int(x) for x in line.split()[1:]] for line in status.splitlines() if line.startswith('NSpid:'))
 mapping.append({'tp_rank':int(m[1]),'ep_rank':int(m[2]),'physical_npu':int(physical),'host_pid':pid,'namespace_pids':nspids})
assert {x['tp_rank'] for x in mapping}=={x['physical_npu'] for x in mapping}==set(range(8))
identity={'role':role,'container_id':i['Id'],'image_id':i['Image'],'started_at':i['State']['StartedAt'],
          'launch_argv_sha256':hashlib.sha256(json.dumps(argv).encode()).hexdigest(),'worker_mapping':sorted(mapping,key=lambda x:x['tp_rank'])}
captures=list(profile.glob('*ascend_pt'));assert len(captures)==8
for capture in captures:
 m=re.search(r'_rank(\d+)_(\d+)_\d+_ascend_pt$',capture.name);assert m
 rank,pid=map(int,m.groups());assert pid==next(x for x in mapping if x['tp_rank']==rank)['namespace_pids'][-1]
def manifest():
 out={}
 for p in sorted(profile.rglob('*')):
  if not p.is_file():continue
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
  out[str(p.relative_to(profile))]={'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':h.hexdigest()}
 return out
first=manifest();first_at=datetime.datetime.now(datetime.timezone.utc).isoformat();time.sleep(2);second=manifest();assert first==second,'Profiler output still changing; preserve observation and inspect same epoch'
source_manifest={}
for directory in selected:
 for p in directory.rglob('*'):
  if not p.is_file():continue
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
  source_manifest[str(p.relative_to(original_profile))]={'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':h.hexdigest()}
assert source_manifest==second,'Source native captures changed during immutable copy'
stable={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'first_observed_at':first_at,'role':role,'identity':identity,
        'profile_root_container':'/task/vllm_G4_CPU_profile_g148/frozen_raw','original_profile_root':str(original_profile),'selected_capture_names':[p.name for p in selected],'profile_root_host':str(profile),'files':second,
        'two_actual_SHA_metadata_observations_equal':True,'raw_bytes':sum(x['bytes'] for x in second.values()),'raw_files':len(second)}
(q/'identity.json').write_text(json.dumps(identity,indent=2));(q/f'raw_stability_{role}.json').write_text(json.dumps(stable,indent=2))
package=q/'package';package.mkdir()
for filename,item in files.items():
 b=base64.b64decode(item['b64']);assert hashlib.sha256(b).hexdigest()==item['sha256'];(package/filename).write_bytes(b)
croot='/task/vllm_G4_CPU_profile_g148'
commands=[['python3',croot+'/package/parallel_profile_archive.py','--root',stable['profile_root_container'],'--identity',croot+'/identity.json','--stability-receipt',croot+f'/raw_stability_{role}.json','--out',croot+'/analysis_manifest.json','--work-root',croot+'/analysis_copies','--analyse'],
          ['python3',croot+'/package/analyse_role.py','--root',croot,'--role',role]]
job="""from pathlib import Path
import subprocess,json,os,datetime
r=Path(ROOT)
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[-2:]));os.nice(10)
for index,c in enumerate(COMMANDS):
 result=subprocess.run(['docker','exec','-e','TORCH_DEVICE_BACKEND_AUTOLOAD=0',CONTAINER]+c,cwd=r)
 (r/('stage_'+str(index)+'_returncode.json')).write_text(json.dumps({'rc':result.returncode,'argv':c,'at':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
 assert result.returncode==0
""".replace('ROOT',repr(str(q))).replace('COMMANDS',repr(commands)).replace('CONTAINER',repr(name))
(q/'job.py').write_text(job);args=['timeout','--signal=TERM','--kill-after=60','1800','python3','-u',str(q/'job.py')]
with (q/'CPU_analysis.log').open('x') as log:p=subprocess.Popen(args,cwd=q,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
handle={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'role':role,'pid':p.pid,'argv':args,'scope':'CPU SDK analysis on copied immutable G4 raw;2 bounded rank workers. OriginalrawSHA retained; no model requests or new NPU workload.'}
(q/'CPU_analysis_handle.json').write_text(json.dumps(handle,indent=2));time.sleep(2)
print('G148_CPU_ANALYSIS_START_JSON '+json.dumps({'handle':handle,'actual_live':p.poll() is None,'raw_files':len(second),'raw_bytes':stable['raw_bytes'],'all8_actual_rank_namespacePID_device_identity_verified':True}),flush=True)
print('__G148_CPU_ANALYSIS_START_DONE__',flush=True)
