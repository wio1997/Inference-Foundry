from pathlib import Path
import os,json,subprocess,re,time
p=Path('/data/tiankuan/wio/910c-deepinfer-172');version='prefill-affinity-v19';parent='prefill-kv-gather-v18';control=p/'serving'/parent;dest=p/'diagnostics'/version;dest.mkdir(exist_ok=False)
assert json.loads((p/'runs'/f'{parent}-benchmark-status.json').read_text())['state']=='PASS'
assert json.loads((p/'final'/parent/'comparison.json').read_text())['outcome']=='INCONCLUSIVE'
for job in (control/'jobs').iterdir():assert len(list(job.glob('result-dp*.json')))==2 and not list(job.glob('error-rank*.txt')),job
smi=subprocess.check_output(['npu-smi','info'],text=True);pids=[int(x) for x in re.findall(r'^\|\s*\d+\s+\d+\s*\|\s*(\d+)\s*\|',smi,re.M)];assert len(pids)==16
workers=[]
for pid in pids:
 cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode();assert 'ascend910c_w8a8_worker.py' in cmd and '/serving/prefill-kv-gather-v18' in cmd,(pid,cmd)
 env=dict(x.split('=',1) for x in Path(f'/proc/{pid}/environ').read_bytes().decode().split('\0') if '=' in x);rank=int(env['LOCAL_RANK']);assert 0<=rank<16
 allowed=sorted(os.sched_getaffinity(pid));assert allowed==list(range(640)),('unexpected existing affinity',pid,len(allowed))
 pool=allowed[rank*40:(rank+1)*40];assert len(pool)==40
 tasks=[]
 for f in sorted(Path(f'/proc/{pid}/task').iterdir()):
  tid=int(f.name)
  try:tasks.append({'tid':tid,'name':(f/'comm').read_text().strip(),'before':sorted(os.sched_getaffinity(tid))})
  except ProcessLookupError:continue
 workers.append({'rank':rank,'pid':pid,'main':pool[2:-2],'acl':[pool[-2]],'release':[pool[-1]],'irq_reserved_unchanged':pool[:2],'tasks':tasks})
assert sorted(x['rank'] for x in workers)==list(range(16))
record={'state':'PREPARED','version':version,'parent_model_runtime':parent,'reference':'Pinned Ascend A3 cpu_binding.py GLOBAL_SLICE;640CPU/16NPU,2IRQ reserved,36main,1ACL,1release','scope':'only owned worker/TID affinity; no IRQ writes, no host service changes, no memory migration','workers':workers,'started':time.time()};(dest/'transition.json').write_text(json.dumps(record,indent=2))
changed=[]
try:
 for w in workers:
  for task in w['tasks']:
   cpus=w['acl'] if 'acl_thread' in task['name'] else w['release'] if 'release_thread' in task['name'] else w['main']
   try:os.sched_setaffinity(task['tid'],cpus);changed.append(task);task['after']=sorted(os.sched_getaffinity(task['tid']));assert task['after']==cpus
   except ProcessLookupError:task['exited']=True
 record.update(state='APPLIED',finished=time.time());(dest/'transition.json').write_text(json.dumps(record,indent=2))
except BaseException:
 for task in reversed(changed):
  try:os.sched_setaffinity(task['tid'],task['before'])
  except ProcessLookupError:pass
 record['state']='ROLLED_BACK';(dest/'transition.json').write_text(json.dumps(record,indent=2));raise
# Runtime metadata aliases the same already-live model/API; no NPU restart.
e=p/'evidence'/(version+'-source');e.mkdir(exist_ok=False);(e/'SHA256.json').write_bytes((p/'evidence'/(parent+'-source')/'SHA256.json').read_bytes())
s=json.loads((p/'runs'/('serving-'+parent+'-status.json')).read_text());assert s['state']=='READY';s.update(runtime_label=version,actual_control_dir=str(control),cpu_affinity_record=str(dest/'transition.json'),model_runtime_unchanged=parent)
(p/'runs'/('serving-'+version+'-status.json')).write_text(json.dumps(s,indent=2))
print('OWN_WORKER_AFFINITY_APPLIED',json.dumps({'workers':len(workers),'threads':len(changed),'roles':[{'rank':w['rank'],'main':str(w['main'][0])+'-'+str(w['main'][-1]),'acl':w['acl'],'release':w['release']} for w in workers]}))
