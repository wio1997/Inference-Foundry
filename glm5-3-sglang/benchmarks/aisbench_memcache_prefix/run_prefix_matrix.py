#!/usr/bin/env python3
"""GLM-5.3 4P2D warm-only AISBench runner; no deployment or service restart."""
import argparse,json,os,subprocess,sys,shutil,time,fcntl
from pathlib import Path
K=Path(__file__).resolve().parent
sys.path.insert(0,str(K/'scripts'))
from bench_profile import PROFILE,PROFILE_FILE,validate_run_profile
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('action',choices=['plan','check','run'])
p.add_argument('--source',default=os.environ.get('SOURCE_ROOT','/work/glm53-prefix-data'))
p.add_argument('--out',default=os.environ.get('RUN_ROOT','/work/aisbench-glm53-'+time.strftime('%Y%m%d-%H%M%S')))
p.add_argument('--audit-hook',default=os.environ.get('AUDIT_HOOK',''))
a=p.parse_args();r=Path(a.out).resolve()
plan=dict(model='GLM-5.3 W8A8',input_tokens=80000,output_tokens=600,prefix_tokens=74400,cached_tokens=74368,
          requests_by_concurrency={'16':64,'32':128,'64':256},total_requests=448,
          max_prefixes_per_group=64,groups=7,prewarm_in_throughput=False,
          service_profile='Ascend SGLang4P2D; dedicated MemCache; see GLM_BENCH_PROFILE',
          data_source=str(Path(a.source).resolve()),output=str(r))
if a.action=='plan':print(json.dumps(plan,ensure_ascii=False,indent=2));sys.exit(0)
if a.action=='run':validate_run_profile()
assert not r.exists(),'Use a NEW output directory; existing traces are never overwritten'
r.mkdir(parents=True);shutil.copytree(K/'scripts',r/'scripts')
for d in ['logs','configs']:(r/d).mkdir()
lock=open(r/'run.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if a.action=='run':
 matrix_dir=Path(PROFILE['lock_roots'][0]);matrix_dir.mkdir(parents=True,exist_ok=True)
 matrix_lock=(matrix_dir/'matrix.lock').open('a');fcntl.flock(matrix_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
env=os.environ.copy();
if PROFILE_FILE:env['GLM_BENCH_PROFILE']=str(Path(PROFILE_FILE).resolve())
env.update(NO_PROXY='*',no_proxy='*',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',AIS_BENCH_REQUEST_TIMEOUT='7200')
for key in ['http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY']:env.pop(key,None)
env['PYTHONPATH']=str(r/'scripts')+os.pathsep+env.get('PYTHONPATH','')
def step(label,script,args=(),root=None):
 cmd=[sys.executable,str(r/'scripts'/script),'--root',str(root or r),*map(str,args)]
 with(r/'logs'/(label+'.log')).open('w')as log:
  z=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 assert z.returncode==0,f'{label} failed; read {r}/logs/{label}.log; raw files preserved'
def audit(cc,root,start=0,label=None,manifest='dataset-manifest.jsonl'):
 label=label or str(cc)
 argv=[str(Path(a.audit_hook).resolve()),'--cc',str(cc),'--start-row',str(start),'--audit-label',label,'--manifest',manifest]
 z=subprocess.run(argv,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=900)
 (r/'logs'/('storage-'+label+'.log')).write_text(z.stdout+z.stderr)
 assert z.returncode==0,('storage audit hook failed',label)
 values=[]
 for line in z.stdout.splitlines():
  try:x=json.loads(line)
  except json.JSONDecodeError:continue
  if isinstance(x,dict)and x.get('phase')=='STORAGE_PREFIX_AUDIT_PASS':values.append(x)
 assert len(values)==1 and not values[0]['pending_rows'],('No actual storage proof',label)
 x=values[0];assert x['requests']==(64 if str(cc).isdigit()else 1)
 assert x['source_start_row']==start and x['cached_tokens_each']==74368
 (Path(root)/'logs'/f'prefix-storage-cc{cc}.json').write_text(json.dumps(x,indent=2)+'\n')
def admin(label,operation,root):step(label+'-'+Path(root).name,'cache_admin.py',[operation,'--label',label],root)
step('offline-check','verify_installation.py')
if a.action=='check':print('OFFLINE_CHECK_PASS: no model HTTP requests');sys.exit(0)
assert a.audit_hook and Path(a.audit_hook).is_file()and os.access(a.audit_hook,os.X_OK),'Set executable AUDIT_HOOK; never replace real L3 verification with sleep'
(r/'contract.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
step('prepare','prepare_fourbatch.py',['--source',a.source])
step('offline-datasets','verify_installation.py',['--check-datasets'])
# One known-key correctness smoke, then a separate80K/600 smoke, outside formal metrics.
for cc,manifest,label in [('needle','needle-manifest.jsonl','known-key-l3-smoke'),('smoke600','dataset-manifest.jsonl','long80k-600-l3-smoke')]:
 admin('before-'+cc,'clear_l1_l2_l3',r)
 step('prewarm-'+cc,'prewarm.py',['--cc',cc,'--manifest',manifest,'--execute'])
 audit(cc,r,manifest=manifest)
 admin('after-warm-'+cc,'flush_l1_l2',r)
 args=['--cc',cc,'--output','600','--label',label]
 if cc=='needle':args+=['--expected-key',json.loads((r/'needle-manifest.jsonl').read_text())['expected_key']]
 step(label,'smoke.py',args)
data=json.loads((r/'dataset-audit.json').read_text());results=[]
for g in data['cohorts']:
 cc=str(g['cc']);leaf=Path(g['root']);cid=g['cohort_id']
 admin('before-cc'+cc,'clear_l1_l2_l3',leaf)
 step('prewarm-'+cid,'prewarm.py',['--cc',cc,'--execute'],leaf)
 audit(cc,leaf,start=g['source_start_row'],label=cid)
 admin('after-warm-cc'+cc,'flush_l1_l2',leaf)
 step('formal-'+cid,'run_cc.py',['--cc',cc,'--execute'],leaf)
 result=json.loads((leaf/'reports'/('cc'+cc)/'result.json').read_text());assert result['valid']
 results.append(dict(cohort=cid,cc=int(cc),result_file=str(leaf/'reports'/('cc'+cc)/'result.json'),prewarm_s=result['prewarm_separate']['duration_s']))
(r/'MATRIX_COMPLETE.json').write_text(json.dumps(dict(valid=True,formal_requests=448,groups=results,prewarm_in_throughput=False,raw_audit_and_aggregation_required=True),indent=2)+'\n')
subprocess.run([sys.executable,str(K/'aggregate_results.py'),'--root',str(r)],env=env,check=True)
print(json.dumps(dict(status='MATRIX_AND_REQUEST_AUDIT_COMPLETE',formal_requests=448,output=str(r))))
