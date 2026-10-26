"""Stage independent L10 candidate/rollback; fixed-image CPU hook test, no model change."""
from pathlib import Path
import ast,base64,concurrent.futures,hashlib,importlib.util,json,os,shlex
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/20261018T-L10-target-graph-stage-v1'
MANIFESTS={'config/sglang-C32-target-graph-fullmesh-stage-v1.json':'00186c2f3c426bcd02668fa595d44dfe2ebb4a48b9f24b49c4a71834edaefa52','config/sglang-C32-target-graph-L04-rollback-stage-v1.json':'d8ba19a408845ffffe35ddb3350578f260a2e72d72456a0477b924d05c84a410'}
CPU=r'''import pathlib,subprocess,json,hashlib
root=pathlib.Path(REMOTE)
for n,item in FILES.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==item['sha256']
r=subprocess.run(ARGS,capture_output=True,text=True,timeout=120)
print(json.dumps({'node':36,'project_id':'glm53','rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'model_requests':0,'SDK_init_calls':0,'NPU_calls':0,'existing_models_changed':False}))
'''
def loadlib():
 p=ROOT/'scripts/stage_sglang_C32_packed_normal_and_rollback_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='13545bf1c90368d57113dc810738da12c64d3efa5c01ba3fd38057e21cb1724f';s=importlib.util.spec_from_file_location('stage_library',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def bundles():
 out=[]
 for p,h in MANIFESTS.items():
  q=ROOT/p;assert hashlib.sha256(q.read_bytes()).hexdigest()==h;out.extend(json.loads(q.read_text())['bundles'])
 for b in out:
  for x in b['files'].values():assert hashlib.sha256((ROOT/x['local']).read_bytes()).hexdigest()==x['sha256']
 return out
def cpu_code(m,b):
 args=['docker','run','--rm','--pull=never','--network=none','--cpus=2','--memory=4g','--pids-limit=256','--label','project_id=glm53','--name','glm53-L10-target-graph-CPU-qual-v1-20261018-36','-v',b['remote']+':/opt/glm53/staged:ro','-e','PYTHONDONTWRITEBYTECODE=1','--entrypoint','python3',b['image'],'/opt/glm53/staged/python/qualify_image_cpu_v1.py']
 return m.replace(CPU,[('REMOTE',b['remote']),('FILES',b['files']),('ARGS',args)])
def main():
 q=ROOT/'results/C32-target-graph-fullmesh-local-compatibility-v1.json';assert json.loads(q.read_text())['pass'];m=loadlib();bs=bundles();RUN.mkdir(exist_ok=False);(RUN/'execution-once.lock').write_text(str(os.getpid()));receipt={'project_id':'glm53','manifest_sha256':MANIFESTS,'stages':{},'model_requests':0,'SDK_init_calls':0,'NPU_calls':0,'existing_models_changed':False}
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   tasks={pool.submit(m.stage,b,n):(i,n) for i,b in enumerate(bs) for n in b['nodes']}
   for f in concurrent.futures.as_completed(tasks):
    i,n=tasks[f];key=str(i)+'-'+str(n)
    try:r=f.result()
    except Exception as e:r={'error':repr(e)}
    (RUN/('stage-'+key+'.json')).write_text(json.dumps(r,indent=2)+'\n');receipt['stages'][key]=r.get('rc',r.get('error'));print('stage',key,receipt['stages'][key],flush=True)
  assert len(receipt['stages'])==8 and all(v==0 for v in receipt['stages'].values())
  code=cpu_code(m,bs[0]);ast.parse(code);(RUN/'CPU-host-source.py').write_text(code);r=m.invoke(36,code,timeout=150);(RUN/'CPU-image-qualification.json').write_text(json.dumps(r,indent=2)+'\n');assert r['rc']==0
  d=json.loads(r['stdout']);assert d['rc']==0,d['stderr'];v=json.loads(d['stdout'].split('GLM53_LOCAL_COMPATIBILITY=')[-1]);assert v['pass'] and v['NPU_calls']==0 and v['SDK_init_calls']==0
  receipt['fixed_image_CPU_qualification']=v;receipt['pass']=True;print(json.dumps({'fixed_image_CPU_pass':True,'stage_files_per_bundle':[len(b['files']) for b in bs],'model_requests':0,'NPU_calls':0}),flush=True)
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
