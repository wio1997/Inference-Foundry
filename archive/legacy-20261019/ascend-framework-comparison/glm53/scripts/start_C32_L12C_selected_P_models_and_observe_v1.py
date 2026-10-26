"""Once-only selected P start after independent actual capacity/fixture audit."""
import concurrent.futures,hashlib,importlib.util,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RID='20261019T-L12C-selected-P-models-and-observe-v1';RUN=ROOT/'runs'/RID
LIB_SHA='ac856e9ead23a242bf07466a2a6b5899cae67f43bc377c34e1e3ec106c7f87d1'
def plan():
 p=ROOT/'scripts/glm53_C32_L12C_P_transition_library_v1.py';assert hashlib.sha256(p.read_bytes()).hexdigest()==LIB_SHA;s=importlib.util.spec_from_file_location('L12B',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m,m.plan_all()
def main():
 m,plans=plan();review=json.loads((ROOT/'results/C32-L12C-P-model-start-actual-review-v1.json').read_text());assert {k:hashlib.sha256(v.encode()).hexdigest() for k,v in plans.items()}==review['host_plans_sha256']
 cap_path=ROOT/'results/sglang-C32-L12C-selected-P-pool-capacity-v1.json';cap=json.loads(cap_path.read_text());assert cap['pass'] and cap['actual_MC_bytes']==568*1024**3 and not cap['new_P_models_started']
 kind=cap['selected_variant'];assert kind in ['candidate','rollback']
 if kind=='candidate':assert cap['native_82_kernel_fixture_qualified']
 receipt={'project_id':'glm53','kind':kind,'max_SSH':2,'external_model_requests':0,'capacity_sha256':hashlib.sha256(cap_path.read_bytes()).hexdigest(),'starts':{},'observations':{}}
 RUN.mkdir(exist_ok=False);(RUN/'execution-once.lock').write_text(str(os.getpid()));base=m.stage()
 def save(name,obj):(RUN/name).write_text(json.dumps(obj,indent=2)+'\n')
 try:
  for phase,timeout in [('start',150),('observe',1950)]:
   with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    jobs={pool.submit(base.invoke,n,plans[kind+'-'+phase+'-'+str(n)],timeout):n for n in [34,35]}
    for f in concurrent.futures.as_completed(jobs):
     n=jobs[f]
     try:r=f.result()
     except Exception as e:r={'error':repr(e)}
     save(phase+'-'+str(n)+'.json',r);receipt['starts' if phase=='start' else 'observations'][str(n)]=r.get('rc',r.get('error'));print(phase,n,r.get('rc',r.get('error')),flush=True)
   save('receipt.json',receipt);assert len(receipt['starts' if phase=='start' else 'observations'])==2 and all(v==0 for v in receipt['starts' if phase=='start' else 'observations'].values())
  receipt['startup_pass']=True
 finally:save('receipt.json',receipt)
 print(json.dumps(receipt))
if __name__=='__main__':main()
