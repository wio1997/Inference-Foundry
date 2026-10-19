"""One-time dependent L04 model-parent restore then native startup observation."""
from pathlib import Path
import ast,hashlib,importlib.util,json,re
ROOT=Path(__file__).resolve().parents[1]
RID='20261018T-L09-packed-L04-restore-and-observe-v1'
def load(name,file,expected):
 p=ROOT/'scripts'/file;assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def review():
 models=load('restore_models','start_sglang_C32_packed_L04_rollback_models_v1.py','0ff03c8cffa21454d6f273301eb426c8b8cf868d74d60821292141401a6f5899')
 observer=load('restore_observer','inspect_sglang_C32_packed_L04_rollback_startup_v1.py','68bc344967a549c33ee9dfbc21511209971e51f96f6e6018ae68371b717356c3')
 cap=json.loads((ROOT/'results/sglang-C32-packed-L04-rollback-pool-capacity-v1.json').read_text());assert cap['pass']
 checks={}
 for n in [34,35,36,41]:
  for name,body in [('precheck',models.PRECHECK),('start',models.START)]:
   code=models.guard.render(n,body,cap,models.base.FILES,models.base.STAGED);ast.parse(code);checks[f'{n}-{name}']=hashlib.sha256(code.encode()).hexdigest()
  values={'EXPECTED':observer.b.NODES[n][1],'NODE':n,'CID':observer.IDS[n],'NAME':f'glm53-sglang-C32-packed-normal-L04-rollback-v1-20261018-{n}','META':observer.META}
  code=re.sub(r'\b(?:'+'|'.join(values)+r')\b',lambda m:repr(values[m[0]]),observer.HOST);ast.parse(code);checks[f'{n}-observe']=hashlib.sha256(code.encode()).hexdigest()
 return models,observer,checks
def main():
 run=ROOT/'runs'/RID;run.mkdir(exist_ok=True)
 with (run/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 models,observer,checks=review();receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_host_AST_sha256':checks,'max_SSH':2,'external_model_requests':0,'models_completed':False,'observer_completed':False}
 try:
  models.main();receipt['models_completed']=True
  observer.main();receipt['observer_completed']=True
 finally:(run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
