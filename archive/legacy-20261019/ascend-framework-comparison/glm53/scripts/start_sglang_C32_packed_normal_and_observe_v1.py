"""Once-only reviewed packed-normal model start then bounded native startup observer."""
from pathlib import Path
import ast,hashlib,importlib.util,json,re
ROOT=Path(__file__).resolve().parents[1]
RID='20261018T-L09-packed-normal-model-and-observe-v1'
def load(name,file,expected):
 p=ROOT/'scripts'/file;assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def review():
 models=load('restore_models','start_sglang_C32_packed_normal_models_v1.py','b7c28ea437d35e527eed2a83d13e26ac7e1581c43d0a8be24d42f2beb619e936')
 observer=load('restore_observer','inspect_sglang_C32_packed_normal_startup_v1.py','860ccd515d989947a6d50cb1022a3084768b70a37e3f42ec3d50ab4360875fef')
 cap=json.loads((ROOT/'results/sglang-C32-packed-normal-pool-capacity-v1.json').read_text());assert cap['pass']
 checks={}
 for n in [34,35,36,41]:
  for name,body in [('precheck',models.PRECHECK),('start',models.START)]:
   code=models.guard.render(n,body,cap,models.base.FILES,models.base.STAGED);ast.parse(code);checks[f'{n}-{name}']=hashlib.sha256(code.encode()).hexdigest()
  values={'EXPECTED':observer.b.NODES[n][1],'NODE':n,'CID':observer.IDS[n],'NAME':f'glm53-sglang-C32-packed-normal-v1-20261018-{n}','META':observer.META}
  code=re.sub(r'\b(?:'+'|'.join(values)+r')\b',lambda m:repr(values[m[0]]),observer.HOST);ast.parse(code);checks[f'{n}-observe']=hashlib.sha256(code.encode()).hexdigest()
 frozen=ROOT/'results/C32-L09-actual-candidate-model-start-review-v1.json';assert hashlib.sha256(frozen.read_bytes()).hexdigest()=='67867a5790e6f21cd9b8fe86d985de696928ca9fe4dc83faeda4bf309583d227'
 assert checks==json.loads(frozen.read_text())['actual_CID_rendered_AST_sha256']
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
