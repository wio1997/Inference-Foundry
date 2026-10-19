"""Once-only L10 v2 model starts and streaming bounded startup observation."""
from pathlib import Path
import ast,hashlib,importlib.util,json,re
ROOT=Path(__file__).resolve().parents[1];RID='20261018T-L10-target-graph-model-and-observe-v4'
def load(name,file,sha):
 p=ROOT/'scripts'/file;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha;s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def review():
 cap_path=ROOT/'results/sglang-C32-target-graph-fullmesh-pool-capacity-v3.json';assert hashlib.sha256(cap_path.read_bytes()).hexdigest()=='04f5658ecfe8bde5a3466d6db12b0e4e61ce3d086f8276a18eddac68a2ffd3f9';cap=json.loads(cap_path.read_text());assert cap['pass']
 models=load('models','start_sglang_C32_target_graph_models_v4.py','13c8673ee518f9c32d81bbaa510f3ca6ffee3086bdc177f68e3178f090545e0f');obs=load('observer','inspect_sglang_C32_target_graph_startup_v4.py','53642bf41d2ddb342d45461ac433160aa2307684ca71a17f584b1215f2ed8f8f');checks={}
 for n in [34,35,36,41]:
  for kind,body in [('precheck',models.PRECHECK),('start',models.START)]:
   code=models.guard.render(n,body,cap,models.base.FILES,models.base.STAGED);ast.parse(code);checks[str(n)+'-'+kind]=hashlib.sha256(code.encode()).hexdigest()
  v={'EXPECTED':obs.b.NODES[n][1],'NODE':n,'CID':obs.IDS[n],'NAME':'glm53-sglang-C32-target-graph-fullmesh-v3-20261018-'+str(n),'META':obs.META};code=re.sub(r'\b(?:'+'|'.join(v)+r')\b',lambda m:repr(v[m[0]]),obs.HOST);ast.parse(code);checks[str(n)+'-observer']=hashlib.sha256(code.encode()).hexdigest()
 p=ROOT/'results/C32-L10-actual-model-start-observer-review-v4.json';assert hashlib.sha256(p.read_bytes()).hexdigest()=='7cc1b4bcc75d47e409c74003c0b533428503a0e88cd7f33440b618de70dba6dc';assert checks==json.loads(p.read_text())['actual_CID_rendered_AST_sha256']
 return models,obs,checks
def main():
 models,obs,checks=review();run=ROOT/'runs'/RID;run.mkdir(exist_ok=False);(run/'execution-once.lock').write_text(str(__import__('os').getpid()));receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'actual_host_AST':checks,'models_completed':False,'observer_completed':False,'max_SSH':2,'external_model_requests':0}
 try:models.main();receipt['models_completed']=True;obs.main();receipt['observer_completed']=True
 except BaseException as e:receipt['error']=repr(e);raise
 finally:(run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
