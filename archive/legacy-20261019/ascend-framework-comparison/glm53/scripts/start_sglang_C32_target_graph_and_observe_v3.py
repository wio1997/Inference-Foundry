"""Once-only L10 v2 model starts and streaming bounded startup observation."""
from pathlib import Path
import ast,hashlib,importlib.util,json,re
ROOT=Path(__file__).resolve().parents[1];RID='20261018T-L10-target-graph-model-and-observe-v3'
def load(name,file,sha):
 p=ROOT/'scripts'/file;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha;s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def review():
 cap_path=ROOT/'results/sglang-C32-target-graph-fullmesh-pool-capacity-v2.json';assert hashlib.sha256(cap_path.read_bytes()).hexdigest()=='6f660dbc952927b31b66c7c79503bfab99afa2533255d6ac5f94432f60dd12f6';cap=json.loads(cap_path.read_text());assert cap['pass']
 models=load('models','start_sglang_C32_target_graph_models_v3.py','cd1125891536c19929bcbb36d21d1c3774737c4da0c94838b23121a0649683aa');obs=load('observer','inspect_sglang_C32_target_graph_startup_v3.py','ad8de3be907c2df4c747e8bd3450b0262a784172213094621493b864c8d4e4c0');checks={}
 for n in [34,35,36,41]:
  for kind,body in [('precheck',models.PRECHECK),('start',models.START)]:
   code=models.guard.render(n,body,cap,models.base.FILES,models.base.STAGED);ast.parse(code);checks[str(n)+'-'+kind]=hashlib.sha256(code.encode()).hexdigest()
  v={'EXPECTED':obs.b.NODES[n][1],'NODE':n,'CID':obs.IDS[n],'NAME':'glm53-sglang-C32-target-graph-fullmesh-v2-20261018-'+str(n),'META':obs.META};code=re.sub(r'\b(?:'+'|'.join(v)+r')\b',lambda m:repr(v[m[0]]),obs.HOST);ast.parse(code);checks[str(n)+'-observer']=hashlib.sha256(code.encode()).hexdigest()
 p=ROOT/'results/C32-L10-actual-model-start-observer-review-v3.json';assert hashlib.sha256(p.read_bytes()).hexdigest()=='2d56acbd06e6fb3e05bcfe6934439560c2b8583895e694fd163e761a758dfd09';assert checks==json.loads(p.read_text())['actual_CID_rendered_AST_sha256']
 return models,obs,checks
def main():
 previous=json.loads((ROOT/'runs/20261018T-L10-target-graph-model-start-v2/receipt.json').read_text());assert previous['starts']=={},'No restart/replay of a started parent'
 models,obs,checks=review();run=ROOT/'runs'/RID;run.mkdir(exist_ok=False);(run/'execution-once.lock').write_text(str(__import__('os').getpid()));receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'actual_host_AST':checks,'models_completed':False,'observer_completed':False,'max_SSH':2,'external_model_requests':0}
 try:models.main();receipt['models_completed']=True;obs.main();receipt['observer_completed']=True
 except BaseException as e:receipt['error']=repr(e);raise
 finally:(run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
