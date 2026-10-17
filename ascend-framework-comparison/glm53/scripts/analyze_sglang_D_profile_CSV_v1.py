"""Read all32 native profile CSVs on their nodes; one SSH at a time, no model calls."""
import ast,hashlib,importlib.util,json,re,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/20261017T1942-sglang-D-profile-CSV-analysis-v1'
def load(name,path,sha):
 p=ROOT/path;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=load('base','scripts/inventory_readonly.py','e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777')
identity=load('identity','scripts/glm53_shared_model_identity_v1.py','e89e2a6a45878328ea0772849bf629a76a177abbfb58fb27988c36c8cf6a6ec4')
codec=load('codec','scripts/glm53_native_output_transport_v1.py','ed265b69437cc93a9fe8afdc1ac87f151e8168e9c82ff28a5d374f14d0c622d5')
HOST=r'''import pathlib,gzip,json,csv,traceback
scope=IDENTITY_ENV;exec(compile(IDENTITY_SOURCE,'identity','exec'),scope)
root=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1720-sglang-shared-pool-v2/profile/20261017T1920-sglang-formal-C16-D-profile-v1');assert root.is_dir()
archive=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1942-sglang-D-profile-CSV-analysis-v1');assert not archive.exists();archive.mkdir(parents=True)
exec(compile(CORE,'native-profile-csv-analysis','exec'),globals())
out={'project_id':'glm53','node':NODE,'model_identity':scope['model_identity'],'model_requests':0,'SDK_init_calls':0,'profile_control_requests':0,'raw_trace_downloaded':False,'workers':{},'analysis_pass':False}
for name,settings in WORKERS.items():
 try:out['workers'][name]=analyze_worker(root/name,settings)
 except Exception as error:
  out['worker_error']={'worker':name,'error':repr(error),'traceback':traceback.format_exc()}
  out['schema_samples']={}
  for relative in settings['files']:
   path=root/name/relative
   if path.suffix=='.csv':
    with path.open(newline='') as file:
     reader=csv.DictReader(file);out['schema_samples'][relative]=[row for _,row in zip(range(2),reader)]
  break
out['analysis_pass']=len(out['workers'])==len(WORKERS) and not out.get('worker_error')
'''
def render(code,values):
 pattern=r'\b(?:'+'|'.join(re.escape(k) for k in values)+r')\b';s=re.sub(pattern,lambda m:repr(values[m[0]]),code);ast.parse(s);return s
def main():
 p=ROOT/'config/diagnostic-runs-v1/sglang-D-profile-CSV-analysis-v1.json';assert hashlib.sha256(p.read_bytes()).hexdigest()=='7ff106af9e1b2ea445620839be516c458600274e903586c8837d211e3fadbca4';settings=json.loads(p.read_text())
 core=ROOT/'scripts/glm53_profile_csv_analysis_v1.py';assert hashlib.sha256(core.read_bytes()).hexdigest()==settings['source_sha256']=='c8e661f21c6a7413cd5e8ac34e490d7e457151c40811d566be4652ce2278a3a3'
 for relative,sha in settings['capture_evidence_sha256'].items():assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==sha
 prior=json.loads((ROOT/'runs/20261017T1924-sglang-D-profile-native-v1/receipt.json').read_text());assert set(prior['nodes'])=={'34','35','36','41','43'} and prior['nodes']['36']==prior['nodes']['41']==0
 runtime=json.loads((ROOT/'config/diagnostic-runs-v1/sglang-C16-D-profile.json').read_text());RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model_requests':0,'SDK_init_calls':0,'nodes':{}}
 try:
  for node in [36,41]:
   pod,host=base.NODES[node];mapping=base.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert mapping['rc']==0 and mapping['stdout']==f'172.16.12.{node}'
   item=runtime['runtime_containers'][str(node)];values={'IDENTITY_ENV':{'CID':item['container_id'],'NAME':item['name'],'IMAGE':runtime['model_image'],'NODE':node},'IDENTITY_SOURCE':identity.HOST,'CORE':core.read_text(),'NODE':node,'WORKERS':settings['workers'][str(node)]}
   code='import socket\nassert socket.gethostname()=='+repr(host)+'\n'+render(HOST,values)+codec.ENCODER;ast.parse(code)
   command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-']);result=codec.decode(base.invoke([command],code,timeout=900));(RUN/f'node-{node}.json').write_text(json.dumps(result,indent=2)+'\n');receipt['nodes'][str(node)]=result['rc'];print(node,result['rc'],flush=True)
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
