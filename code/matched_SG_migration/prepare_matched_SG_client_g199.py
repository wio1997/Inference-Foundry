"""Stage the existing exact-token AISBench client for the matched SG epoch."""
from pathlib import Path
import ast,base64,hashlib
T=Path(__file__).resolve().parent;files={}
for n in ['run_aisbench_cold_hot.py','audit_aisbench.py']:
 s=(T/'benchmark_pipeline_vllm_g51'/n).read_text()
 if n=='run_aisbench_cold_hot.py':
  s=s.replace('client_configs_vllm_g44','client_configs_sg_g199')
  s=s.replace("base=Path('<TASK_ROOT>/vllm_g4')","base=Path('<TASK_ROOT>/sg_pd_g181')")
 ast.parse(s);files[n]={'b64':base64.b64encode(s.encode()).decode(),'sha256':hashlib.sha256(s.encode()).hexdigest()}
body='''from pathlib import Path
import ast,base64,datetime,hashlib,json,socket,subprocess
assert socket.gethostname()=='<ROLE_NODE>'
root=Path('<TASK_ROOT>');client=root/'client_vllm_g42';r=root/'benchmark_pipeline_sg_g199';cfgroot=root/'client_configs_sg_g199'
assert not r.exists() or not any(r.iterdir()),'Existing nonempty runner stage requires reconciliation'
assert not cfgroot.exists() or not any(cfgroot.iterdir()),'Existing nonempty config stage requires reconciliation'
assets=json.loads((client/'assets_state.json').read_text());assert assets['phase']=='complete'
manifest=json.loads((client/'dataset_manifest.json').read_text())
needed={(phase,n) for phase in ['warmup','warmup2','warmup3','warmup4','warmup5','formal'] for n in [32768,65536,131072,262144]}
assert needed<={(e['phase'],e['input_tokens']) for e in manifest['profiles']},str([(e['phase'],e['input_tokens']) for e in manifest['profiles']])
info=json.loads(subprocess.check_output(['docker','inspect','yxy-glm53-aisbench-0002-g42']))[0]
assert info['State']['Running'] and info['Id']=='0a2d37de7eb21cf1e07b04ab2c1dee7a18d15a149bccf2a6cdfbce92bc3067a9'
assert not info['HostConfig']['Devices'] and info['Image']=='sha256:05358d58d23f2679c71e48d93797d5d2a28d72eedc7fb4900479c89955465d76'
ledger=client/'formal_submissions_sglang.json';assert not ledger.exists(),'SG formal ledger already exists; inspect before a first submission'
r.mkdir(exist_ok=True);cfgroot.mkdir(exist_ok=True);published=[]
class Route(ast.NodeTransformer):
 def visit_Call(self,node):
  self.generic_visit(node)
  if isinstance(node.func,ast.Name) and node.func.id=='dict':
   keys=[k.arg for k in node.keywords]
   if 'type' in keys and 'model' in keys:
    for key,value in [('host_ip','<HOST>'),('host_port',18080),('abbr','sglang-glm53flash-910b-pd')]:
     if key in keys:node.keywords[keys.index(key)].value=ast.Constant(value=value)
   for keyword in node.keywords:
    if keyword.arg=='path' and isinstance(keyword.value,ast.Constant) and '/datasets/' in str(keyword.value.value):keyword.value=ast.Constant(value=self.dataset_path)
  return node
 def visit_Dict(self,node):
  self.generic_visit(node)
  keys=[k.value if isinstance(k,ast.Constant) else None for k in node.keys]
  if 'type' in keys and 'model' in keys:
   for key,value in [('host_ip','<HOST>'),('host_port',18080),('abbr','sglang-glm53flash-910b-pd')]:
    if key in keys:node.values[keys.index(key)]=ast.Constant(value=value)
  for i,key in enumerate(keys):
   if key=='path' and isinstance(node.values[i],ast.Constant) and '/datasets/' in str(node.values[i].value):node.values[i]=ast.Constant(value=self.dataset_path)
  return node
for entry in manifest['profiles']:
 if (entry['phase'],entry['input_tokens']) not in needed:continue
 phase=entry['phase'];length=entry['input_tokens'];dataset=client/'datasets'/f'{phase}_{length}.jsonl'
 assert hashlib.sha256(dataset.read_bytes()).hexdigest()==entry['sha256']
 source=root/'client_configs_vllm_g44'/f'vllm_formal_{length}.py'
 transformer=Route();transformer.dataset_path=f'/task/client_vllm_g42/datasets/{phase}_{length}.jsonl'
 tree=transformer.visit(ast.parse(source.read_text()));ast.fix_missing_locations(tree)
 out=cfgroot/f'sglang_{phase}_{length}.py';out.write_text(ast.unparse(tree)+'\\n');ast.parse(out.read_text())
 published.append({'phase':phase,'input_tokens':length,'requests':entry['requests'],'dataset_sha256':entry['sha256'],'source_config_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'config_sha256':hashlib.sha256(out.read_bytes()).hexdigest()})
for name,e in files.items():
 b=base64.b64decode(e['b64']);assert hashlib.sha256(b).hexdigest()==e['sha256'];(r/name).write_bytes(b)
receipt={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'client_id':info['Id'],'client_image':info['Image'],'datasets_reused_byte_identical':True,'profiles':published,'formal_not_submitted':True,'model_requests_sent':False,'configuration':'Same frozen dataset/tokenizer, VLLMCustomAPI OpenAI streaming parser, C16,1024output; engineSG and route change only','next':'Only after real G181 PD functional/registration/graph gates, run first independent warmup with warmup_only and P cache audit'}
(r/'stage_receipt.json').write_text(json.dumps(receipt,indent=2));print('G199_SG_CLIENT_STAGE_JSON '+json.dumps(receipt))
'''
(T/'stage_matched_SG_client_g199.sh').write_text("python3 - <<'REMOTE_PY'\nfiles="+repr(files)+'\n'+body+'REMOTE_PY\n')
print('Prepared CPU-only matchedSG client stage; no model requests or scored runs')
