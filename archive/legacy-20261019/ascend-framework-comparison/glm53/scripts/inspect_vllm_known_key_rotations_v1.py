"""Pure CPU read-only key-order verification from actual accepted long prefix IDs."""
import ast,hashlib,importlib.util,json,re
from pathlib import Path
P=Path(__file__).resolve();path=P.with_name('run_vllm_shared_long_semantic_v2.py');assert hashlib.sha256(path.read_bytes()).hexdigest()=='6cb913034a2db4d24aecb8e0d92e70d62af39a7f8385b7448e2180cd11d201fc'
spec=importlib.util.spec_from_file_location('base',path);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
RUN=b.b.ROOT/'runs/20261017T1785-vllm-key-rotation-audit-v1'
HOST=r'''import pathlib,json,hashlib,pickle,struct
root=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1779-vllm-shared-long-semantic-v2')
helper=root/'glm53_vllm_semantic_meta_probe_v4.py';assert hashlib.sha256(helper.read_bytes()).hexdigest()=='10d668029ae9b968f1395ed33b1a2e8f3fd8432c11f4c625202c6ecf7e30b7ee'
def h(x):return hashlib.sha256(pickle.dumps(x,protocol=pickle.HIGHEST_PROTOCOL)).digest()
results=[]
for index in [0,1]:
 request=json.loads((root/f'need-metadata-{index}.json').read_text());ids=request['prefix_ids'];namespace=request['namespace'];assert namespace==f'glm53-sem-vllm-shared-v2-20261017T1779-case{index}-warm'
 assert len(ids)==74400 and hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest()==request['prefix_sha256_le_u32']
 expected=['61cb81d7a85d8642f5ab52dd700c8232e04ae141c8d03b46a9d968107c2ced75','0b9df0586836fac65d90ed0126d197183932708eeb3ed726522e0ff029ce0fa3'][index];assert request['prefix_sha256_le_u32']==expected
 parent=h('0');keys=[]
 for offset in range(0,len(ids)-127,128):
  parent=h((parent,tuple(ids[offset:offset+128]),(namespace,) if offset==0 else None));keys.append('GLM-5.3-w8a8@dcp:0@head_or_tp_rank:0@pp_rank:0@group:0@cache_role:kv@cache_family:default@'+parent.hex())
 assert len(keys)==len(set(keys))==581
 metadata=json.loads((root/f'metadata-gate-{index}.json').read_text());assert metadata['pass'] and metadata['metadata']['known_keys_sha256']==hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest()
 rows=[]
 for rank in range(8):
  shifted=keys[rank%len(keys):]+keys[:rank%len(keys)];assert set(shifted)==set(keys)
  rows.append({'TP_rank':rank,'rotation_pages':rank,'key_count':len(shifted),'key_list_sha256':hashlib.sha256(json.dumps(shifted,separators=(',',':')).encode()).hexdigest()})
 results.append({'case':index,'namespace':namespace,'prefix_sha256_le_u32':expected,'unique_physical_keys':581,'TP_read_orderings':rows})
print(json.dumps({'project_id':'glm53','pass':True,'SDK_init_calls':0,'model_requests':0,'cases':results}))
'''
def main():
 assert (b.b.ROOT/'runs/20261017T1783-vllm-shared-long-native-v2/receipt.json').is_file()
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 ast.parse(HOST);result=b.invoke(43,HOST,120);(RUN/'node-43.json').write_text(json.dumps(result,indent=2)+'\n');(RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'rc':result['rc'],'model_requests':0,'SDK_init_calls':0},indent=2)+'\n');print('readonly key-order',result['rc'])
if __name__=='__main__':main()
