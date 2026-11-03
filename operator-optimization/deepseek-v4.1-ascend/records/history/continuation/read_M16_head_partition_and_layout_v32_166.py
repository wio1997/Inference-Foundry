from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18');OUT=M/'M16-selected-head-postprocess-gather-layout-and-Markov-owner-contract-v32.json';assert not OUT.exists()
code=r'''from pathlib import Path
import ast,json,hashlib
A=Path('/vllm-workspace/vllm-ascend/vllm_ascend');V=Path('/vllm-workspace/vllm/vllm');rows=[]
specs=[(A/'ops/vocab_parallel_embedding.py',None,[('function',533),('function',555),('function',598)]),
       (V/'model_executor/layers/logits_processor.py',None,[('function',109),('function',130),('function',144)]),
       (V/'distributed/device_communicators/base_device_communicator.py',None,[('function',261)]),
       (V/'model_executor/layers/vocab_parallel_embedding.py',None,[('function',82),('partition_fields',None)]),
       (V/'model_executor/layers/utils.py',None,[('function',90)]),
       (V/'model_executor/models/qwen3_dspark.py',None,[('function',93),('owner_init',93)]),
       (A/'models/deepseek_v41/model.py','101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91',[('function',1782)]),
       (A/'models/deepseek_v41/dspark.py',None,[('function',332),('function',324)])]
for p,expected,selections in specs:
 b=p.read_bytes();sha=hashlib.sha256(b).hexdigest()
 if expected:assert sha==expected
 text=b.decode();ls=text.splitlines();t=ast.parse(text);ff=[]
 for mode,line in selections:
  if mode=='partition_fields':
   cs=[c for c in ast.walk(t)if isinstance(c,ast.ClassDef)and c.name=='VocabParallelEmbedding'];assert len(cs)==1
   fs=[f for f in cs[0].body if isinstance(f,ast.FunctionDef)and f.name=='__init__'];assert len(fs)==1;f=fs[0]
   attrs={'disable_tp','tp_size','tp_rank','shard_indices','num_embeddings_per_partition','num_embeddings_padded','org_num_embeddings_padded','num_embeddings','org_vocab_size','org_num_embeddings'}
   statements=[]
   for z in ast.walk(f):
    if not isinstance(z,(ast.Assign,ast.AnnAssign)):continue
    targets=z.targets if isinstance(z,ast.Assign)else[z.target]
    names={a.attr for target in targets for a in ast.walk(target)if isinstance(a,ast.Attribute)}
    if not(names&attrs):continue
    assert z.end_lineno-z.lineno<30
    statements.append(dict(line=z.lineno,target_attributes=sorted(names),source=[str(j+1)+': '+ls[j]for j in range(z.lineno-1,z.end_lineno)]))
   ff.append(dict(name=f.name,owners=[cs[0].name],mode=mode,start=f.lineno,end=f.end_lineno,signature=ast.unparse(f.args),statements=statements));continue
  fs=[f for f in ast.walk(t)if isinstance(f,ast.FunctionDef)and f.lineno<=line<=f.end_lineno];assert len(fs)==1,(str(p),line)
  f=fs[0];owners=[c for c in ast.walk(t)if isinstance(c,ast.ClassDef)and c.lineno<=f.lineno<=c.end_lineno]
  if mode=='owner_init':
   assert len(owners)==1;fs=[f for f in owners[0].body if isinstance(f,ast.FunctionDef)and f.name=='__init__'];assert len(fs)==1;f=fs[0]
  assert f.end_lineno-f.lineno<155,(str(p),f.name,'too large;stop')
  ff.append(dict(name=f.name,owners=[c.name for c in owners],mode=mode,start=f.lineno,end=f.end_lineno,source=[str(j+1)+': '+ls[j]for j in range(f.lineno-1,f.end_lineno)]))
 rows.append(dict(path=str(p),sha256=sha,functions=ff))
assert sum(len(f.get('source',[]))+sum(len(s['source'])for s in f.get('statements',[]))for r in rows for f in r['functions'])<720
print(json.dumps(dict(files=rows,no_model_or_torch_import=True,scope='Onlynative-frame-selectedhead/gather/postprocess/MarkovmethodsandVocabParallelEmbeddingpartitionfieldassignments;notwholeconstructororrepositoryaudit.')))
'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=35);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(created_utc=datetime.now(timezone.utc).isoformat(),no_model_import_device_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
