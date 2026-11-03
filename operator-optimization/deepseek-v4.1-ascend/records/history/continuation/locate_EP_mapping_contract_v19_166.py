from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-communications-v15')
OUT=P/'EP-expert-mapping-symbol-locations-v19.json';assert not OUT.exists()
code=r'''from pathlib import Path
import ast,json,hashlib
specs=[('/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py',['class DeepseekV41MoE','expert_placement','physical_expert','hash','FusedMoE','expert_map']),('/vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py',['dispatch_ffn_combine','expert_map','topk_ids','weights.w1','expert_token_nums']),('/vllm-workspace/vllm-ascend/vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py',['ascend_expert_map','class AscendW8A8DynamicFused','def process_weights','scale_bias','w13_weight','w2_weight']),('/vllm-workspace/vllm/vllm/model_executor/layers/fused_moe/expert_map_manager.py',['def ','placement_strategy','map_global_to_local']),('/vllm-workspace/vllm/vllm/model_executor/layers/fused_moe/routed_experts.py',['def weight_loader','map_global_to_local','def _load','expert_id =','global_expert_id','placement_strategy']),('/vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py',['ascend_expert_map','expert_map','topk_ids','router'])]
rows=[]
for name,tokens in specs:
 p=Path(name);s=p.read_text();ls=s.splitlines();tree=ast.parse(s);nodes=[n for n in ast.walk(tree)if isinstance(n,(ast.FunctionDef,ast.ClassDef,ast.AsyncFunctionDef))];hits=[]
 for i,l in enumerate(ls,1):
  if any(t in l for t in tokens):
   parents=sorted([n for n in nodes if n.lineno<=i<=n.end_lineno],key=lambda n:n.end_lineno-n.lineno)
   hits.append(dict(line=i,text=l,enclosing=[dict(name=n.name,start=n.lineno,end=n.end_lineno)for n in parents[:2]]))
 assert len(hits)<=110,(name,len(hits))
 rows.append(dict(path=name,sha256=hashlib.sha256(s.encode()).hexdigest(),hits=hits))
print(json.dumps(dict(files=rows,no_module_import_or_device_work=True)))
'''
q=subprocess.run(['docker','exec','bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d','python3','-c',code],capture_output=True,text=True,timeout=80);assert q.returncode==0,q.stderr
v=json.loads(q.stdout);v.update(created_utc=datetime.now(timezone.utc).isoformat(),scope='Six profile-raised exact mapping/loader/call files; symbol locations only,no repository traversal,no model import.',overall_complete=False)
b=(json.dumps(v,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=v,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
