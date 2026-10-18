"""Pinned SFA cache semantics with a CPU model of graph-owned Cast; no device result."""
from pathlib import Path
from types import SimpleNamespace
import ast,hashlib,json
p=Path('/Users/wio/work/glm53-execution-evidence-backup/20261018-prefill-graph-source/ascend-sfa_v1.py');data=p.read_bytes();sha=hashlib.sha256(data).hexdigest();assert sha=='9b94f67f30364d40caf4a7a67af0f0365d1ba0b5048939873ab0464e95b096dc'
node=next(n for n in ast.parse(data).body if isinstance(n,ast.FunctionDef) and n.name=='_int64_kv_slots');node.returns=None
for a in node.args.args:a.annotation=None
owner='outside';graphs={'native':[],'replica':[],'prefix':[]}
class Tensor:
 def __init__(self,values,dtype='int32',origin=None):self.values=list(values);self.dtype=dtype;self.origin_graph=origin
 def to(self,dtype):
  if self.dtype==dtype:return self
  out=Tensor(self.values,dtype,owner);graphs[owner].append(lambda:out.values.__setitem__(slice(None),self.values));return out
ns={'torch':SimpleNamespace(Tensor=Tensor,int64='int64')};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(p),'exec'),ns);convert=ns['_int64_kv_slots']
slots=Tensor([0,1,2]);shared=SimpleNamespace()
owner='native';a=convert(slots,shared)
owner='replica';b=convert(slots,shared)
assert a is b and len(graphs['native'])==1 and len(graphs['replica'])==0
slots.values=[10,11,12]
for work in graphs['replica']:work()
assert b.values==[0,1,2] and b.values!=slots.values
owner='prefix';fresh=SimpleNamespace();c=convert(slots,fresh);slots.values=[20,21,22]
for work in graphs['prefix']:work()
assert c.values==slots.values and c is not a
result={'completed':True,'source_sha256':sha,'function':'_int64_kv_slots','shared_metadata_second_capture_reuses_first_cast_output':a is b,'shared_metadata_second_graph_cast_count':len(graphs['replica']),'shared_metadata_second_replay_stale_slots':b.values!=slots.values,'fresh_metadata_has_own_cast':c.origin_graph=='prefix' and len(graphs['prefix'])==1,'fresh_metadata_replay_updates_slots':c.values==slots.values,'scope':'CPU execution of exact pinned function AST with modeled graph-owned Cast, explains a concrete possible dependency in D2. No NPU observation of D2 internal cache, no unique full-model root-cause or performance proof; D3 also adds variant warmup.'}
print(json.dumps(result,indent=2))
