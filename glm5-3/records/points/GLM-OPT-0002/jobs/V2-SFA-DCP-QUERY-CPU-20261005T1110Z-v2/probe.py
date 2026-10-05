from pathlib import Path
import ast,json,hashlib,types,torch
from vllm_ascend.attention.context_parallel import sfa_cp
j=Path(__file__).parent
classes=[c for c in vars(sfa_cp).values()if isinstance(c,type)and "_build_slot_mapping_replicated_view"in vars(c)]
assert len(classes)==1;cls=classes[0];method=cls._build_slot_mapping_replicated_view
import inspect
source=inspect.getsource(method);(j/"native_method_source.py.txt").write_text(source)
snapshot=j.parent/"AUDIT-RUN218-CONFIG-20261005T1050Z/01_sfa_cp.py";actual=Path(inspect.getsourcefile(method));assert actual.read_bytes()==snapshot.read_bytes()
def fixture(qsl,nactual,ninput,positions):
 table=torch.zeros((len(qsl)-1,1128),dtype=torch.int32);table[:,0]=1
 slots=torch.empty(ninput,dtype=torch.int64)
 obj=types.SimpleNamespace(device=torch.device("cpu"),dcp_size=8,replicated_view_block_size=128,_ensure_replicated_view_buffers=lambda *args:(table,None,slots))
 c=types.SimpleNamespace(num_reqs=len(qsl)-1,num_input_tokens=ninput,num_actual_tokens=nactual,query_start_loc=torch.tensor(qsl,dtype=torch.int32),query_start_loc_cpu=torch.tensor(qsl,dtype=torch.int32),positions=torch.tensor(positions,dtype=torch.int64))
 return obj,c,table
rows=[]
for name,args,expect in [
 ("unpadded_1",([0,1],1,1,[19]),[147]),
 ("padded_extended_actual1",([0,1,2,3],1,3,[19,0,0]),[147,-1,-1]),
 ("padded_flat_actual3",([0,1,1,1],3,3,[19,20,21]),"error"),
 ("padded_flat_actual1",([0,1,1,1],1,3,[19,0,0]),"error")]:
 obj,c,table=fixture(*args)
 try:
  got=method(obj,c,table).tolist();error=None;assert got==expect
 except RuntimeError as e:
  got=None;error=str(e);assert expect=="error"and("allocated size does not match required size"in error or"output_size"in error)
 rows.append(dict(name=name,query_CPU=c.query_start_loc_cpu.tolist(),num_actual_tokens=c.num_actual_tokens,num_input_tokens=c.num_input_tokens,outputs=got,error=error,table_shape=list(table.shape),repeat_output_size=c.num_input_tokens,query_sum=int(c.query_start_loc_cpu[-1])))
# CPU candidate calculation only: preserve allocated input-slot length/pad=-1, size repeat by actual query extent.
candidate=source.replace("num_actual_tokens = min(common_attn_metadata.num_actual_tokens, num_input_tokens)","query_extent = int(common_attn_metadata.query_start_loc_cpu[num_reqs] - common_attn_metadata.query_start_loc_cpu[0])\n        num_actual_tokens = min(common_attn_metadata.num_actual_tokens, num_input_tokens, query_extent)").replace("output_size=num_input_tokens,","output_size=query_extent,")
import textwrap
scope=dict(vars(sfa_cp),torch=torch);exec(compile(textwrap.dedent(candidate),"<CPU candidate>", "exec"),scope);fixed=scope[method.__name__]
fixedrows=[]
for name,args,expect in [
 ("unpadded_1",([0,1],1,1,[19]),[147]),
 ("padded_extended_actual1",([0,1,2,3],1,3,[19,0,0]),[147,-1,-1]),
 ("padded_flat_actual3",([0,1,1,1],3,3,[19,20,21]),[147,-1,-1]),
 ("padded_flat_actual1",([0,1,1,1],1,3,[19,0,0]),[147,-1,-1]),
 ("empty_queries",([0,0,0,0],0,3,[0,0,0]),[-1,-1,-1])]:
 obj,c,table=fixture(*args);got=fixed(obj,c,table).tolist();assert got==expect;fixedrows.append(dict(name=name,outputs=got))
(j/"CPU_candidate_source.py.txt").write_text(candidate)
out=dict(native_class=cls.__name__,native_method_source_SHA=hashlib.sha256(source.encode()).hexdigest(),native_module=dict(path=str(actual),sha256=hashlib.sha256(actual.read_bytes()).hexdigest()),native_fixtures=rows,CPU_candidate_fixtures=fixedrows,CPU_only=True,NPU_tensors=0,newmodels=0,newinference=0,native_patched=False,scope="Native actual method on CPU SimpleNamespace fixture; allocator helper fixture notnative constructor; synthetic metadata not actual217GPU input; CPU candidate notinstalled",limits=["PyTorchCPU repeat_interleave size mismatch verified; native217CANN values257/258 andextent3384 compatible withhypothesis butnotGPUrootcause proof","ActualruntimequeryGPU/CPU andpadding counts pending219 metadata observation; no correctness/performance/KEEP fromCPUfixture"])
(j/"probe.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(dict(CPU_fixture_VALID=True,native_rows=4,candidate_rows=5,NPU_tensors=0)))
