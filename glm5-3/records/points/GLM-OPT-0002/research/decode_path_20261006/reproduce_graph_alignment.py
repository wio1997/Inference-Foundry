"""Execute actual saved graph dispatch/sizing functions without Torch or NPU."""
from pathlib import Path
import ast,dataclasses,enum,hashlib,itertools,json,logging,sys,types
r=Path(sys.argv[1]);sources=r/'graph_sources'
ns=dict(enum=enum,replace=dataclasses.replace,product=itertools.product,logger=logging.getLogger('saved-graph-source'),VllmConfig=object)
def execute_nodes(path,names):
 text=path.read_text();tree=ast.parse(text);nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
 assert {n.name for n in nodes}==set(names)
 module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
 exec(compile(ast.fix_missing_locations(module),str(path),'exec'),ns)
execute_nodes(sources/'vllm__vllm__config__compilation.py',['CUDAGraphMode'])
rows=json.loads((sources/'descriptor_source_snippets.json').read_text());snippet=rows[0]['snippets'][0]
exec('from __future__ import annotations\n'+snippet['text'],ns);ns['BatchDescriptor']=dataclasses.dataclass(frozen=True)(ns['BatchDescriptor'])
execute_nodes(sources/'vllm__vllm__v1__cudagraph_dispatcher.py',['CudagraphDispatcher'])
execute_nodes(sources/'vllm-ascend__vllm_ascend__platform.py',['_get_default_max_cudagraph_capture_size'])
exec('from __future__ import annotations\n'+rows[1]['snippets'][0]['text'],ns)
E=ns['CUDAGraphMode'];D=ns['CudagraphDispatcher'];S=types.SimpleNamespace
output=[]
for maxseq in (4,8):
 for mode in (E.FULL_DECODE_ONLY,E.FULL_AND_PIECEWISE):
  config=S(compilation_config=S(cudagraph_capture_sizes=[16,32],max_cudagraph_capture_size=32,compile_sizes=[],cudagraph_specialize_lora=False),scheduler_config=S(max_num_seqs=maxseq),lora_config=None,num_speculative_tokens=1,parallel_config=S(tensor_parallel_size=16),speculative_config=S(num_speculative_tokens=1))
  d=D.__new__(D);d.vllm_config=config;d.compilation_config=config.compilation_config;d.uniform_decode_query_len=2;d.specialize_lora_count=False;d.cudagraph_keys={E.PIECEWISE:set(),E.FULL:set()};d.keys_initialized=False;d.cudagraph_mode=E.NONE
  d.initialize_cudagraph_keys(mode,2);runtime,desc=d.dispatch(16,uniform_decode=True)
  output.append(dict(maxseq=maxseq,mode=mode.name,FULL_keys=len(d.cudagraph_keys[E.FULL]),runtime=runtime.name,descriptor=dataclasses.asdict(desc)))
  if maxseq==4:assert len(d.cudagraph_keys[E.FULL])==0 and runtime!=E.FULL
  else:assert len(d.cudagraph_keys[E.FULL])==1 and runtime==E.FULL
config=S(compilation_config=S(cudagraph_capture_sizes=None,max_cudagraph_capture_size=None),scheduler_config=S(max_num_seqs=4),speculative_config=S(num_speculative_tokens=1),parallel_config=S(tensor_parallel_size=16))
default_max=ns['_get_default_max_cudagraph_capture_size'](config)
default_tp_sizes=ns['update_sizes_for_sequence_parallelism'](config,list(range(1,default_max+1)))
assert default_max==8 and default_tp_sizes==[]
# Execute the unchanged scalar dummy-run request construction and assertions.
runner=r/'vllm-ascend__vllm_ascend__worker__model_runner_v1.py';text=runner.read_text();node=next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name=='_dummy_run')
branch=next(n for n in node.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='create_mixed_batch').orelse[0]
checks=[n for n in node.body if isinstance(n,ast.Assert) and ('sum(num_scheduled_tokens_list)' in ast.get_source_segment(text,n) or 'len(num_scheduled_tokens_list)' in ast.get_source_segment(text,n))]
assert len(checks)==2
code=ast.Module(body=[ast.If(test=branch.test,body=branch.body,orelse=[])]+checks,type_ignores=[])
dummy_ns=dict(uniform_decode=True,max_num_reqs=4,num_tokens=16,max_query_len=2,cdiv=lambda a,b:(a+b-1)//b)
failed=False
try:exec(compile(ast.fix_missing_locations(code),str(runner),'exec'),dummy_ns)
except AssertionError:failed=True
assert failed and sum(dummy_ns['num_scheduled_tokens_list'])==8
identities={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [sources/'vllm__vllm__config__compilation.py', sources/'descriptor_source_snippets.json', sources/'vllm__vllm__v1__cudagraph_dispatcher.py', sources/'vllm-ascend__vllm_ascend__platform.py',runner]}
result=dict(source_identity=identities,CPU_only=True,actual_saved_source_execution=True,performance_gain=None,graph_model_correctness=False,stock_dispatch_cases=output,default_max=default_max,default_TP_aligned_sizes=default_tp_sizes,capture_dummy=dict(physical_tokens=16,maxseq=4,uniform_len=2,scheduled_list=dummy_ns['num_scheduled_tokens_list'],scheduled_sum=8,assertion_failed=True),scope='Inactive under current enforce_eager/NONE; source blocker to replay activation, not a second performance candidate.')
(r/'graph_alignment_reproduction.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
