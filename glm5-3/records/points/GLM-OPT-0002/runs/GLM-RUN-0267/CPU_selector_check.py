"""Test-only selector parity and constant-off Graph observer; CPU doubles only."""
from __future__ import annotations
import ast,importlib.util,itertools,json,os,sys,tempfile,types
from pathlib import Path
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parent
RESEARCH=ROOT.parents[1]/'research/decode_path_20261006'
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
cpu=module('mtp_actual_CPU',RESEARCH/'early_padded_mtp/check_CPU_v4.py')
# Extract the actual test-only functions, not a reimplementation.
tree=ast.parse((ROOT/'shim/model_runner_v1.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('_h10_') or isinstance(n,ast.Assign) and all(isinstance(t,ast.Name) and t.id.startswith('_H10_') for t in n.targets)]
ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_H10_selector','exec'),ns)
fake=types.ModuleType('vllm.distributed');fake.get_tp_group=lambda:NS(rank_in_group=0);sys.modules['vllm.distributed']=fake
checked=0;transitions=[]
with tempfile.TemporaryDirectory() as td:
 p=Path(td);(p/'witnesses').mkdir();(p/'candidate').mkdir();(p/'candidate/model_runner_v1.py').write_bytes((ROOT/'candidate/model_runner_v1.py').read_bytes());(p/'early_draft_mode.bin').write_bytes(b'\x00');os.environ['GLM_EARLY_MTP_DIAGNOSTIC_ROOT']=td
 runner=NS(input_batch=NS(sampling_metadata=NS(all_greedy=True),num_reqs=1,generators={}),vllm_config=NS(compilation_config=NS(cudagraph_capture_sizes=[2],cudagraph_mode='FULL_DECODE_ONLY')),use_async_scheduling=False,num_spec_tokens=1,num_prompt_logprobs={},routed_experts_initialized=False,model_config=NS(hf_config=NS(model_type='glm_moe_dsa')),speculative_config=NS(enforce_eager=True))
 for mode in (0,1,0,1):
  with (p/'early_draft_mode.bin').open('r+b') as f:f.write(bytes([mode]));f.flush();os.fsync(f.fileno())
  for settings in itertools.product((1,2),(False,True),('mtp','eagle'),(False,True),(False,True),(False,True),(False,True),(False,True),(1,2),(0,1),('glm_moe_dsa','other'),(False,True)):
   old,new=cpu.predicate(False,settings),cpu.predicate(True,settings)
   got=ns['_h10_select'](new,settings[0]>1,runner,NS(num_spec_tokens_to_schedule=1),object(),NS(uniform=True,num_tokens=2))
   assert got==(new if mode else old),(mode,settings,got,old,new);checked+=1
  row=json.loads((p/'witnesses'/('mtp_mode%d_rank0.json'%mode)).read_text());transitions.append(row['transition_count']);assert row['mode']==mode
 assert transitions==[1,2,3,4]
 old=(p/'witnesses/mtp_mode1_rank0.json').stat().st_mtime_ns
 for _ in range(3):assert ns['_h10_select'](True,False,runner,NS(num_spec_tokens_to_schedule=1),object(),NS(uniform=True,num_tokens=2))
 assert (p/'witnesses/mtp_mode1_rank0.json').stat().st_mtime_ns==old
 os.environ.pop('GLM_EARLY_MTP_DIAGNOSTIC_ROOT');assert not ns['_h10_select'](True,False,runner,None,None,None);assert ns['_h10_select'](True,True,runner,None,None,None)
 graphcpu=module('constant_off_graph_CPU',RESEARCH/'stream_ordered_replay/check_stream_order_CPU.py');(p/'stream_order_mode.bin').write_bytes(b'\x00');os.environ['GLM_STREAM_ORDER_DIAGNOSTIC_ROOT']=td
 shim=graphcpu.setup(False,ROOT/'shim/breakable_aclgraph.py');baseline=graphcpu.setup(False);wrapper,entry=shim[3],shim[5]
 wrapper.vllm_config=NS(compilation_config=NS(mode=0,cudagraph_mode='FULL_DECODE_ONLY',cudagraph_capture_sizes=[2]),model_config=NS(enforce_eager=False),speculative_config=NS(enforce_eager=True));entry.batch_descriptor=NS(num_tokens=2);entry.capture.num_graphs=1;entry.capture.num_eager_breaks=0
 graphcases=0
 for settings in itertools.product((graphcpu.Mode.NONE,graphcpu.Mode.FULL,graphcpu.Mode.PIECEWISE),(False,True),(False,True),('ASCEND_SFA','ASCEND_MLA'),(False,True),('glm_moe_dsa','other',None),(False,True),(False,True),(False,True)):
  expected,_=graphcpu.case(baseline,settings);actual,_=graphcpu.case(shim,settings);assert actual==expected;graphcases+=1
 os.environ.pop('GLM_STREAM_ORDER_DIAGNOSTIC_ROOT')
assert not (hasattr(cpu.torch,'npu') and cpu.torch.npu.is_initialized())
result=dict(passed=True,actual_selector_branch_cases=checked,mode_transitions=transitions,repeated_same_mode_witness_write=False,missing_environment_baseline=True,existing_PP_route_preserved=True,constant_off_graph_observer_cases=graphcases,production_CPU_result='early_padded_mtp/CPU_result_v4.json',NPU_initialized=False,performance_claim=False)
(ROOT/'CPU_selector_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
