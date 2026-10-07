"""Actual H9 shim AST against production/off baselines; no torch or NPU import."""
import importlib.util,itertools,json,os,sys,tempfile,types
from pathlib import Path

ROOT=Path(__file__).resolve().parent
path=ROOT.parents[1]/'research/decode_path_20261006/stream_ordered_replay/check_stream_order_CPU.py'
spec=importlib.util.spec_from_file_location('actual_H9_CPU_contract',path);cpu=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpu)
fake=types.ModuleType('vllm.distributed');fake.get_tp_group=lambda:types.SimpleNamespace(rank_in_group=0)
sys.modules['vllm.distributed']=fake
checked=0;writes=[]
with tempfile.TemporaryDirectory() as td:
    directory=Path(td);(directory/'witnesses').mkdir();modefile=directory/'stream_order_mode.bin';modefile.write_bytes(b'\x00')
    os.environ['GLM_STREAM_ORDER_DIAGNOSTIC_ROOT']=td
    shim=cpu.setup(True,ROOT/'shim/breakable_aclgraph.py');wrapper,entry=shim[3],shim[5]
    config=types.SimpleNamespace(compilation_config=types.SimpleNamespace(mode=0,cudagraph_mode='FULL_DECODE_ONLY',cudagraph_capture_sizes=[2]),model_config=types.SimpleNamespace(enforce_eager=False),speculative_config=types.SimpleNamespace(enforce_eager=True))
    wrapper.vllm_config=config;entry.batch_descriptor=types.SimpleNamespace(num_tokens=2)
    entry.capture.num_graphs=1;entry.capture.num_eager_breaks=0
    baseline={0:cpu.setup(False),1:cpu.setup(True)}
    for mode in (0,1,0,1):
        with modefile.open('r+b') as file:file.write(bytes([mode]));file.flush();os.fsync(file.fileno())
        for settings in itertools.product((cpu.Mode.NONE,cpu.Mode.FULL,cpu.Mode.PIECEWISE),(False,True),(False,True),('ASCEND_SFA','ASCEND_MLA'),(False,True),('glm_moe_dsa','other',None),(False,True),(False,True),(False,True)):
            expected,_=cpu.case(baseline[mode],settings);actual,_=cpu.case(shim,settings)
            assert actual==expected,(mode,settings,actual,expected);checked+=1
        row=json.loads((directory/'witnesses'/('mode%d_rank0.json'%mode)).read_text());writes.append(row['transition_count']);assert row['mode']==mode
    assert writes==[1,2,3,4],writes
    safe=(cpu.Mode.FULL,True,False,'ASCEND_SFA',False,'glm_moe_dsa',False,False,False)
    old=(directory/'witnesses/mode1_rank0.json').stat().st_mtime_ns
    for _ in range(3):cpu.case(shim,safe)
    assert (directory/'witnesses/mode1_rank0.json').stat().st_mtime_ns==old
    os.environ.pop('GLM_STREAM_ORDER_DIAGNOSTIC_ROOT');expected,_=cpu.case(baseline[0],safe);actual,_=cpu.case(shim,safe);assert actual==expected
result=dict(passed=True,actual_selector_branch_cases=checked,mode_transitions=writes,repeated_same_mode_witness_write=False,disabled_environment_preserves_host_sync=True,NPU_initialized=False,CPU_queue_doubles=True,performance_claim=False)
(ROOT/'CPU_selector_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
