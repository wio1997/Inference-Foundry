from pathlib import Path
import sys,hashlib,json,types,torch,copy
j=Path(__file__).parent;sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
import sfa_dcp_query_extent_contract as contract
from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPMetadataBuilder as cls
native=cls._build_slot_mapping_replicated_view
before=Path(__import__("inspect").getsourcefile(native)).read_bytes()
def fixture(qsl,actual,ninput,positions,table):
 t=torch.tensor(table,dtype=torch.int32);slots=torch.empty(ninput,dtype=torch.int64)
 obj=types.SimpleNamespace(device=torch.device("cpu"),dcp_size=8,replicated_view_block_size=128,_ensure_replicated_view_buffers=lambda *args:(t,None,slots))
 c=types.SimpleNamespace(num_reqs=len(qsl)-1,num_input_tokens=ninput,num_actual_tokens=actual,query_start_loc=torch.tensor(qsl,dtype=torch.int32),query_start_loc_cpu=torch.tensor(qsl,dtype=torch.int32),positions=torch.tensor(positions,dtype=torch.int64))
 return obj,c,t
# CPU thought experiment: host capture boundaries remain extended while live
# GPU-side query boundaries flatten padding. This is not an observed221draft.
obj,c,t=fixture([0,1,2,3],3,3,[19,0,0],[[1]*8,[2]*8,[3]*8])
capture_cpu=c.query_start_loc_cpu.tolist();c.query_start_loc.copy_(torch.tensor([0,1,1,1],dtype=torch.int32))
assert contract.query_extent(c)==3
errors=[]
for name,fn in [("original_native",native),("experimental_build",contract.build_slot_mapping)]:
 try:fn(obj,c,t);raise AssertionError("stalecapture extent unexpectedly accepted")
 except RuntimeError as e:
  assert "allocated size does not match required size" in str(e);errors.append(dict(path=name,error=str(e)))
assert Path(__import__("inspect").getsourcefile(native)).read_bytes()==before
out=dict(CPU_contract_VALID=True,synthetic_capture_boundary_counterexample=True,CPU_capture_boundaries=capture_cpu,live_device_boundaries=c.query_start_loc.tolist(),errors=errors,candidate_extent=contract.query_extent(c),NPU_tensors=0,modelstarts=0,inference=0,runtime_activation=False,native_file_unchanged=True,limits=["Synthetic hostcapture/device replay contract mismatch; NOT observedactual221draftargs/rootcause","ExperimentalCPUextent candidate relies hostCPU queryextent; it cannotfix this stalehostcaptured-input scenario, notselected","Does notprove actualGraph executes these operands; real222 draftNONE isolation decides narrower failure dependency"])
(j/"probe.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(dict(counterexample=True,NPU_tensors=0)))
