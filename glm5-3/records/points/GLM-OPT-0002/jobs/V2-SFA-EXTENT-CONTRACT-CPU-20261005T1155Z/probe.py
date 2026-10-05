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
# Exact manual expectations for active queries, crossing a logical block,
# a request with zero query tokens, and allocated padding slots.
rows=[]
cases=[
 ("one",([0,1],1,1,[19],[[1]*8]),[147]),
 ("flat_pad",([0,1,1,1],3,3,[19,20,21],[[1]*8,[2]*8,[3]*8]),[147,-1,-1]),
 ("flat_pad_actual1",([0,1,1,1],1,3,[19,20,21],[[1]*8,[2]*8,[3]*8]),[147,-1,-1]),
 ("extended_pad",([0,1,2,3],1,3,[19,0,0],[[1]*8,[2]*8,[3]*8]),[147,-1,-1]),
 ("zero",([0,0,0],0,3,[0,0,0],[[1]*8,[2]*8]),[-1,-1,-1]),
 ("crossblock_zeroquery",([0,0,2,2,5],5,5,[127,128,129,130,131],[[1]*8,[2,5,0,0,0,0,0,0],[3]*8,[4,7,0,0,0,0,0,0]]),[383,640,897,898,899]),
 ("mixed_padding",([0,0,2,2,5],5,8,[127,128,129,130,131,0,0,0],[[1]*8,[2,5,0,0,0,0,0,0],[3]*8,[4,7,0,0,0,0,0,0]]),[383,640,897,898,899,-1,-1,-1]),
]
for name,args,expected in cases:
 obj,c,t=fixture(*args);got=contract.build_slot_mapping(obj,c,t).tolist();assert got==expected
 if c.num_input_tokens==contract.query_extent(c):assert native(obj,c,t).tolist()==expected
 rows.append(dict(name=name,expected=expected,actual=got))
invalid=[]
for name,args in [
 ("negative_query",([0,-1],0,1,[0],[[1]*8])),
 ("too_long",([0,2],1,1,[0],[[1]*8])),
 ("actual_too_long",([0,1],2,1,[0],[[1]*8])),
 ("nonmonotone",([0,2,1],1,3,[0,0,0],[[1]*8,[2]*8]))]:
 obj,c,t=fixture(*args)
 try:contract.build_slot_mapping(obj,c,t);raise AssertionError("invalid metadata accepted")
 except ValueError as e:invalid.append(dict(name=name,error=str(e)))
contract.install()
for name,args,expected in cases:
 obj,c,t=fixture(*args);assert cls._build_slot_mapping_replicated_view(obj,c,t).tolist()==expected
assert Path(__import__("inspect").getsourcefile(native)).read_bytes()==before
cls._build_slot_mapping_replicated_view=native;del cls._glm_query_extent_contract_installed
out=dict(CPU_contract_VALID=True,manual_mapping_cases=rows,invalid_boundary_cases=invalid,installed_in_CPU_process_only=True,native_file_unchanged=True,NPU_tensors=0,modelstarts=0,inference=0,operator_implementation_changes=0,metadata_only=True,runtime_activation=False,native_module_sha=hashlib.sha256(before).hexdigest(),candidate_module_sha=hashlib.sha256(Path(contract.__file__).read_bytes()).hexdigest(),limits=["Synthetic CPU fixtures/manualmapping/source guard; notactualGraphGPUproof orperformanceKEEP","Experimentalmodule notactivated inanynativeworker; current221 diagnostic remainsunmodified nativeSFAMethod apartobservation","Must verifyactualsameGraph querypadding field beforeactivation; fullnativeAPI/knownthinkingbudgetgap remainsopen"])
(j/"probe.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(dict(CPU_VALID=True,mapping_cases=7,invalid_cases=4,noNPU=True)))
