from pathlib import Path
import sys,json,hashlib,re
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;p=j.parents[1];r=p/"runs/GLM-RUN-0049";audit=p/"jobs/REDUCE-PD49-NATIVE-20261002T0330Z"
def ref(f):
 d=f.read_bytes();return{"path":str(f),"bytes":len(d),"sha256":hashlib.sha256(d).hexdigest()}
assert ref(audit/"reduction.json")["sha256"]=="c690e123f58343b2b4e5e4a7ed9e9e4235c566ea4843b795fae10b869101e7e5"
a=json.loads((audit/"reduction.json").read_text());assert a["inference_attempts"]==0 and a["state"]["status"]=="failed"and a["source_pins"]==18
details={}
for rank in [0,1]:
 f=audit/("D"+str(rank)+".native.log");text=f.read_text();lines=text.splitlines()
 errors=[(i,l)for i,l in enumerate(lines)if"Communication_Error_Bind_IP_Port(EI0020)"in l and"port 16666 have already been bound"in l]
 assert errors;idx,line=errors[0]
 details["D"+str(rank)]={"first_port_error":line,"surrounding":lines[max(0,idx-4):idx+5],"all_bind_conflict_lines":[v for _,v in errors],"raw":ref(f),"OOM_error_observed":"OutOfMemoryError"in text,"failed_stack":"NativeDeepseekV2MoE quantizationinitializer ->W8A8Dynamic.get_hccl_comm_name ->hcclCommInitRootInfoConfig code7; deviceNICport16666 EI0020"}
 assert not details["D"+str(rank)]["OOM_error_observed"]
memory={}
for node in["166","167"]:
 f=r/("resident_P_"+node+".npu-smi");text=f.read_text();hb=[int(z)for z in re.findall(r"\|\s+[01]\s+[0-9]+\s+\|[^\n]*?\s+([0-9]+)/\s*65536\s+\|",text)];assert len(hb)==16
 proc=[int(z)for z in re.findall(r"VLLMWorker_DP\s+\|\s+([0-9]+)",text)];assert len(proc)==16
 memory[node]={"raw":ref(f),"board_HBM_MB":{"min":min(hb),"max":max(hb),"per_chip":hb,"total_MB":65536},"native_P_process_memory_MB":{"min":min(proc),"max":max(proc)},"native_KV_capacity_per_DP":91924}
out={"at":utc(),"run_id":r.name,"classification":"nativeHCCLmulti-processdeviceNICportconflict beforeDmodelinitializationcompleted","verdict":"REJECT","scope":"Specificdualresidentnativeconfiguration withdefaultHCCLNPU16666; no memory/capacity/hardwarelimit","terminal_audit":ref(audit/"reduction.json"),"state":a["state"],"D_native_errors":details,"P_resident_memory":memory,"inference_attempts":0,"effective_output_tokens":0,"NPU_workers":"P32resident, D32initmetadata recorded thenfailedduringmodelconstructor; not0workers","next_legitimate_config":{"HCCL_NPU_SOCKET_PORT_RANGE":{"P":"63000-63063","D":"63100-63163"},"HCCL_HOST_SOCKET_PORT_RANGE":{"P":"62400-62463","D":"62500-62563"},"native_source_operator_changes":0},"official_sources":[{"url":"https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/maintenref/envvar/envref_07_0144.html","claim":"Rootinfocommunicator supportsNPUport ranges/A3 sharedNPUprocesses; default16666"},{"url":"https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/API/hcclug/docs/zh/user_guide/hccl_env/HCCL_CONNECT_TIMEOUT.md","claim":"HCCL_HOST_SOCKET_PORT_RANGE ranges/priorityoverIFbase andNPUranges; onlineversionnotexactinstalledCANN9.1build"}],"signals":0,"models":0,"new_inference":0,"limits":["Hostss checks do notinventorydeviceNICports","NoOOM/weightcomplete/dualresidentfit/PDtransfer/runtimeclaim","Earlierterminalauditor genericlimits mentioned successful7pilot condition, actual49had0requests; correctedcuratedstatement here/rawfrozen","UniqueengineIDs nativeCPmappingalias separateCPUcontrolproof; no49actualPD"]}
atomic_json(j/"reduction.json",out);brief=json.loads((r/"reduction_brief.json").read_text());brief.update(verdict="REJECT",classification=out["classification"],HCCL_failure_detail=ref(j/"reduction.json"),limits=out["limits"]);atomic_json(r/"reduction_brief.json",brief)
m=json.loads((r/"manifest.json").read_text());m.update(verdict="REJECT",status="failed",valid=False,HCCL_failure_detail=ref(j/"reduction.json"));atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# "+r.name+"\n\nREJECT specificdefaultHCCLNPU16666dualresidentstartup. P32workers fullyresident/healthy, KV91924tokens perDP andboardHBM33924..34280MB/65536; D32workers beganinitialization thennativeMoEW8A8Dynamic communicator construction failed withhcclCommInitRootInfoConfig code7/EI0020: NPUIP/16666 alreadybound. NoOOMobserved, noDmodelcomplete/PDfit/requests/outputs/totalhardwarebound. Terminal18pins/native9files/owners auditauthentic. New50nativeA3supportedroleNPU/HOSTport_ranges plusuniqueper-schedulerIDs, no operators/vendorpatch. Actualdualresidentmemory/transport unknown, CurrentNone.\n")
atomic_json(j/"result.json",{"schema_version":1,"job_id":j.name,"status":"completed","summary":"49nativeHCCL16666 EI0020 duringDmodelconstructor REJECTspecificdefaultport config; P32healthyresidentHBM/KVquantified, D32initfailed,0requests/noOOMproof; officialsupportedroleports next50","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[],"evidence":[dict(id="reduction",locator="frozen49rawHCCLtrace/Pmemory andearlier18pinaudit",**ref(j/"reduction.json"))],"unknowns":out["limits"],"decision_request":None,"next_check_at":None})
print(json.dumps({"verdict":"REJECT","failure":"HCCLNPU16666alreadybound","P_HBM_MB":{k:v["board_HBM_MB"]for k,v in memory.items()},"reduction":ref(j/"reduction.json")}))

