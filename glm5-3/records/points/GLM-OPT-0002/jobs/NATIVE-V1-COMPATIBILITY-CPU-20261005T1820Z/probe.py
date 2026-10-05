from pathlib import Path
import asyncio,json,hashlib,sys
j=Path(__file__).parent;b=j/"prototype_bundle";sys.path.insert(0,str(b))
from capability_placement import CapabilityPlacement,request_features
from native_engines_service_config import compile_config,checked_config
def ref(f):
 raw=Path(f).read_bytes();return dict(path=str(f),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
replicas=[dict(id="D0",url="http://native-v2.invalid"),dict(id="D1",url="http://native-v1.invalid")]
groups=[dict(id="V2",epoch="a"*64,members=["D0"]),dict(id="V1",epoch="b"*64,members=["D1"])]
shape=dict(input_threshold_bytes=100,prefill_members=["D1"],decode_members=["D0"])
compat={k:["D1"]for k in ["structured_output","thinking_budget"]}
def make(**kw):return CapabilityPlacement(replicas,"shape_split_idle_spill",groups,None,shape,**kw)
def body(v):return json.dumps(v,separators=(",",":")).encode()
checks=[]
async def reject(coro,contains):
 try:await coro
 except RuntimeError as e:assert contains in str(e)
 else:raise AssertionError("unexpected compatibility admission")
async def main():
 p=make(compatibility_config=compat)
 plain=body(dict(messages=[],tool_choice="auto"));assert not request_features("/v1/chat/completions","POST",plain)
 required=body(dict(messages=[],tool_choice="required"));frozen=bytes(required)
 cases=[dict(tool_choice="required"),dict(tool_choice={"type":"function","function":{"name":"x"}}),dict(response_format={"type":"json_schema"}),dict(structured_outputs={"regex":"[0-9]+"}),dict(text={"format":{"type":"json_schema"}}),dict(thinking_token_budget=32)]
 for v in cases:
  raw=body(v);lease=await p.acquire_request(8,1,None,"/v1/chat/completions","POST",raw);assert lease.replica.key=="D1";await p.release(lease,False)
 assert required==frozen
 assert not request_features("/v1/chat/completions","GET",required)
 assert not request_features("/tokenize","POST",required)
 assert not request_features("/v1/chat/completions","POST",b"{")
 assert not request_features("/v1/chat/completions","POST",b"[]")
 assert not request_features("/v1/chat/completions","POST",body(dict(response_format={"type":"text"},tool_choice="none")))
 ls=await asyncio.gather(*[p.acquire_request(8,1,None,"/v1/chat/completions","POST",required)for _ in range(32)])
 assert len(p.leases)==32 and all(x.replica.key=="D1"for x in ls)
 other=await p.acquire_request(8,1,None,"/v1/chat/completions","POST",plain);assert other.replica.key=="D0"
 for lease in ls+[other]:assert await p.release(lease,False)
 checks.append("concurrent structured admission filters V1 before shape/load; plain short retains V2; all leases conserve")
 owner1=dict(replica="D1",url=replicas[1]["url"],group="V1",epoch="b"*64)
 owner0=dict(replica="D0",url=replicas[0]["url"],group="V2",epoch="a"*64)
 lease=await p.acquire_request(8,1,owner1,"/v1/responses","POST",required);assert lease.replica.key=="D1";await p.release(lease,False)
 await reject(p.acquire_request(8,1,owner0,"/v1/responses","POST",required),"no cross-owner replay")
 bad=dict(owner1,epoch="c"*64)
 await reject(p.acquire_request(8,1,bad,"/v1/responses","POST",required),"native response owner unavailable")
 assert not p.leases
 checks.append("bound owner preserved; incompatible V2 or stale epoch rejected before lease/RPC")
 await p.remove("D1")
 await reject(p.acquire_request(8,1,None,"/v1/chat/completions","POST",required),"no eligible compatible")
 lease=await p.acquire_request(8,1,None,"/v1/chat/completions","POST",plain);assert lease.replica.key=="D0";await p.release(lease,False)
 checks.append("drained V1 never spills structured requests to incompatible V2")
 await p.close()
 p=make(compatibility_config=compat)
 lease=await p.acquire_request(8,1,None,"/v1/chat/completions","POST",required);await p.release(lease,True)
 await reject(p.acquire_request(8,1,None,"/v1/chat/completions","POST",required),"no eligible compatible")
 assert p.member_group["D1"].faulted and not p.member_group["D0"].faulted
 await p.close();checks.append("native backend failure preserves whole-group quarantine")
 p=make()
 lease=await p.acquire_request(8,1,None,"/v1/chat/completions","POST",required);assert lease.replica.key=="D0";await p.release(lease,False);await p.close()
 checks.append("opt-out uses unchanged shape/owner admission")
asyncio.run(main())
p=j.parent.parent
material=json.loads((p/"runs/GLM-RUN-0228/restored/native_engines_resident.json").read_text())
source=[ref(p/"jobs/AUDIT-RUN236-STRUCTURED-20261005T1710Z/reduction.json"),ref(p/"jobs/AUDIT-RUN238-V1-COMPAT-20261005T1750Z-v2/reduction.json")]
baseline=compile_config(material,j/"CPU_state");assert "compatibility"not in baseline and "GLM_COMPATIBILITY_MEMBERS"not in baseline["environment"]
material["compatibility"]=dict(kind="native_v1",sources=source)
v=compile_config(material,j/"CPU_state")
assert v["compatibility"]["members"]==compat and json.loads(v["environment"]["GLM_COMPATIBILITY_MEMBERS"])==compat
f=j/"CPU_resident.json";f.write_text(json.dumps(material))
v["resident_evidence"]={"native_engines_resident.json":dict(path=str(f),sha256=ref(f)["sha256"])}
c=j/"CPU_config.json";c.write_text(json.dumps(v));assert checked_config(c)[0]==v
bad=json.loads(json.dumps(material));bad["compatibility"]["sources"][0]["sha256"]="0"*64
try:compile_config(bad,j/"CPU_state")
except ValueError:pass
else:raise AssertionError("changed evidence accepted")
checks.append("compatibility compiled from native runner plans, pinned source236/238 and owner epochs; changed evidence rejected")
out=dict(CPU_contract_VALID=True,checks=checks,prototype_sources=[ref(b/n)for n in ["capability_placement.py","response_affinity_gateway_v11.py","native_engines_service_config.py"]],native_models_or_SDK_or_NPU_calls=0,active_runtime_changed=False,limits=["Synthetic admission and recorded evidence checks only; gateway protocol and live native functionality remain unverified","V2 grammar cause unresolved; native thinking-budget and broader full API need real E2E","Existing bound V2 structured request fails availability before RPC; no STORE migration/replay"])
f=j/"reduction.json";f.write_text(json.dumps(out,indent=2)+"\n")
v=dict(schema_version=1,job_id=j.name,status="completed",summary="CPU V1 compatibility admission valid: concurrency, shape, owner-conflict, drain, group-fault, opt-out and pinned native provenance; no activation/GPU",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(f),locator="CPU compatibility eligibility/affinity/fault/provenance")],unknowns=out["limits"],decision_request=None,next_check_at=None);(j/"result.json").write_text(json.dumps(v,indent=2)+"\n");print(json.dumps(dict(CPU_VALID=True,checks=len(checks),**ref(f))))
