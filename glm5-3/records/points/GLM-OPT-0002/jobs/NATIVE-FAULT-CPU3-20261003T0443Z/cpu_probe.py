import asyncio,json,sys,hashlib
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
import httpx
from response_affinity_gateway_v11 import create_app
j=Path(sys.argv[1]);state=j/"fixture_state";state.mkdir()
replicas=[dict(id="D0",url="http://CPU-D0"),dict(id="D1",url="http://CPU-D1")]
groups=[dict(id="local-166",epoch="CPU-epoch-D0",members=["D0"]),dict(id="local-167",epoch="CPU-epoch-D1",members=["D1"])]
fixture=json.loads((j/"native_response_fixture.json").read_text());calls=[];contracts=[];contexts={}
class Body(httpx.AsyncByteStream):
 def __init__(self,raw):self.raw=raw
 async def __aiter__(self):yield self.raw
def upstream(request):
 calls.append(dict(method=request.method,url=str(request.url)))
 body=dict(fixture,id=request.url.path.rsplit("/",1)[-1])
 return httpx.Response(200,stream=Body(json.dumps(body).encode()),headers={"Content-Type":"application/json"})
async def app_for(gs,path):
 app=create_app(config=replicas,policy="active_count",groups=gs,response_owner_state_path=str(path/"owners.json"),fault_state_path=str(path/"fault.json"),transport=httpx.MockTransport(upstream))
 context=app.router.lifespan_context(app);await context.__aenter__();contexts[id(app)]=context;return app
async def close(app):
 await contexts.pop(id(app)).__aexit__(None,None,None)
async def main():
 app=await app_for(groups,state);place=app.state.placement;idx=app.state.response_owner_index
 a=await place.acquire_for_owner(32,21,None);b=await place.acquire_for_owner(32,21,None);assert a.replica.key=="D0"and b.replica.key=="D1"
 idx.bind("resp_CPU_D0",a);idx.bind("resp_CPU_D1",b);await place.release(a)
 owner0=idx.resolve("resp_CPU_D0");owner1=idx.resolve("resp_CPU_D1")
 async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://CPU")as client:
  for name,key,body in[
   ("unknown_group","missing",dict(epoch="CPU-epoch-D1",reason="CPU identity loss")),
   ("stale_epoch","local-167",dict(epoch="CPU-stale",reason="CPU identity loss")),
   ("bool_epoch","local-167",dict(epoch=True,reason="CPU identity loss")),
   ("missing_reason","local-167",dict(epoch="CPU-epoch-D1")),
   ("extra_clear","local-167",dict(epoch="CPU-epoch-D1",reason="CPU identity loss",faulted=False)),
   ("oversize_reason","local-167",dict(epoch="CPU-epoch-D1",reason="x"*513))]:
   res=await client.post("/control/native-groups/"+key+"/fault",json=body);assert res.status_code==409and not place.execution_groups["local-167"].faulted
   contracts.append(name+"_rejected_before_mutation")
  assert not calls and len(place.leases)==1and b.lease_id in place.leases
  res=await client.post("/control/native-groups/local-167/fault",json=dict(epoch="CPU-epoch-D1",reason="CPU recorded identity loss"));assert res.status_code==200and res.json()["changed"]is True
  assert place.execution_groups["local-167"].faulted and not place.execution_groups["local-166"].faulted and b.lease_id in place.leases
  contracts.append("epoch_specific_quarantine_retains_existing_lease_and_healthy_group")
  durable=json.loads((state/"fault.json").read_text());assert durable["open"]and durable["groups"]["local-167"]["faulted"]and not durable["groups"]["local-166"]["faulted"]
  contracts.append("quarantine_persisted_before_ACK")
  res=await client.post("/control/native-groups/local-167/fault",json=dict(epoch="CPU-epoch-D1",reason="CPU repeated loss"));assert res.status_code==200and res.json()["changed"]is False
  contracts.append("same_epoch_fault_idempotent")
  n=len(calls);res=await client.get("/v1/responses/resp_CPU_D1");assert res.status_code==503and len(calls)==n
  contracts.append("faulted_bound_owner_rejected_before_upstream_no_cross_owner_replay")
  res=await client.get("/v1/responses/resp_CPU_D0");assert res.status_code==200and res.json()==dict(fixture,id="resp_CPU_D0")and calls[-1]["url"].startswith("http://cpu-d0/")
  contracts.append("independent_bound_owner_survives_unchanged_native_mock_bytes")
  x=await place.acquire_for_owner(64,1,None);assert x.replica.key=="D0";await place.release(x)
  contracts.append("unbound_admission_excludes_faulted_epoch")
  await place.release(b);await place.remove("D1");await place.add(replicas[1])
  try:await place.acquire_for_owner(32,21,owner1)
  except RuntimeError:pass
  else:raise AssertionError("logicalreadd clearednativefault")
  contracts.append("logical_readd_does_not_clear_native_fault")
 await close(app);assert not json.loads((state/"fault.json").read_text())["open"]
 app=await app_for(groups,state);place=app.state.placement;idx=app.state.response_owner_index
 assert idx.resolve("resp_CPU_D0")==owner0 and idx.resolve("resp_CPU_D1")==owner1
 assert place.execution_groups["local-167"].faulted and not place.execution_groups["local-166"].faulted
 contracts.append("healthy_shutdown_restart_preserves_fault_and_both_owner_pointers")
 await close(app)
 new=[dict(x)for x in groups];new[1]=dict(new[1],epoch="CPU-new-D1");app=await app_for(new,state);place=app.state.placement;idx=app.state.response_owner_index
 async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://CPU")as client:
  res=await client.post("/control/native-groups/local-167/fault",json=dict(epoch="CPU-epoch-D1",reason="CPU stale watcher"));assert res.status_code==409and not place.execution_groups["local-167"].faulted
  contracts.append("stale_observation_cannot_quarantine_replacement_epoch")
  n=len(calls);res=await client.get("/v1/responses/resp_CPU_D1");assert res.status_code==503and len(calls)==n
  res=await client.get("/v1/responses/resp_CPU_D0");assert res.status_code==200
  contracts.append("replacement_epoch_retains_D0_and_never_replays_old_D1_STORE_pointer")
 await close(app)
 bad= j/"journal_failure_state";bad.mkdir();app=await app_for(groups,bad);place=app.state.placement
 def io_failure(opened):raise OSError("CPU injected journal I/O failure")
 place._persist=io_failure
 async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://CPU")as client:
  res=await client.post("/control/native-groups/local-167/fault",json=dict(epoch="CPU-epoch-D1",reason="CPU loss"));assert res.status_code==503and place.execution_groups["local-167"].faulted and place.journal_error
  contracts.append("journal_failure_keeps_in_memory_quarantine_no_success_ACK")
 await close(app)
 assert json.loads((bad/"fault.json").read_text())["open"]
 app=await app_for(groups,bad);assert all(x["group_faulted"]for x in await app.state.placement.snapshot())
 contracts.append("unconfirmed_durability_restart_is_conservative")
 await close(app)
asyncio.run(main())
print(json.dumps(dict(event="native_fault_CPU_valid",contracts=contracts,upstream_calls=calls,native_requests=0,models=0,NPU_workers_started=0,signals=0,fixture=True,limits=["CPU ASGI/mocknative only; physicalprocessloss/SSHobserver and real nativeSTORE E2E still require evidence","Native response fixture copied139 solely forsynthetictransportshape; no newnative inference credit orqualityclaim"])))
