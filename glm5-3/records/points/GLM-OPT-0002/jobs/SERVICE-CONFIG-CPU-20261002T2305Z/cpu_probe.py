import asyncio,copy,json,os,subprocess,sys,tempfile,hashlib
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from service_config import compile_config
from response_affinity_gateway_v11 import create_app
j=Path(sys.argv[1]);resident=j.parents[1]/"runs/GLM-RUN-0121"
plans=json.loads((resident/"planned_launch.json").read_text())
owners=json.loads((resident/"adopted_model_identities.json").read_text())
members=json.loads((resident/"native_member_identities.json").read_text())
state=Path(tempfile.mkdtemp(prefix="service_fixture_",dir=str(j)))
config=compile_config(plans,owners,members,state)
output=j/"actual_cli_config.json"
z=subprocess.run([sys.executable,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime/service_config.py","--resident-dir",str(resident),"--state-dir",str(state),"--output",str(output)],capture_output=True)
(j/"render_cli.stdout").write_bytes(z.stdout);(j/"render_cli.stderr").write_bytes(z.stderr);z.check_returncode()
cli=json.loads(output.read_text());assert all(cli[k]==v for k,v in config.items())
for kind in["argv_mismatch","duplicate_worker"]:
 a=copy.deepcopy(owners);b=copy.deepcopy(members)
 if kind=="argv_mismatch":a["D1"]["argv"].append("--invalid-fixture");b["D1"]["API_owner"]=a["D1"]
 else:b["D1"]["npu_worker_pids"][-1]=b["D1"]["npu_worker_pids"][0]
 try:compile_config(plans,a,b,state)
 except ValueError:pass
 else:raise AssertionError(kind+"notrejected")
changed=copy.deepcopy(members);worker=changed["D1"]["npu_worker_pids"][0]
target=next(x for x in changed["D1"]["owned_targets"]if x["pid"]==worker)
target["identity"]["start_ticks"]=str(int(target["identity"]["start_ticks"])+1)
new=compile_config(plans,owners,changed,state)
assert config["native_domains"][0]==new["native_domains"][0]
assert config["native_domains"][1]["epoch"]!=new["native_domains"][1]["epoch"]
def app_for(cfg):
 os.environ.update(cfg["environment"]);return create_app()
async def verify():
 app=app_for(config);place=app.state.placement;idx=app.state.response_owner_index
 try:
  lease=await place.acquire_for_owner(32,9000,None);assert lease.replica.key=="D1"
  idx.bind("resp_service_native_fixture",lease);binding=idx.resolve("resp_service_native_fixture")
  await place.release(lease,backend_failure=False)
 finally:idx.close();await place.close()
 # A new gateway process over exactly the same native domains keeps ownership.
 app=app_for(config);place=app.state.placement;idx=app.state.response_owner_index
 try:
  assert idx.resolve("resp_service_native_fixture")==binding
  lease=await place.acquire_for_owner(32,10,binding)
  assert lease.replica.key=="D1"  # native owner overrides short-input P preference.
  await place.release(lease,backend_failure=False)
 finally:idx.close();await place.close()
 # A changed physical worker identity must invalidate the prior native binding.
 app=app_for(new);place=app.state.placement;idx=app.state.response_owner_index
 try:
  assert idx.resolve("resp_service_native_fixture")==binding
  try:await place.acquire_for_owner(32,10,binding)
  except RuntimeError as err:assert str(err)=="native response owner unavailable; no cross-owner replay"
  else:raise AssertionError("changed native domain silently replayed old response")
  assert not place.leases
 finally:idx.close();await place.close()
asyncio.run(verify())
print(json.dumps(dict(event="service_config_CPU_valid",contracts=["actual_render_CLI","mismatched_API_plan_rejected","duplicate_NPU_member_rejected","NPU_identity_changes_only_native_D_epoch","gateway_restart_persists_native_owner_over_shape_hint","physical_domain_change_rejects_old_owner_without_lease"],native_requests=0,models=0,NPU_workers_started=0,fixture_journal=str(state/"response_owners.json"),rendered_config_sha256=hashlib.sha256(output.read_bytes()).hexdigest())))

