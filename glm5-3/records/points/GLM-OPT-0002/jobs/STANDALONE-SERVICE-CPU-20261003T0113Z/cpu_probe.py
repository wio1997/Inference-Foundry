import asyncio,copy,hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from standalone_service_config import compile_config,checked_config
from response_affinity_gateway_v11 import create_app
j=Path(sys.argv[1]);fixture=j/"resident_fixture";fixture.mkdir()
argv=json.loads((j/"native_CPU_argv.json").read_text())
plans={};roots={};members={}
for rank,host in enumerate(["166","167"]):
 key="node"+str(rank);a=argv[:];a[a.index("--node-rank")+1]=str(rank);a[a.index("--port")+1]="9081"
 if rank:a.append("--headless")
 plans[key]=dict(host=host,node_rank=rank,api=rank==0,argv=a,environment=dict(VLLM_PP_LAYER_PARTITION="42,36",fixture=True))
 root=dict(host=host,pid=1000+rank,argv=a,role="headless"if rank else"API",identity=dict(boot_id="CPU-fixture-"+host,start_ticks="100"))
 roots[key]=root
 targets=[dict(pid=2000+i,identity=dict(boot_id=root["identity"]["boot_id"],start_ticks=str(200+i)))for i in range(16)]
 members[key]=dict(root=root,root_alive=True,npu_worker_pids=[x["pid"]for x in targets],owned_targets=targets)
for name,a in[("standalone_launch.json",plans),("standalone_root_identities.json",roots),("standalone_native_members.json",members)]:
 (fixture/name).write_text(json.dumps(a,indent=2)+"\n")
state=j/"fixture_state";config=compile_config(plans,roots,members,state)
assert config["geometry"]["world_size"]==32and config["geometry"]["local_world_size"]==16
assert json.loads(config["environment"]["GLM_PD_PRODUCERS"])=={}
output=j/"rendered_config.json"
z=subprocess.run([sys.executable,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime/standalone_service_config.py","--resident-dir",str(fixture),"--state-dir",str(state),"--output",str(output)],capture_output=True);(j/"render.stdout").write_bytes(z.stdout);(j/"render.stderr").write_bytes(z.stderr);z.check_returncode()
assert checked_config(output)[0]["native_domains"]==config["native_domains"]
contracts=["actual_render_and_checked_provenance"]
for kind in["missing_headless","duplicate_NPU","root_argv_mismatch","headless_worker_missing","different_engine_arguments","KV_transfer_forbidden","wrong_host_boot"]:
 p,r,m=map(copy.deepcopy,[plans,roots,members])
 if kind=="missing_headless":
  del p["node1"];del r["node1"];del m["node1"]
 elif kind=="duplicate_NPU":m["node1"]["npu_worker_pids"][-1]=m["node1"]["npu_worker_pids"][0]
 elif kind=="root_argv_mismatch":r["node1"]["argv"].append("--invalid-fixture")
 elif kind=="headless_worker_missing":m["node1"]["owned_targets"].pop()
 elif kind=="different_engine_arguments":
  a=p["node1"]["argv"];a[a.index("--seed")+1]="1025";r["node1"]["argv"]=a;m["node1"]["root"]=r["node1"]
 elif kind=="KV_transfer_forbidden":
  for key in p:p[key]["argv"]+=["--kv-transfer-config","{}"];r[key]["argv"]=p[key]["argv"];m[key]["root"]=r[key]
 elif kind=="wrong_host_boot":m["node1"]["owned_targets"][0]["identity"]["boot_id"]="CPU-wrong-host"
 try:compile_config(p,r,m,state)
 except ValueError:contracts.append(kind+"_rejected")
 else:raise AssertionError(kind+"accepted")
changed=copy.deepcopy(members);changed["node1"]["owned_targets"][0]["identity"]["start_ticks"]="999"
new=compile_config(plans,roots,changed,state);assert new["native_domains"][0]["epoch"]!=config["native_domains"][0]["epoch"]
contracts.append("headless_NPU_identity_retires_whole_engine_epoch")
def app_for(c):
 os.environ.pop("GLM_GROUP_FAULT_STATE_PATH",None);os.environ.update(c["environment"]);return create_app()
async def check():
 app=app_for(config);place=app.state.placement;idx=app.state.response_owner_index
 try:
  lease=await place.acquire_for_owner(64,81932,None);assert lease.replica.key=="D0"
  idx.bind("resp_standalone_CPU",lease);binding=idx.resolve("resp_standalone_CPU")
  await place.release(lease,backend_failure=False)
 finally:idx.close();await place.close()
 app=app_for(config);place=app.state.placement;idx=app.state.response_owner_index
 try:
  assert idx.resolve("resp_standalone_CPU")==binding
  lease=await place.acquire_for_owner(64,5,binding);assert lease.replica.key=="D0"
  await place.release(lease,backend_failure=False)
 finally:idx.close();await place.close()
 contracts.append("same_coupled_epoch_gateway_restart_preserves_owner")
 app=app_for(new);place=app.state.placement;idx=app.state.response_owner_index
 try:
  try:await place.acquire_for_owner(64,5,idx.resolve("resp_standalone_CPU"))
  except RuntimeError as e:assert str(e)=="native response owner unavailable; no cross-owner replay"
  else:raise AssertionError("headless worker replacement did not retire bound response")
  assert not place.leases
 finally:idx.close();await place.close()
 contracts.append("changed_headless_epoch_rejects_bound_response_without_lease")
asyncio.run(check())
bad=json.loads(output.read_text());bad["environment"]["GLM_REPLICAS"]="[]";path=j/"tampered_config.json";path.write_text(json.dumps(bad))
try:checked_config(path)
except ValueError:contracts.append("tampered_environment_rejected")
else:raise AssertionError("tampered environment accepted")
bad=json.loads(output.read_text());bad["resident_evidence"]["standalone_launch.json"]["sha256"]="0"*64;path.write_text(json.dumps(bad))
try:checked_config(path)
except ValueError:contracts.append("tampered_provenance_rejected")
else:raise AssertionError("tampered provenance accepted")
z=subprocess.run([sys.executable,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime/standalone_service_entry.py","--help"],capture_output=True);z.check_returncode();(j/"entry_help.stdout").write_bytes(z.stdout);assert b"--config"in z.stdout;contracts.append("actual_standalone_entry_CLI_help")
print(json.dumps(dict(event="standalone_service_CPU_valid",contracts=contracts,fixture=True,models=0,native_requests=0,NPU_workers_started=0,render_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),limits=["Synthetic process ancestry CPU fixture, not live32NPU engine/physical restart/STORE replication","Actual V11 factory and standalone CLI used; no listener/nativeHTTP/model operations"])))
