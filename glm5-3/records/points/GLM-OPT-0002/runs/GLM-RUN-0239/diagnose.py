from common import *
folder=r/"restored";plans=json.loads((folder/"standalone_launch.json").read_text());roots=json.loads((folder/"standalone_root_identities.json").read_text());members=json.loads((folder/"standalone_native_members.json").read_text())
shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source "+str(site/"scripts/pd_common_env.sh")+"; export PYTHONPATH="+str(bundle)+":"+str(r)+":$PYTHONPATH; python3 "+str(plug/"native_acl_lifecycle.py")+" script "+str(r/"client.py")
raw=run("166",["docker","exec","glm52-single","bash","-c",shell],"diagnostic_client",timeout=240)
events=[json.loads(l)for l in raw.decode().splitlines()if l.startswith('{"event":')]
assert events[0]["event"]=="task_acl_init"and events[-1]["event"]=="task_acl_finalize"and all(x["returncode"]==0for x in events)
summary=json.loads((r/"diagnostic_client_summary.json").read_text());assert summary["HTTP_status"]==200and summary["semantic_pass"]and summary["effective_output_tokens"]==2
fresh(roots["node1"],members["node1"],"terminal_D1");native_idle("167",9900,"terminal")
def totals(f):
 out={}
 for line in f.read_text().splitlines():
  if not line or line.startswith("#"):continue
  k=line.split("{")[0].split()[0]
  if k.startswith("vllm:")and k.endswith("_total"):out[k]=out.get(k,0)+float(line.rsplit(" ",1)[1])
 return out
assert totals(r/"before_167.metrics")==totals(r/"terminal_167.metrics")
raw=run("166",["cat",plans["node0"]["log"]],"diagnostic_native")
text=raw.decode(errors="replace");selected=[l for l in text.splitlines()if any(t in l for t in[" File ","RuntimeError:","Index out of range","Dumping scheduler output"])]
(r/"diagnostic_error_excerpt.txt").write_text("\n".join(selected)+"\n")
inactive={x["pid"]:not same_process(dict(pid=x["pid"],**x["identity"]))for x in members["node0"]["owned_targets"]}
npuraw=run("166",["npu-smi","info"],"terminal166_NPU").decode();npus=re.findall(r"^\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|\s+[^|]+\|\s+\d+\s+\|",npuraw,re.M)
out=dict(at=utc(),run_id=r.name,diagnostic_completed=True,functional_acceptance=summary["semantic_pass"],verdict="INCONCLUSIVE"if summary["semantic_pass"]else"INVALID",diagnostic=summary,SDKclient_init_finalize0=True,D1_all_totals_unchanged=True,D1_native16=True,D0_inactive_targets=inactive,D0_NPU_processes=len(npus),native_source=ref(r/"diagnostic_native.stdout"),error_excerpt=ref(r/"diagnostic_error_excerpt.txt"),blocking_debug=False,draft_decode_dense_diagnostic=True,metadata_observation_only=True,performance_claim=False,Current=None,limits=["One blocking diagnostic request; no performance/fullAPI equivalence/KEEP","InstalledV2 knownthinking_token_budget gap open; exactoriginatingindexcall readfrom rawstack"])
atomic_json(r/"diagnostic_summary.json",out);print(json.dumps(dict(diagnostic_completed=True,HTTP_status=summary["HTTP_status"],D0_NPU=len(npus),D1retained=True)))

fresh(roots["node0"],members["node0"],"terminal_D0");native_idle("166",9081,"terminal")
assert not(r/"diagnostic_enable.json").exists()
selected=[json.loads(l.split("GLM_BATCH_QUEUE_CAP_SELECTED ",1)[1])for l in text.splitlines()if"GLM_BATCH_QUEUE_CAP_SELECTED "in l]
assert len(selected)==1and selected[0]["cap"]==2and selected[0]["serial"]==1and selected[0]["native_resource_capacity"]==3and selected[0]["transition_only_empty"]
placement=request("/control/replicas");rows={x["id"]:x for x in placement["replicas"]}
assert rows["D1"]["compatibility_features"]==["structured_output","thinking_budget"]and rows["D0"]["compatibility_features"]==[]and all(not x["group_faulted"]and not x["active_requests"]and not x["draining"]for x in rows.values())
for id in["resp_glm_run230_D0_new","resp_glm_run211_D1_new"]:
 try:
  with http.open("http://127.0.0.1:8000/v1/responses/"+id,timeout=15)as response:code=response.status;raw=response.read()
 except urllib.error.HTTPError as e:code=e.code;raw=e.read()
 (r/(id+".wire")).write_bytes(raw)
 assert code==(503 if "D0"in id else 200)
 if"D1"in id:assert json.loads(raw)==json.loads((r/"before_D1_native_STORE.wire").read_text())
atomic_json(r/"recovery_summary.json",dict(at=utc(),plain_native_new_D0_function=True,outputs=2,SDKinit_finalize0=True,native32_owned_idle=True,D0_new_epoch=True,D1_epoch_STORE_counters_retained=True,old_D0_STORE503_no_replay=True,admission_cap2_native_resource3=True,CPU_GPU_diagnostic_off=True,public_version="V13",compatibility_features_declared_only=True,compatibility_E2E_pending=True,Current=None))
