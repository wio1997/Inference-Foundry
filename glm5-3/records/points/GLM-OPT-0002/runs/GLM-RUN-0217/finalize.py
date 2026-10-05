from common import *
folder=r/"restored";roots=json.loads((folder/"standalone_root_identities.json").read_text());members=json.loads((folder/"standalone_native_members.json").read_text())
for key,o in roots.items():fresh(o,members[key],"terminal_"+key);native_idle(o["host"],9081 if o["host"]=="166"else 9900,"terminal")
for node,path,cohort,serial in[("166","local_pp216","0216",1),("167","local_pp211","0211",1)]:
 expected=dict(schema_version=1,cohort_id="GLM-COHORT-"+cohort,budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=serial)
 assert json.loads(run(node,["cat",str(site/("plugins/"+path+"/issue_budget_policy.json"))],"terminal_policy_"+node))==expected
plans=json.loads((folder/"standalone_launch.json").read_text());raw=run("166",["cat",plans["node0"]["log"]],"terminal_D0_native")
events=[json.loads(l.split("GLM_ISSUE_BUDGET_SELECTED ",1)[1])for l in raw.decode(errors="replace").splitlines()if"GLM_ISSUE_BUDGET_SELECTED "in l]
assert events and all(x["cohort"]=="GLM-COHORT-0216"and x["budget_tokens"]==8192and x["prefill_threshold_tokens"]==1024and x["prefill_cadence"]==1and x["serial"]==1and not x["fallback"]and x["native_max"]==8192for x in events)
atomic_json(r/"policy_consumption.json",dict(at=utc(),native_D0_selected=events,policy_files_verified=True,D1_unchanged=True,operator_math_changes=0))
clients={m:json.loads((r/("client_"+m+"_summary.json")).read_text())for m in["final"]}
assert clients["final"]["valid"]
final=clients["final"];assert final["effective_output_tokens"]==59and final["native_delta"]["D1"]["generation_tokens_total"]==0and final["native_delta"]["D0"]["generation_tokens_total"]==59
public=json.loads((folder/"public_service_proof.json").read_text());assert same_process(public["host"])and public["SDK_init0"]and request("/healthcheck")["request_num"]==0
assert all(not x["active_requests"]and not x["group_faulted"]and not x["draining"]for x in request("/control/replicas")["replicas"])
assert all(x["status"]=="healthy"for x in json.loads((folder/"identity_observer/latest.json").read_text())["groups"])
out=dict(at=utc(),valid=True,functional_acceptance=True,verdict="INCONCLUSIVE",new_native_completed=5,effective_output_tokens=59,D1_new_output_tokens=0,D0_new_output_tokens=59,clients={m:ref(r/("client_"+m+"_summary.json"))for m in clients},policy=ref(r/"policy_consumption.json"),public=ref(folder/"public_service_proof.json"),native32_healthy_idle=True,D1_epoch_unchanged=True,D0_new_epoch=True,SDK_clients_init_finalize0=True,public_SDK_init0_active=True,PDhelpers=0,STORE_replication=False,request_replay=False,prior_before_survivor_evidence=ref(p/"jobs/AUDIT-RUN216-PARTIAL-20261005T1010Z-v2/reduction.json"),models_reloaded=0,models_signalled=0,native_V2_K2_Graph_capture=ref(folder/"native_dense_capture_proof.json"),actual_runtime_batch_dispatch_unknown=True,Current=None,prototype_only=True,V2_thinking_token_budget_gap_open=True,full_native_request_equivalence=False,limits=["Functional boundedV2 K2sameGraph3n/nativefit only; thinking_token_budget featuregap open; noKEEP/stablecapacity/globalbound","OldD0STORE214503preleaseRPC/epoch sticky; D1STORE211retained byteexact/counters0; no STOREmigration/replay","NativeGraph8/8 keys3,6,9,12,15,18,21,24; maxseq8 K2 width3; jointK-Graph notisolatedKgain"])
atomic_json(r/"functional_summary.json",out)
print(json.dumps(out))

def totals(f):
 out={}
 for line in f.read_text().splitlines():
  if not line or line.startswith("#"):continue
  k=line.split("{")[0].split()[0]
  if k.startswith("vllm:")and k.endswith("_total"):out[k]=out.get(k,0)+float(line.rsplit(" ",1)[1])
 return out
assert totals(r/"before_167.metrics")==totals(r/"terminal_167.metrics")
