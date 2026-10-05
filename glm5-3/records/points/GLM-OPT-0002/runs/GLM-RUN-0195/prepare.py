from common import *
from native_engines_service_config import compile_config
old=p/"runs/GLM-RUN-0194";s=json.loads((old/"state.json").read_text());assert s["status"]=="failed"and s["failure_phase"]=="deploy"and not same_process(s["owner"])
assert all(json.loads((old/(n+".phase.json")).read_text())["status"]=="succeeded"for n in["prepare","retire"])
assert json.loads((old/"retire190_public.json").read_text())["SDKinit_finalize0"]and (p/"runs/GLM-RUN-0190/restored/identity_observer/terminal.json").exists()
roots=json.loads((r/"standalone_root_identities.json").read_text());members=json.loads((r/"standalone_native_members.json").read_text());plans=json.loads((r/"standalone_launch.json").read_text())
for k,o in roots.items():fresh(o,members[k],"adopt_"+k)
raw=native_idle("166",9081,"adopt");assert all(float(v)==0for v in re.findall(r"^vllm:(?:generation_tokens_total|request_success_total|num_preemptions_total)(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",raw.decode(),re.M))
for k,x in plans.items():
 node=x["host"];text=run(node,["cat",x["log"]],"native_resident_"+k).decode(errors="replace");assert "Graph capturing finished"in text
 guards=[json.loads(l.split("GLM_PP_EMPTY_TOKEN_GUARD_INSTALLED ",1)[1])for l in text.splitlines()if"GLM_PP_EMPTY_TOKEN_GUARD_INSTALLED "in l];assert len(guards)==16and {x["fixed_source_sha256"]for x in guards}=={"3e8cccfae16feac0f8dfbc894ae922193ffd94052734d61ce478440f8b6c4fcb"}
 pol=json.loads(run(node,["cat","/data/tiankuan/wio/glm52-pd/deploy/plugins/coupled_pp194/issue_budget_policy.json"],"adopt_policy_"+node));assert pol==dict(schema_version=1,cohort_id="GLM-COHORT-0194",budget_tokens=8192,prefill_threshold_tokens=1024,prefill_cadence=1,serial=1)
 for f in(r/"plugin_src").iterdir():
  actual=run(node,["sha256sum","/data/tiankuan/wio/glm52-pd/deploy/plugins/coupled_pp194/"+f.name],"adopt_source_"+node+"_"+f.name).decode().split()[0];assert actual==ref(f)["sha256"]
folder=r/"restored"
for n in["standalone_launch.json","standalone_root_identities.json","standalone_native_members.json"]:atomic_json(folder/n,json.loads((r/n).read_text()))
for n in["PP_empty_guard_workers.json","local_cadence_installed.json"]:(folder/n).write_bytes((old/"restored"/n).read_bytes())
material=dict(placement=dict(kind="work_seconds"),engines=[dict(id="joint-194",replica_id="D0",plans=plans,roots=roots,members=members)])
negative=dict(material,placement="work_seconds")
try:compile_config(negative,state)
except ValueError as e:assert "explicit native API placement is required"in str(e)
else:raise AssertionError("malformedplacement accepted")
atomic_json(folder/"native_engines_resident.json",material);config=render(folder,state);atomic_json(folder/"service_config.json",config)
checked_config(folder/"service_config.json")
atomic_json(folder/"adopted_model_identities.json",dict(D0=dict(roots["node0"],rank=0,native_dp_rank=0,native_dp_size=1,logical_alias="D0",port=9081)))
atomic_json(folder/"deployment_summary.json",dict(at=utc(),ready=True,native_domains=config["native_domains"],NPU_workers=32,frontend_count=1,headless_count=1,PDhelpers=0,STORE_replication=False,request_replay=False,native_operator_math_changes=0,reused194_nativefit=True,replayed_model_operations=0,placement_CPU_negative_guard=True,capacity_claim=False))
atomic_json(r/"prepared_state.json",dict(at=utc(),native32_owned_idle=True,new_native_inference=0,oldpublicSDKfinal0_reused194=True,replayed_model_operations=0,malformed_placement_rejected=True))
