from common import *
shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source "+str(site/"scripts/pd_common_env.sh")+"; export PYTHONPATH="+str(g/"runtime")+":"+str(r)+":$PYTHONPATH; python3 "+str(plug/"native_acl_lifecycle.py")+" script "+str(r/"pilot.py")
raw=run("166",["docker","exec","glm52-single","bash","-c",shell],"D1_pilot",timeout=900)
acks=[json.loads(l)for l in raw.decode().splitlines()if l.startswith('{"event":')];assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
folder=r/"restored";rows=json.loads((r/"pilot_summary.json").read_text())["requests"];assert len(rows)==2and sum(x["usage"]["completion_tokens"]for x in rows)==96
short=next(x for x in rows if x["name"]=="short_D1");full=next(x for x in rows if x["name"]=="full_D1")
config,_=checked_config(folder/"service_config.json");epoch=next(x["epoch"]for x in config["native_domains"]if x["id"]=="local-167")
hint=dict(schema_version=1,native_owner_epoch=epoch,cohort_id="GLM-COHORT-0137",decode_tps=31/(short["wall_s"]-short["ttft_s"]),prefill_bytes_per_s=(r/"full_D1.body.json").stat().st_size/full["ttft_s"],method="Observed standalone short32 residual31/(HTTPwall-TTFT), cold81932 serializedHTTPbodybytes/TTFT",sources=[ref(r/"pilot_summary.json"),ref(r/"short_D1.body.json"),ref(r/"short_D1.wire"),ref(r/"full_D1.body.json"),ref(r/"full_D1.wire")],limitations=["Two finite single-request diagnostics, not GPU time or a batch/progress/background predictor","D0 prior138 hint reused for unchanged physical epoch; D1 replaces retired137 epoch","No isolated performance gain or capacity acceptance"])
atomic_json(folder/"observed_D1.json",hint)
material=json.loads((folder/"native_engines_resident.json").read_text());next(x for x in material["engines"]if x["replica_id"]=="D1")["routing_hint"]=ref(folder/"observed_D1.json");atomic_json(folder/"native_engines_resident.json",material);atomic_json(folder/"service_config.json",render(folder,state))
atomic_json(r/"pilot_SDK_summary.json",dict(at=utc(),SDKinit_finalize0=True,new_D1_epoch=epoch,output_tokens=96,calibration=ref(folder/"observed_D1.json")))
