from common import *
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import threading
folder=r/"restored";roots=json.loads((folder/"standalone_root_identities.json").read_text());members=json.loads((folder/"standalone_native_members.json").read_text())
for key,o in roots.items():fresh(o,members[key],"pilot_before_"+key)
metrics=["generation_tokens_total","prompt_tokens_total","prefix_cache_hits_total","external_prefix_cache_hits_total","num_preemptions_total","request_success_total"]
def counters(b):
 return {m:sum(float(x)for x in re.findall(r"^vllm:"+m+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",b.decode(),re.M))for m in metrics}
before={node:counters(native_idle(node,9900 if node=="167"else 9081,"pilot_before"))for node in["166"]}
stop=threading.Event();samples=[];errors=[]
def observe():
 try:
  while not stop.is_set():
   guard()
   with http.open("http://172.16.10.166:9081/metrics",timeout=10)as response:b=response.read()
   f=r/("pilot_observe_%04d.metrics"%len(samples));f.write_bytes(b);samples.append(dict(at=utc(),metrics=ref(f)))
   atomic_json(r/"pilot_observations.json",samples);stop.wait(1)
 except BaseException as e:errors.append(type(e).__name__+":"+str(e))
thread=threading.Thread(target=observe);thread.start()
def one(i):
 script=r/("pilot_"+str(i))/"pilot.py"
 shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source "+str(site/"scripts/pd_common_env.sh")+"; export PYTHONPATH="+str(g/"runtime")+":"+str(r)+":$PYTHONPATH; python3 "+str(plug/"native_acl_lifecycle.py")+" script "+str(script)
 raw=run("166",["docker","exec","glm52-single","bash","-c",shell],"D0_pilot_"+str(i),timeout=900)
 events=[json.loads(l)for l in raw.decode().splitlines()if l.startswith('{"event":')]
 assert events[0]["event"]=="task_acl_init"and events[-1]["event"]=="task_acl_finalize"and events[0]["returncode"]==events[-1]["returncode"]==0
 v=json.loads((script.parent/"pilot_summary.json").read_text());assert v["measurement_valid"]and v["functional_acceptance"]and v["effective_public_output_tokens"]==96
 return v
try:
 with ThreadPoolExecutor(max_workers=2)as pool:results=list(pool.map(one,[0,1]))
finally:stop.set();thread.join(timeout=15)
assert not thread.is_alive()and not errors,errors
after={node:counters(native_idle(node,9900 if node=="167"else 9081,"pilot_after"))for node in["166"]}
delta={node:{m:after[node][m]-before[node][m]for m in metrics}for node in before}
assert set(delta)=={"166"}
d=delta["166"];assert d["generation_tokens_total"]==192and d["prompt_tokens_total"]==163906and d["request_success_total"]==4and d["prefix_cache_hits_total"]==d["external_prefix_cache_hits_total"]==d["num_preemptions_total"]==0,delta
rows=[x for v in results for x in v["requests"]];assert len(rows)==4and all(x["completed"]and x["contract"].get("done")for x in rows)
short=[x for x in rows if x["name"]=="short_D0"];full=[x for x in rows if x["name"]=="full_D0"]
times=[(datetime.fromisoformat(x["attempted_at"]).timestamp(),x["wall_s"])for x in full]
overlap=min(a+b for a,b in times)-max(a for a,b in times);assert overlap>0
for key,o in roots.items():fresh(o,members[key],"pilot_after_"+key)
config,_=checked_config(folder/"service_config.json");epoch=next(x["epoch"]for x in config["native_domains"]if x["id"]=="joint-193")
sources=[]
for i in[0,1]:
 for name in["pilot_summary.json","short_D0.body.json","short_D0.wire","full_D0.body.json","full_D0.wire"]:sources.append(ref(r/("pilot_"+str(i))/name))
hint=dict(schema_version=1,native_owner_epoch=epoch,cohort_id="GLM-COHORT-0193",decode_tps=sum(31/(x["wall_s"]-x["ttft_s"])for x in short)/2,prefill_bytes_per_s=sum((r/("pilot_"+str(i))/"full_D0.body.json").stat().st_size for i in[0,1])/sum(x["ttft_s"]for x in full),method="Observed paired standalone short32 residual31/(HTTPwall-TTFT) mean; paired cold81932 serializedHTTPbodybytes/sumTTFT",sources=sources,limitations=["Finite batch2 HTTP diagnostics; not GPU time, batch/progress predictor, repeated causal gain, stable capacity or KEEP","One joint32 engine singleHTTP frontend; not two independent replicas; changedresource category"])
assert hint["decode_tps"]>0and hint["prefill_bytes_per_s"]>0
atomic_json(folder/"observed_D0.json",hint)
material=json.loads((folder/"native_engines_resident.json").read_text());next(x for x in material["engines"]if x["replica_id"]=="D0")["routing_hint"]=ref(folder/"observed_D0.json");atomic_json(folder/"native_engines_resident.json",material);atomic_json(folder/"service_config.json",render(folder,state))
atomic_json(r/"pilot_SDK_summary.json",dict(at=utc(),measurement_valid=True,SDKinit_finalize0=True,actual_clients=2,new_D0_epoch=epoch,output_tokens=192,native_delta=delta,cold_HTTP_overlap_s=overlap,requests=rows,calibration=ref(folder/"observed_D0.json"),observations=ref(r/"pilot_observations.json"),actual_native_batch8192_claim=False))
