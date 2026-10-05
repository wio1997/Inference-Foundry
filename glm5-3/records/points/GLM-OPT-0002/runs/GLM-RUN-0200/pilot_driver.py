
from common import *
from concurrent.futures import ThreadPoolExecutor
folder=r/"restored";roots=json.loads((folder/"standalone_root_identities.json").read_text());members=json.loads((folder/"standalone_native_members.json").read_text())
for k,o in roots.items():fresh(o,members[k],"pilot_before_"+k)
metrics=["generation_tokens_total","prompt_tokens_total","prefix_cache_hits_total","external_prefix_cache_hits_total","num_preemptions_total","request_success_total"]
def counters(b):return{m:sum(float(x)for x in re.findall(r"^vllm:"+m+r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",b.decode(),re.M))for m in metrics}
before={node:counters(native_idle(node,9081 if node=="166"else 9900,"pilot_before"))for node in["166","167"]}
def one(i):
 script=r/("pilot_"+str(i))/"pilot.py"
 shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source "+str(site/"scripts/pd_common_env.sh")+"; export PYTHONPATH="+str(g/"runtime")+":"+str(r)+":$PYTHONPATH; python3 "+str(plug/"native_acl_lifecycle.py")+" script "+str(script)
 raw=run("166",["docker","exec","glm52-single","bash","-c",shell],"pilot_"+str(i),timeout=900)
 acks=[json.loads(l)for l in raw.decode().splitlines()if l.startswith('{"event":')]
 assert [x["event"]for x in acks]==["task_acl_init","task_acl_finalize"]and all(x["returncode"]==0for x in acks)
 v=json.loads((script.parent/"pilot_summary.json").read_text());assert v["measurement_valid"]and v["functional_acceptance"]and v["effective_public_output_tokens"]==96;return v
with ThreadPoolExecutor(max_workers=2)as pool:results=list(pool.map(one,[0,1]))
after={node:counters(native_idle(node,9081 if node=="166"else 9900,"pilot_after"))for node in["166","167"]}
delta={node:{m:after[node][m]-before[node][m]for m in metrics}for node in before}
for node,d in delta.items():assert d["generation_tokens_total"]==96and d["prompt_tokens_total"]==81953and d["request_success_total"]==2and d["prefix_cache_hits_total"]==d["external_prefix_cache_hits_total"]==d["num_preemptions_total"]==0,delta
for k,o in roots.items():fresh(o,members[k],"pilot_after_"+k)
atomic_json(r/"pilot_SDK_summary.json",dict(at=utc(),measurement_valid=True,SDKinit_finalize0=True,actual_clients=2,output_tokens=192,native_delta=delta,requests=[x for v in results for x in v["requests"]],capacity_claim=False))
