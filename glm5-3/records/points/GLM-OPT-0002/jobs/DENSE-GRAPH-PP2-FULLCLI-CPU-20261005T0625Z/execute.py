from pathlib import Path
import json,sys,hashlib,subprocess,shlex,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;b=j.parents[1];old=b/"runs/GLM-RUN-0202";roots=json.loads((old/"restored/standalone_root_identities.json").read_text());members=json.loads((old/"restored/standalone_native_members.json").read_text())
audit=json.loads((b/"jobs/AUDIT-RUN203-20261005TPOSTMIXED/reduction.json").read_text());assert audit["measurement_valid"]and audit["verdict"]=="REJECT"
controller=json.loads((b/"runs/GLM-RUN-0203/state.json").read_text());assert controller["status"]=="completed"and not same_process(controller["owner"])
def ref(p):
 raw=Path(p).read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def call(node,args,label,input=None,timeout=150):
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=input,capture_output=True,timeout=timeout);(j/(label+".stdout")).write_bytes(z.stdout);(j/(label+".stderr")).write_bytes(z.stderr);z.check_returncode();return z.stdout
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def probe(label):
 values={}
 for key,o in roots.items():
  raw=call(o["host"],["python3","-c",(old/"live_probe.py").read_text()],label+"_"+key,input=json.dumps(dict(owner=o,NPU_count=16)).encode());v=json.loads(raw)
  expected={x["pid"]:x["identity"]for x in members[key]["owned_targets"]}
  assert v["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(expected[x["pid"]]==x["identity"]for x in v["owned_targets"]if x["pid"]in v["npu_worker_pids"])
  port=9081 if o["host"]=="166"else 9900
  with http.open("http://172.16.10."+o["host"]+":"+str(port)+"/metrics",timeout=15)as res:values[key]=res.read()
  (j/(label+"_"+key+".metrics")).write_bytes(values[key])
 return values
before=probe("before")
files={p.name:p.read_text()for p in(j/"plugin_src").iterdir()};private=Path("/data/tiankuan/wio/glm52-pd/deploy/plugins/local_pp204")
install="from pathlib import Path;import sys,json,hashlib,os;a=json.load(sys.stdin);p=Path(a['path']);assert not p.exists();p.mkdir();rows=[]\nfor n,s in a['files'].items():\n f=p/n;f.write_text(s);os.chmod(f,0o600 if n.endswith('.json')else 0o644);b=f.read_bytes();rows.append(dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))\nprint(json.dumps(rows))"
raw=call("166",["python3","-c",install],"install",input=json.dumps(dict(path=str(private),files=files)).encode());atomic_json(j/"installed_sources.json",json.loads(raw))
x=json.loads((j/"candidate_plan.json").read_text())
shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; export PYTHONPATH="+str(private)+":$PYTHONPATH VLLM_HOST_IP=172.16.10.166 "+" ".join(k+"="+shlex.quote(v)for k,v in x["environment"].items())+"; python3 "+str(private/"native_acl_lifecycle.py")+" script "+str(private/"api_config_probe.py")+" "+shlex.quote(json.dumps(x["argv"]))
raw=call("166",["docker","exec","glm52-single","bash","-c",shell],"fullCLI_CPU",timeout=300)
events=[json.loads(l)for l in raw.decode().splitlines()if l.startswith('{"event":')]
assert events[0]["event"]=="task_acl_init"and events[-1]["event"]=="task_acl_finalize"and events[0]["returncode"]==events[-1]["returncode"]==0
native=next(x for x in events if x["event"]=="fullCLI_PP32_exactplan")
assert native["TP"]==native["DCP"]==8and native["PP"]==2and native["world"]==native["local_world"]==16and native["nnodes"]==1and native["workers"]==native["models"]==native["inference"]==native["NPU_tensors"]==0
dense=next(x for x in events if x["event"]=="native_dense_Graph_CPU");assert dense["full_native_config"]and len(dense["native_dispatch"])==2
after=probe("after")
def counters(raw):
 out={}
 for line in raw.decode().splitlines():
  if not line or line.startswith("#"):continue
  k=line.split("{",1)[0].split()[0]
  if k.startswith("vllm:")and (k.endswith("_total")or k in["vllm:num_requests_running","vllm:num_requests_waiting","vllm:kv_cache_usage_perc"]):out[k]=out.get(k,0)+float(line.rsplit(" ",1)[1])
 return out
assert all(counters(before[k])==counters(after[k])for k in before)
out=dict(at=utc(),CPU_config_VALID=True,exact_fullCLI=native,native_dense_Graph_CPU=dense,SDKinit_finalize0=True,current_native32_same_idle_counters=True,new_workers=0,new_models=0,new_inference=0,new_NPU_tensors=0,candidate_plan=ref(j/"candidate_plan.json"),private_sources=ref(j/"installed_sources.json"),Current=None,limits=["CPU fullCLI/legal denseGraph config and complete-native-config CPU dispatcher only; actual native capture fit/currentGPU batch padding/fullE2E/performance unknown",
"Native operators/MTP-K3/TP8PP2DCP8/Graphmax32/maxseq8/guard/8192t1024c1 preserved; extra nativecapturekeys may increasememory andcapturecost",
"Reuse118120 samePP2K3-vsK5 no reliableoverallbenefit and heavy105graph-union conditional source; noK5rerun/heavyprof/globalbound/KEEP",
"Current202D0+200D1/native32/STORE epochs/source/policy unchanged; CPU no native model lifecycle/inference and processCPUqueryexcludedfromnativevllmcounters"])
atomic_json(j/"reduction.json",out)
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Native D0 PP2TP8DCP8 K3/denseGraph fullCLI-CPUdispatcher VALID/SDK0/native32same/noGPUmodels-inference; actualfit-E2E unknown",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="fullCLI/SDK/nativebothfreshNPU16/countersunchanged")],unknowns=out["limits"],decision_request=None,next_check_at=None))
print(json.dumps(dict(valid=True,reduction=ref(j/"reduction.json"))))
