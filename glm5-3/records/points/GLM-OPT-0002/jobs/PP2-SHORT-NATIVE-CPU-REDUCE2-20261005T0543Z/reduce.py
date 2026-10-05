from pathlib import Path
import json,hashlib,sys,subprocess,shlex
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;b=j.parents[1];old=b/"jobs/PP2-SHORT-NATIVE-FULLCONFIG-CPU-20261005T0540Z";resident=b/"runs/GLM-RUN-0200"
def ref(p):
 raw=Path(p).read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def counters(p):
 out={}
 for l in p.read_text().splitlines():
  if not l or l.startswith("#"):continue
  k=l.split("{",1)[0].split()[0]
  if k.endswith("_total")or k in["vllm:num_requests_running","vllm:num_requests_waiting","vllm:kv_cache_usage_perc"]:out[k]=out.get(k,0)+float(l.rsplit(" ",1)[1])
 return out
proofs={};foreign_deltas={}
roots=json.loads((resident/"restored/standalone_root_identities.json").read_text());members=json.loads((resident/"restored/standalone_native_members.json").read_text())
for key,o in roots.items():
 before=json.loads((old/("before_"+key+".stdout")).read_text());after=json.loads((old/("after_"+key+".stdout")).read_text())
 assert before["root"]==after["root"]==o and before["npu_worker_pids"]==after["npu_worker_pids"]==members[key]["npu_worker_pids"]
 a=counters(old/("before_"+key+".metrics"));z=counters(old/("after_"+key+".metrics"))
 nativeA={k:v for k,v in a.items()if k.startswith("vllm:")};nativeZ={k:v for k,v in z.items()if k.startswith("vllm:")}
 assert nativeA==nativeZ and all(nativeA["vllm:"+k]==0for k in["num_requests_running","num_requests_waiting","kv_cache_usage_perc"])
 diff={k:dict(before=a.get(k),after=z.get(k),delta=z.get(k,0)-a.get(k,0))for k in a.keys()|z.keys()if a.get(k)!=z.get(k)}
 assert set(diff)=={"process_cpu_seconds_total"} and 0<diff["process_cpu_seconds_total"]["delta"]<1
 foreign_deltas[key]=diff
 args=["python3","-c",(resident/"live_probe.py").read_text()]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 res=subprocess.run(args,input=json.dumps(dict(owner=o,NPU_count=16)).encode(),capture_output=True,timeout=120);(j/(key+".owner.stdout")).write_bytes(res.stdout);(j/(key+".owner.stderr")).write_bytes(res.stderr);res.check_returncode();live=json.loads(res.stdout)
 ids={v["pid"]:v["identity"]for v in members[key]["owned_targets"]};assert live["npu_worker_pids"]==members[key]["npu_worker_pids"]and all(ids[v["pid"]]==v["identity"]for v in live["owned_targets"]if v["pid"]in live["npu_worker_pids"])
 proofs[key]=dict(native_counters_before=nativeA,native_counters_after=nativeZ,live=ref(j/(key+".owner.stdout")),before_owner=ref(old/("before_"+key+".stdout")),after_owner=ref(old/("after_"+key+".stdout")),before_metrics=ref(old/("before_"+key+".metrics")),after_metrics=ref(old/("after_"+key+".metrics")))
raw=(old/"fullCLI_CPU.stdout").read_text();ev=[json.loads(l)for l in raw.splitlines()if l.startswith('{"event":')]
assert ev[0]["event"]=="task_acl_init"and ev[-1]["event"]=="task_acl_finalize"and ev[0]["returncode"]==ev[-1]["returncode"]==0
full=next(v for v in ev if v["event"]=="fullCLI_PP32_exactplan");assert (full["TP"],full["PP"],full["DCP"],full["K"],full["world"],full["nnodes"])==(8,2,8,3,16,1)
assert full["workers"]==full["models"]==full["inference"]==full["NPU_tensors"]==full["SDKcommunicators"]==0
installed=json.loads((old/"installed_sources.json").read_text())
for p in installed:assert ref(p["path"])==p and ref(old/"plugin_src"/Path(p["path"]).name)["sha256"]==p["sha256"]
limits=["CPUlegal-only; actualCPU1 nativeEngineArgs result reused, no rerun/noGPUmodels or new inference; TP8PP2/K3/42,36 nativefit/fullE2E/performance unknown","CPU1 wholejobINVALID because broad_total included process_cpu_seconds_total expected .02CPUtime increase perAPIserver duringqueries; allnative vllm counters unchanged/SDK0/currentNPU32same. Originalsource/raw preserved; correctedreadonlyreduction doesnotrewriteCPU1 status","Singlecontroller GPUreplacement pending/no currentprivatepolicy edits/operatorchanges; CurrentNone/stablecapacity/globalboundunknown"]
out=dict(at=utc(),CPU_config_VALID=True,exact_fullCLI=full,SDKinit_finalize0=True,current_native32_same_idle_counters=True,new_workers=0,new_models=0,new_inference=0,new_NPU_tensors=0,CPU1_original_INVALID=True,actualCPU1_config_reused=True,replayed_nativeCPU_config_calls=0,source=ref(old/"fullCLI_CPU.stdout"),candidate_plan=ref(old/"candidate_plan.json"),private_sources=ref(old/"installed_sources.json"),native_proofs=proofs,nonnative_process_CPU_deltas=foreign_deltas,Current=None,limits=limits)
atomic_json(j/"reduction.json",out)
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="ReadonlyactualCPU1 nativefullCLI TP8PP2DCP8 K3 legal/SDK0/native32counters same; CPU1 broadprocessCPU counterfixtureINVALID preserved/noGPUreplay",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="actualCPU1config/SDK/native32same/foreignprocessCPUdelta")],unknowns=limits,decision_request=None,next_check_at=None))
print(json.dumps(dict(valid=True,reduction=ref(j/"reduction.json"))))
