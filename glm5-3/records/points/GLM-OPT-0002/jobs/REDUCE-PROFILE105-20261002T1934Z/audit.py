
from pathlib import Path
import json,sys,subprocess,shlex,hashlib,re,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from sse_observer import NativeSSEObserver
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0105";s=json.loads((r/"state.json").read_text());assert s["status"]=="failed"and s["failure_phase"]=="profile"and not same_process(s["owner"])
for x in json.loads((r/"controller_spec.json").read_text())["stages"][0]["sources"]:
 assert Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes()and hashlib.sha256(Path(x["path"]).read_bytes()).hexdigest()==x["sha256"]
gate=json.loads((r/"profile_gate.json").read_text());expected={x["container_pid"]for x in gate["workers"]};assert len(expected)==16
text=(r/"msprof.stdout").read_text()
for command in ["start","stop","quit"]:
 ids={int(x)for x in re.findall(r"dynamic profiling for pid (\d+) "+command+r" success",text)};assert ids==expected
assert not(r/"profile_shutdown_unknown.json").exists()
controls=json.loads((r/"profile_control.json").read_text());assert [x["command"]for x in controls]==["start","stop","quit"];window=controls[1]["monotonic"]-controls[0]["monotonic"];assert 5<=window<7
row=json.loads((r/"attempts.json").read_text())[0];raw=Path(row["wire"]["path"]).read_bytes();assert hashlib.sha256(raw).hexdigest()==row["wire"]["sha256"]and len(raw)==row["wire"]["bytes"]
obs=NativeSSEObserver(max_bytes=2097152,collect_contract=True);obs.feed(raw);c=obs.contract();assert c["done"]and not c["unknown"]and not c["native_error"]and c["finish_reasons"]=={"0":"length"}and c["usage"]==dict(prompt_tokens=21,completion_tokens=1024,total_tokens=1045)
frames=[]
for part in raw.replace(b"\r\n",b"\n").split(b"\n\n"):
 d=b"\n".join(l[5:].removeprefix(b" ")for l in part.splitlines()if l.startswith(b"data:"))
 if d and d!=b"[DONE]":frames.append(json.loads(d))
assert sum(len(c.get("token_ids")or[])for f in frames for c in f.get("choices",[]))==1024 and any(len(f.get("prompt_token_ids")or[])==21 for f in frames)
body=json.loads((r/"profile_C1.body.json").read_text());assert body["cache_salt"].startswith(r.name)and body["max_tokens"]==1024 and body["ignore_eos"]is True and "kv_transfer_params"not in body
indexcode="""from pathlib import Path
import json,hashlib,sys
root=Path(sys.argv[1]);files=[];profiles=[]
for d in sorted(root.glob('PROF_*')):
 info=json.loads((d/'host/info.json').read_text())
 for n in ['host/start_info','host/end_info']:
  f=d/n
  try:info[n]=json.loads(f.read_text())
  except ValueError:info[n]=f.read_text()[:300]
 allfiles=[f for f in d.rglob('*')if f.is_file()]
 info.update(directory=str(d),host_files=len([f for f in allfiles if f.relative_to(d).parts[0]=='host']),device_files=[str(f)for f in allfiles if f.relative_to(d).parts[0]!='host'],total_bytes=sum(f.stat().st_size for f in allfiles))
 profiles.append(info)
 for f in allfiles:
  st=f.stat();h=hashlib.sha256()
  with f.open('rb')as z:
   for b in iter(lambda:z.read(1048576),b''):h.update(b)
  assert f.stat().st_size==st.st_size
  files.append(dict(path=str(f),bytes=st.st_size,sha256=h.hexdigest()))
print(json.dumps(dict(root=str(root),profiles=profiles,files=files,total_bytes=sum(f['bytes']for f in files))))
"""
astmodule=__import__("ast");astmodule.parse(indexcode)
z=subprocess.run(["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(["python3","-c",indexcode,"/data/tiankuan/wio/glm52-pd/deploy/private/profile_run105"])],capture_output=True,timeout=180);(j/"index.stderr").write_bytes(z.stderr);z.check_returncode();idx=json.loads(z.stdout);atomic_json(j/"raw_index.json",idx)
assert len(idx["profiles"])==16 and {int(x["pid"])for x in idx["profiles"]}==expected and all(x["total_bytes"]>0 for x in idx["profiles"])
def metric(f):
 out={}
 for l in f.read_text().splitlines():
  if l.startswith("vllm:"):
   k=l.split("{")[0].split()[0]
   if k.endswith("_total")or k in ["vllm:num_requests_running","vllm:num_requests_waiting","vllm:kv_cache_usage_perc"]:out[k]=out.get(k,0)+float(l.rsplit(" ",1)[1])
 return out
counters={}
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
API=json.loads((r/"adopted_model_identities.json").read_text());members=json.loads((r/"native_member_identities.json").read_text());acks={}
check="""from pathlib import Path
import json,sys,re,subprocess
a=json.load(sys.stdin);o=a['owner'];boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for x in a['workers']+[dict(pid=o['pid'],identity=o['identity'])]:
 p=Path('/proc/'+str(x['pid']));s=(p/'stat').read_text();v=s[s.rfind(')')+2:].split();assert v[0]not in ['Z','X']and dict(boot_id=boot,start_ticks=v[19])==x['identity']
assert [x.decode()for x in Path('/proc/'+str(o['pid'])+'/cmdline').read_bytes().split(bytes([0]))if x]==o['argv']
pol=json.loads((Path(o['argv'][1]).parent/'issue_budget_policy.json').read_text());assert pol==a['policy']
b=subprocess.check_output(['npu-smi','info'],text=True,timeout=60);ids={int(x)for x in re.findall(r'^\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|\s+VLLMWorker',b,re.M)};assert ids=={x['pid']for x in a['workers']}
# Only inspect profiler executables; never signal any process.
alive=[]
for f in Path('/proc').iterdir():
 if not f.name.isdigit():continue
 try:
  args=(f/'cmdline').read_bytes().split(bytes([0]))
  if args and Path(args[0].decode()).name=='msprof':alive.append(int(f.name))
 except (OSError,UnicodeError):pass
assert not alive
print(json.dumps(dict(API_same=True,NPU16_same=True,policy=pol,profiler_executables_remaining=alive)))
"""
for key,o in API.items():
 policy=dict(schema_version=1,cohort_id="GLM-COHORT-0089"if key=="D0"else"GLM-COHORT-0104",budget_tokens=4096,prefill_threshold_tokens=1024,prefill_cadence=1,serial=3 if key=="D0"else 1);ws=[x for x in members[key]["owned_targets"]if x["pid"]in members[key]["npu_worker_pids"]]
 args=["python3","-c",check]
 if o["host"]=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(dict(owner=o,workers=ws,policy=policy)).encode(),capture_output=True,timeout=90);(j/(key+".owner.stdout")).write_bytes(z.stdout);(j/(key+".owner.stderr")).write_bytes(z.stderr);z.check_returncode();acks[key]=json.loads(z.stdout)
 before=metric(r/("initial_"+key+".metrics"));after=metric(r/("after_"+key+".metrics"));delta={k:after[k]-v for k,v in before.items()if k.endswith("_total")}
 assert delta["vllm:prompt_tokens_total"]==(21 if key=="D1"else 0)and delta["vllm:generation_tokens_total"]==(1024 if key=="D1"else 0)and delta["vllm:request_success_total"]==(1 if key=="D1"else 0)and delta.get("vllm:num_preemptions_total",0)==0
 with http.open("http://172.16.10."+o["host"]+":"+str(o["port"])+"/metrics",timeout=10)as reply:b=reply.read();assert reply.status==200
 f=j/(key+".current.metrics");f.write_bytes(b);now=metric(f);assert all(now.get(k,0)==0 for k in ["vllm:num_requests_running","vllm:num_requests_waiting","vllm:kv_cache_usage_perc"])
 assert all(now[k]==after[k]for k in after if k.endswith("_total"));counters[key]=dict(delta=delta,idle=True)
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
out=dict(at=utc(),controller_status="failed",verdict="INVALID",actual_driver_failure="Postcapture nestedindexcode newline SyntaxError; originalcontroller/source/raw unchanged, no GPUrerun",functional_acceptance=False,completed_public_subexperiment_valid=True,effective_public_output_tokens=1024,profile_capture_processes=16,profile_start_stop_quit_all16_confirmed=True,profile_collection_request_window_s=window,device_raw_processes=sum(bool(x["device_files"])for x in idx["profiles"]),raw_index=ref(j/"raw_index.json"),raw_total_bytes=idx["total_bytes"],request=row,body=ref(r/"profile_C1.body.json"),msprof_stdout=ref(r/"msprof.stdout"),profile_command=ref(r/"profile_command.json"),control=controls,native_counters=counters,physical_same=acks,model_operations=0,limits=["Only validated completed publicsubexperiment credit1024; wrapperRun remainsINVALID afterpostcapture indexfixturefailure, noCurrent/KEEP/capacity claim","Profileroverhead included inC1wall; noordinaryperformancecomparison/hardwarebound. Device trace coverage, host/deviceclock calibration andkernel attribution require actualraw parsing","Rank_id nativeprofinfo -1; ownership/NSpid/actualworkercomm mapping establishes16requestednativeworkers; no assumptions from rank_id field","No SDK APIs called by HTTP/index scripts; residentnativeCLI SDKinit0 evidence reused, no fakefinalize"])
atomic_json(j/"reduction.json",out);atomic_json(r/"reduction_brief.json",out);b=(j/"reduction.json").read_bytes()
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="105 wrapperINVALID nestedindexnewline afteractual16workerstart-stop-quit andcomplete1024publicwire; readonlyraw/counters/epoch/idle audit; no GPUrerun/KEEP",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="originalprofile/wire/raw16/processSDKidentity/counters/idle")],unknowns=out["limits"],decision_request=None,next_check_at=None));print(json.dumps({k:out[k]for k in ["effective_public_output_tokens","profile_capture_processes","device_raw_processes","raw_total_bytes","profile_collection_request_window_s"]}))
