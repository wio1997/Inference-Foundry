import json,hashlib,re,subprocess,shlex,sys
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0047";site=Path("/data/tiankuan/wio/glm52-pd/deploy")
def ref(f):
 d=f.read_bytes();return{"path":str(f),"bytes":len(d),"sha256":hashlib.sha256(d).hexdigest()}
def cmd(node,args):
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 return subprocess.check_output(args,timeout=90)
s=json.loads((r/"state.json").read_text());assert s["status"]=="failed"and s["failure_phase"]=="deploy"and not same_process(s["owner"])
spec=json.loads((r/"controller_spec.json").read_text());pins=spec["stages"][0]["sources"]
assert len(pins)==17
for x in pins:assert Path(x["path"]).read_bytes()==Path(x["snapshot"]).read_bytes()and ref(Path(x["path"]))["sha256"]==x["sha256"]
events=json.loads((r/"deployment_events.json").read_text());starts=[x for x in events if x["event"]=="native_role_started"]
assert len(starts)==2 and all(x["role"]=="P"for x in starts)
assert not(r/"pilot.phase.json").exists()and not(r/"pilot_attempts.json").exists()
owners=json.loads((r/"startup_model_identities.json").read_text());current={};logs={}
for node in ["166","167"]:
 data=cmd(node,["cat",str(site/("logs/P_run47_"+str(0 if node=="166"else 1)+".log"))]);f=j/("native_"+node+".log");f.write_bytes(data)
 lines=data.decode(errors="replace").splitlines()
 logs[node]={"ref":ref(f),"metadata_receipts":sum("GLM_DP_METADATA_INSTALLED "in x for x in lines),"mq_receipts":sum("GLM_ATOMIC_MQ_INSTALLED"in x for x in lines),"profile_lines":[x for x in lines if any(z in x for z in ["model weights take","Available KV cache memory","GPU KV cache size","init engine","KV Cache","memory profiling","Graph capturing","handshake"])]}
 top=cmd(node,["docker","top","glm52-single","-eo","pid,ppid,comm,args"]);(j/(node+".processes")).write_bytes(top)
 for name,args in [("sockets",["ss","-ltnp"]),("ephemeral",["docker","exec","glm52-single","cat","/proc/sys/net/ipv4/ip_local_port_range"])]:(j/(node+"."+name)).write_bytes(cmd(node,args))
 o=next(x for x in owners.values()if x["host"]==node)
 code="import pathlib,json;p=pathlib.Path('/proc/"+str(o["pid"])+"');out={'present':p.exists()};\nif p.exists():\n s=(p/'stat').read_text();out.update(boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=s[s.rfind(')')+2:].split()[19],argv=[x.decode()for x in(p/'cmdline').read_bytes().split(bytes([0]))if x]);\nprint(json.dumps(out))"
 a=json.loads(cmd(node,["python3","-c",code]));current[node]=a
 if a["present"]:assert a["boot_id"]==o["identity"]["boot_id"]and a["start_ticks"]==o["identity"]["start_ticks"]and a["argv"]==o["argv"]
 else:assert not any("native_acl_lifecycle.py cli serve "in x or"VLLM::"in x for x in top.decode().splitlines())
 index=json.loads((j.parents[1]/"jobs/PD2-CPU-DIAGNOSTIC-20261002T0158Z/source_index.json").read_text())
 for entry in index:assert cmd(node,["docker","exec","glm52-single","sha256sum",entry["native_path"]]).decode().split()[0]==entry["sha256"]
assert all(x["metadata_receipts"]==16 for x in logs.values())
text=(j/"native_167.log").read_text();assert "Address already in use"in text and"tcp://172.16.10.167:35023"in text
lines=text.splitlines();idx=next(i for i,l in enumerate(lines)if "Address already in use"in l);error=lines[max(0,idx-18):idx+6]
for node in ["166","167"]:
 before=(r/("before_"+node+".sockets")).read_text();assert not re.search(r":350(?:0[0-9]|[12][0-9]|3[01])\s",before)
out={"run_id":r.name,"at":utc(),"measurement_valid":True,"functional_acceptance":False,"verdict":"INCONCLUSIVE","status":"failed","failure_phase":"deploy","source_pins":len(pins),"controller":s,"native_starts":starts,"D_started":False,"inference_attempts":0,"effective_output_tokens":0,"first_observed_error":error,"observed_bind_failure":"tcp://172.16.10.167:35023 EADDRINUSE","port_owner_at_failure":"unknown","current_owners":current,"prior_owners":owners,"native_logs":logs,"deploy_phase":json.loads((r/"deploy.phase.json").read_text()),"deploy_log":ref(r/"deploy.log"),"port_layout":{"P_KV":[35000,35031],"P_rpc":35020,"P_master":35030,"D_KV":[35100,35131],"D_rpc":35120,"D_master":35130,"kernel_ephemeral":"32768..60999 actualbothhosts","finding":"fixedKV namespace overlapsconfiguredRPC/master and ephemeral allocator band; actual35023holder notcaptured, no specificcausalityclaim"},"signals":0,"new_models":0,"new_requests":0,"limits":["P0engineinit isnot groupreadiness/dualresidentfit","NoDmodel/noinference/nocapacity/OOMverdict","FuturestaticKV ports wereunreserved while nativebind0allocatedephemeralports; overlaprisks distinctfromprovenEADDRINUSE","PublicAPI/store/PDfunction remainsunproven"]}
atomic_json(j/"reduction.json",out);atomic_json(r/"reduction_brief.json",out)
m=json.loads((r/"manifest.json").read_text());m.update(status="failed",valid=False,verdict="INCONCLUSIVE",failure_phase="deploy",reduction=ref(j/"reduction.json"),results={"inference_attempts":0,"effective_output_tokens":0,"D_started":False});atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# "+r.name+"\n\nNative P startup failed: Mooncake DP1/TP7/DCP7/EP23 KV sendingthread bind tcp://172.16.10.167:35023 EADDRINUSE; actual occupyingowner unknown. P0 subsequently completedengineinitialization but waitedfor sharedDPready; P1 exited. D neverstarted, zero inference/outputs, no OOM/fit/capacity verdict. 17frozenpins andnative9sources checked; bothP16metadatareceipts. FixedKV35000..35031 overlapsRPC35020/master35030 and OSephemeral32768..60999. NewRun usesdisjointports outsideephemeralband, not proof ofspecificculprit. CurrentNone.\n")
atomic_json(j/"result.json",{"schema_version":1,"job_id":j.name,"status":"completed","summary":"Run47readonlyfailureaudit: P1nativeKV35023 EADDRINUSE, exactholderunknown, P0initnotgroupready/P1gone, Dnotstarted/0inference;17pins/native9sourcechecked","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[],"evidence":[dict(id="reduction",locator="nativefirsterror/source/currentowners/0D/0requests",**ref(j/"reduction.json"))],"unknowns":out["limits"]+["Portowneratfailureunknown"],"decision_request":None,"next_check_at":None})
print(json.dumps({"audit":True,"P_receipts":{k:v["metadata_receipts"]for k,v in logs.items()},"current_present":{k:v["present"]for k,v in current.items()},"outputs":0,"reduction":ref(j/"reduction.json")}))

