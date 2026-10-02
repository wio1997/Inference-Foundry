from pathlib import Path
import json,re,hashlib,subprocess,shlex,sys,statistics
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;p=j.parents[1];old=p/"runs/GLM-RUN-0054";r=p/"runs/GLM-RUN-0056"
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
plans=[json.loads((x/"planned_launch.json").read_text())for x in [old,r]]
for node in["166","167"]:
 xs=[next(x for x in a if x["node"]==node and x["role"]=="P")for a in plans]
 def norm(x,version):
  return [z.replace("run"+version,"runX").replace("glm"+version+"-","glmX-")for z in x["argv"]]
 assert norm(xs[0],"54")==norm(xs[1],"56")
 assert xs[0]["env"]["HCCL_BUFFSIZE"]=="768"and xs[1]["env"]["HCCL_BUFFSIZE"]=="512"
 assert {k:v for k,v in xs[0]["env"].items()if k!="HCCL_BUFFSIZE"}=={k:v for k,v in xs[1]["env"].items()if k!="HCCL_BUFFSIZE"}
def parse(f):
 text=f.read_text()
 process={}
 for line in text.splitlines():
  m=re.match(r"\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|\s*(\S+)\s*\|\s*(\d+)\s*\|",line)
  if m:
   n,c,pid,comm,mem=m.groups();assert comm=="VLLMWorker_DP";key=(int(n),int(c));assert key not in process;process[key]=dict(pid=int(pid),memory_MB=int(mem),comm=comm)
 hbm={}
 for line in text.splitlines():
  m=re.match(r"\|\s*(\d+)\s+(\d+)\s*\|\s*[0-9A-Fa-f:.]+.*?\s+(\d+)\s*/\s*(65536)\s*\|",line)
  if m:hbm[int(m.group(2))]=int(m.group(3))
 assert len(process)==16 and len(hbm)==16
 return process,hbm
rows=[]
for node in["166","167"]:
 past=old/("resident_P_"+node+".npu-smi");now=r/("resident_P_"+node+".npu-smi");assert now.exists()
 proc0,hbm0=parse(past);proc1,hbm1=parse(now);assert set(proc0)==set(proc1)
 prior=p/"runs/GLM-RUN-0055"/("prior_"+node+".owner_preflight.json");expected=set(x["pid"]for x in json.loads(prior.read_text())["owned_targets"]);assert all(x["pid"]in expected for x in proc0.values())
 owners=json.loads((r/"startup_model_identities.json").read_text());o=owners["P"+str(0 if node=="166"else 1)];pids=[x["pid"]for x in proc1.values()]
 code="from pathlib import Path;import json;out=[]\n"
 code+="root="+repr(o)+";p=Path('/proc/'+str(root['pid']));s=(p/'stat').read_text();assert dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=s[s.rfind(')')+2:].split()[19])==root['identity'];assert [z.decode()for z in(p/'cmdline').read_bytes().split(bytes([0]))if z]==root['argv']\n"
 code+="for pid in "+repr(pids)+":\n cur=pid;seen=set()\n while cur!=root['pid']:\n  assert cur>1 and cur not in seen;seen.add(cur);z=Path('/proc/'+str(cur)+'/stat').read_text();cur=int(z[z.rfind(')')+2:].split()[1])\n st=Path('/proc/'+str(pid)+'/stat').read_text();status=Path('/proc/'+str(pid)+'/status').read_text();ns=[int(z)for l in status.splitlines()if l.startswith('NSpid:')for z in l.split()[1:]];out.append(dict(pid=pid,start_ticks=st[st.rfind(')')+2:].split()[19],container_pid=ns[-1]))\nprint(json.dumps(out))"
 args=["python3","-c",code]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 identities=json.loads(subprocess.check_output(args,timeout=40));native=(r/("resident_P_"+node+".log")).read_text();receipts=[json.loads(l.split("GLM_UNUSED_SFA_WORKSPACE_GUARD_INSTALLED ",1)[1])for l in native.splitlines()if"GLM_UNUSED_SFA_WORKSPACE_GUARD_INSTALLED "in l];assert {x["container_pid"]for x in identities}=={x["pid"]for x in receipts}
 perproc=[dict(NPU=n,chip=c,physical_id=2*n+c,old_PID=proc0[(n,c)]["pid"],current_PID=proc1[(n,c)]["pid"],old_MB=proc0[(n,c)]["memory_MB"],current_MB=proc1[(n,c)]["memory_MB"],observed_reduction_MB=proc0[(n,c)]["memory_MB"]-proc1[(n,c)]["memory_MB"])for n,c in sorted(proc0)]
 rows.append(dict(node=node,past=ref(past),current=ref(now),prior_verified_cleanup_membership=ref(prior),current_P_API=o,current_worker_identities=identities,per_process=perproc,device_HBM_deltas_MB={str(k):hbm0[k]-hbm1[k]for k in hbm0}))
delta=[x["observed_reduction_MB"]for row in rows for x in row["per_process"]];assert len(delta)==32
limits=["Matched nativeP CLI/config except ownedplugin/engineIDs andHCCL768→512; P-only resident snapshots beforeD, no inference","Observed processmemory/HBM differences include native warmup/allocator/runtime timing, no isolated allocation-domain physicalsize attribution","32physicalNPUs/32Pworkers; not64devices, no D Graph/dualfit/fullE2E/capacity conclusion","FixedordinaryPG200/DPnativeformula preserved; no throughput/quality claim"]
out=dict(at=utc(),run_id=r.name,comparison_run=old.name,rows=rows,process_reduction_MB=dict(count=32,min=min(delta),median=statistics.median(delta),max=max(delta)),native_requests=0,models_started=0,signals=0,limits=limits);atomic_json(j/"reduction.json",out)
atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="32actualPworkers matched/owned: HCCLenv768→512 observes perprocess"+str(min(delta))+"-"+str(max(delta))+"MB lower residentmemory; sameCLI/nativePG; noDGraph/fit/capacity claim",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",locator="frozenbothnode residentP npu-smi/currentexactancestry/nativePIDreceipts/priorverifiedmembers",**ref(j/"reduction.json"))],unknowns=limits,decision_request=None,next_check_at=None));print(json.dumps(out["process_reduction_MB"]))

