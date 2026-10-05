import pathlib,json,subprocess,re,sys
a=json.load(sys.stdin);o=a["owner"];boot=pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
def proc(pid):
 try:
  p=pathlib.Path("/proc/"+str(pid));s=(p/"stat").read_text();v=s[s.rfind(")")+2:].split()
  return dict(identity=dict(boot_id=boot,start_ticks=v[19]),state=v[0],argv=[x.decode()for x in(p/"cmdline").read_bytes().split(bytes([0]))if x])
 except FileNotFoundError:return None
top=subprocess.check_output(["docker","top","glm52-single","-eo","pid,ppid,stat,comm,args"],text=True);rows={}
for l in top.splitlines()[1:]:
 x=l.split(None,4)
 if len(x)==5:rows[int(x[0])]=(int(x[1]),x[2],x[3],x[4])
if o.get("pid")is None:
 found=[pid for pid in rows if (proc(pid)or{}).get("argv")==o["argv"]]
 assert len(found)==1,dict(native_root_matches=found)
 pid=found[0];o.update(pid=pid,identity=proc(pid)["identity"])
p=proc(o["pid"]);assert p and p["identity"]==o["identity"]and p["state"]not in["Z","X"]and p["argv"]==o["argv"]
ids={o["pid"]}
while True:
 more=ids|{pid for pid,x in rows.items()if x[0]in ids}
 if more==ids:break
 ids=more
targets=[dict(pid=pid,identity=proc(pid)["identity"],state=proc(pid)["state"],comm=rows[pid][2])for pid in sorted(ids)if pid in rows and proc(pid)and proc(pid)["state"]not in["Z","X"]]
unknown=[pid for pid,x in rows.items()if x[1][0]not in["Z","X"]and("/bin/vllm serve "in x[3]or"native_acl_lifecycle.py cli serve "in x[3]or x[2].startswith("VLLM")or"bishengir-compile "in x[3])and pid not in ids]
assert not unknown,dict(unattributed_native=unknown)
raw=subprocess.check_output(["npu-smi","info"],text=True,timeout=60)
npus={int(v)for v in re.findall(r"^\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|\s+VLLMWorker",raw,re.M)}
all_npus={int(v)for v in re.findall(r"^\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|\s+[^|]+\|\s+\d+\s+\|",raw,re.M)}
assert all_npus==npus,"unattributed NPU process"
assert npus<=ids and (a.get("NPU_count")is None or len(npus)==a["NPU_count"])
env=dict(v.decode().split("=",1)for v in pathlib.Path("/proc/"+str(o["pid"])+"/environ").read_bytes().split(bytes([0]))if b"="in v)
print(json.dumps(dict(root=o,root_alive=True,owned_targets=targets,npu_worker_pids=sorted(npus),npu_smi=raw,env={k:env.get(k)for k in["VLLM_USE_V2_MODEL_RUNNER","VLLM_HOST_IP","VLLM_ENABLE_RESPONSES_API_STORE","HCCL_BUFFSIZE","HCCL_LOGIC_SUPERPOD_ID","HCCL_NPU_SOCKET_PORT_RANGE","HCCL_HOST_SOCKET_PORT_RANGE","GLM_ISSUE_BUDGET_POLICY","GLM_ISSUE_BUDGET_COHORT","GLM_ISSUE_BUDGET_DEFAULT","VLLM_PP_LAYER_PARTITION","PROFILING_MODE"]})))
