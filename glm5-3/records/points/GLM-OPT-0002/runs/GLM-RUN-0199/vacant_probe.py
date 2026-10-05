import sys,json,pathlib,subprocess,re
a=json.load(sys.stdin);o=a["old"];boot=pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
try:
 p=pathlib.Path("/proc/"+str(o["pid"]));s=(p/"stat").read_text();v=s[s.rfind(")")+2:].split()
 assert dict(boot_id=boot,start_ticks=v[19])!=o["identity"]or v[0]in["Z","X"]
except FileNotFoundError:pass
top=subprocess.check_output(["docker","top","glm52-single","-eo","pid,ppid,stat,comm,args"],text=True);unknown=[]
for l in top.splitlines()[1:]:
 x=l.split(None,4)
 if len(x)==5and x[2][0]not in["Z","X"]and(x[3].startswith("VLLM")or"native_acl_lifecycle.py cli serve "in x[4]or"/bin/vllm serve "in x[4]or"bishengir-compile "in x[4]):unknown.append(l)
assert not unknown,unknown
raw=subprocess.check_output(["npu-smi","info"],text=True,timeout=60)
all_npus=[int(v)for v in re.findall(r"^\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|\s+[^|]+\|\s+\d+\s+\|",raw,re.M)]
assert not all_npus,all_npus
print(json.dumps(dict(old_identity_inactive=True,active_native_roots=0,NPU_processes=0,npu_smi=raw,unknown_native=[])))
