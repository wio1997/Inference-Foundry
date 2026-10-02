from pathlib import Path
import json,subprocess,sys,hashlib,shlex
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent
def proc(args,timeout=30):
 p=subprocess.run(args,capture_output=True,timeout=timeout);return p
info=proc(["coredumpctl","--no-pager","info","3012814"]);assert info.returncode==0
assert "pd2_isolated_probe.py kv_producer 0"in info.stdout.decode()and"Signal: 6 (ABRT)"in info.stdout.decode()
(j/"core_info.txt").write_bytes(info.stdout+info.stderr)
dump=proc(["coredumpctl","--no-pager","dump","3012814","--output",str(j/"cpu_probe.core")],180);(j/"core_dump.log").write_bytes(dump.stdout+dump.stderr);assert dump.returncode==0
containerpid=proc(["docker","inspect","-f","{{.State.Pid}}","glm52-single"]).stdout.decode().strip();assert containerpid.isdigit()
stack=proc(["eu-stack","--core",str(j/"cpu_probe.core"),"--executable","/proc/"+containerpid+"/root/usr/local/python3.12.13/bin/python3.12","-1","-n","80","-m","-b"],100)
(j/"native_stack.txt").write_bytes(stack.stdout+stack.stderr)
base="/usr/local/python3.12.13/lib/python3.12/site-packages/mooncake/"
code='import pathlib,hashlib,json,importlib.metadata,subprocess; b=pathlib.Path('+repr(base)+'); rows=[{"path":str(f),"bytes":f.stat().st_size,"sha256":hashlib.sha256(f.read_bytes()).hexdigest()}for f in sorted(b.glob("*.so*"))];d=[{"name":x.metadata["Name"],"version":x.version}for x in importlib.metadata.distributions() if "mooncake"in x.metadata.get("Name","").lower()];print(json.dumps({"libs":rows,"distributions":d}));print(subprocess.run(["ldd",str(b/"engine.so")],capture_output=True,text=True).stdout);print(subprocess.run(["readelf","-n",str(b/"engine.so")],capture_output=True,text=True).stdout)'
ident=[]
for node in ["166","167"]:
 args=["docker","exec","glm52-single","python3","-c",code]
 if node=="167":args=["ssh","-o","BatchMode=yes","root@172.16.10.167",shlex.join(args)]
 p=proc(args,60);assert p.returncode==0,p.stderr.decode();f=j/("ABI_"+node+".txt");f.write_bytes(p.stdout+p.stderr)
 ident.append({"node":node,"installed":json.loads(p.stdout.decode().splitlines()[0])})
def ref(f):
 z=f.read_bytes();return{"path":str(f),"bytes":len(z),"sha256":hashlib.sha256(z).hexdigest()}
out={"at":utc(),"origin_core_PID":3012814,"core_info":ref(j/"core_info.txt"),"raw_core":ref(j/"cpu_probe.core"),"eu_stack_returncode":stack.returncode,"stack":ref(j/"native_stack.txt"),"identities":ident,"same_mooncake_sharedlibs":ident[0]["installed"]["libs"]==ident[1]["installed"]["libs"],"models_started":0,"signals":0,"weights":0,"inference":0,"limits":["ExistingtaskCPUcore only, not newengine/run orperformancecounter","Nativeunwind may bepartial becausecontainerELF paths unavailabletohost; retainexit/outputhonestly","Libraryhash/version/linkage identity alone noABI/runtimeinitialization/PDtransfer proof","Native vendor/libs untouched, no LD_PRELOAD or guardbypass"]}
atomic_json(j/"reduction.json",out)
atomic_json(j/"result.json",{"schema_version":1,"job_id":j.name,"status":"completed","summary":"ExistingownedCPUcore unwind andtwohost installedMooncakeABI identities retained;0modelsignals/starts/inference","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[],"evidence":[dict(id="reduction",locator="readonlyexistingtaskcore/ABIidentity",**ref(j/"reduction.json")),dict(id="stack",locator="nativeunwind actualstatus/output",**ref(j/"native_stack.txt"))],"unknowns":out["limits"],"decision_request":None,"next_check_at":None})
print(json.dumps({"stack_exit":stack.returncode,"same_library_hashes":out["same_mooncake_sharedlibs"],"reduction":ref(j/"reduction.json")}))
