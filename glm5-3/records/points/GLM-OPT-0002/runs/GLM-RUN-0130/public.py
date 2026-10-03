from pathlib import Path
import sys,json,subprocess,shlex,re,time,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
from standalone_service_config import checked_config
from owner_guard import start_watchdog,guard
start_watchdog();r=Path(__file__).parent;g=r.parents[4]
z=subprocess.run([sys.executable,str(r/"epoch_check.py"),"before_public"],capture_output=True,timeout=240);(r/"before_public.stdout").write_bytes(z.stdout);(r/"before_public.stderr").write_bytes(z.stderr);z.check_returncode()
config,_=checked_config(r/"service_config.json");assert not re.search(r":8000\s",subprocess.check_output(["ss","-ltnp"],text=True))
argv=["/usr/local/python3.12.13/bin/python3","/data/tiankuan/wio/glm52-pd/deploy/plugins/coupled_run130/native_acl_lifecycle.py","script",str(g/"runtime/standalone_service_entry.py"),"--config",str(r/"service_config.json"),"--host","0.0.0.0","--port","8000"]
log=r/"public.gateway.log"
shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source /data/tiankuan/wio/glm52-pd/deploy/scripts/pd_common_env.sh; export PYTHONPATH="+str(g/"runtime")+":$PYTHONPATH; exec "+shlex.join(argv)+" > "+shlex.quote(str(log))+" 2>&1"
guard();subprocess.run(["docker","exec","-d","glm52-single","bash","-c",shell],capture_output=True,check=True)
http=urllib.request.build_opener(urllib.request.ProxyHandler({}));deadline=time.monotonic()+60;host=None
while True:
 guard();top=subprocess.check_output(["docker","top","glm52-single","-eo","pid,ppid,comm,args"],text=True);matches=[int(x.split()[0])for x in top.splitlines()[1:]if"standalone_service_entry.py --config "+str(r/"service_config.json")in x]
 assert len(matches)==1
 pid=matches[0];s=Path("/proc/"+str(pid)+"/stat").read_text();vals=s[s.rfind(")")+2:].split()
 host=dict(pid=pid,boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),start_ticks=vals[19])
 assert [v.decode()for v in Path("/proc/"+str(pid)+"/cmdline").read_bytes().split(bytes([0]))if v]==argv
 atomic_json(r/"public_service_host.json",dict(at=utc(),identity=host,argv=argv,port=8000,config_path=str(r/"service_config.json")))
 try:
  with http.open("http://127.0.0.1:8000/healthcheck",timeout=3)as response:h=json.loads(response.read())
  if h["status"]=="ok"and h["request_num"]==0:break
 except Exception:pass
 assert time.monotonic()<deadline
 time.sleep(1)
ports=subprocess.check_output(["ss","-ltnp"],text=True);line=next(x for x in ports.splitlines()if re.search(r":8000\s",x));assert"pid="+str(pid)+","in line
with http.open("http://127.0.0.1:8000/control/replicas",timeout=10)as response:placement=json.loads(response.read())
assert len(placement["replicas"])==1and placement["replicas"][0]["id"]=="D0"and not placement["replicas"][0]["group_faulted"]
s=log.read_text();assert"GLM_SERVICE_ENTRY_INSTALLED "in s
acks=[json.loads(l)for l in s.splitlines()if l.startswith('{"event":')];assert acks[0]["event"]=="task_acl_init"and acks[0]["returncode"]==0
marker=next(json.loads(l.split("GLM_SERVICE_ENTRY_INSTALLED ",1)[1])for l in s.splitlines()if"GLM_SERVICE_ENTRY_INSTALLED "in l)
assert marker["native_domains"]==config["native_domains"]
atomic_json(r/"public_service_proof.json",dict(at=utc(),host=host,argv=argv,container_marker=marker,listener=line,placement=placement,SDK_init0=True,retained=True,owner_journal_preserved=True))
print("coupled native public entry ready")
