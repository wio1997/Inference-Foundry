from common import *
from native_client_execution import run_native_client
mode=sys.argv[1]
native=["python3","-u",str(plug/"native_acl_lifecycle.py"),"script",str(r/"features_client.py"),mode]
shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source "+str(site/"scripts/pd_common_env.sh")+"; export PYTHONPATH="+str(bundle)+":"+str(r)+":$PYTHONPATH; exec "+shlex.join(native)
exe=run_native_client(["docker","exec","glm52-single","bash","-c",shell],record_path=r/(mode+"_execution.json"),stdout_path=r/(mode+".stdout"),stderr_path=r/(mode+".stderr"),expected_native_argv=native,timeout_s=1100,guard=guard)
assert exe["status"]=="succeeded"and not exe["native_client_alive"]and not exe["signal_attempts"]
acks=[json.loads(l)for l in(r/(mode+".stdout")).read_text().splitlines()if l.startswith('{"event":')];assert [x["event"]for x in acks]==["task_acl_init","task_acl_finalize"]and all(x["returncode"]==0for x in acks)
assert json.loads((r/(mode+"_summary.json")).read_text())["valid"]
