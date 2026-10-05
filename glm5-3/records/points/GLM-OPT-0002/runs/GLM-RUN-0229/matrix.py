from common import *
shell="ulimit -c 0; export PD_LOCAL_IP=172.16.10.166 PD_NIC=business; source "+str(site/"scripts/pd_common_env.sh")+"; export PYTHONPATH="+str(bundle)+":"+str(r)+":$PYTHONPATH; python3 "+str(plug/"native_acl_lifecycle.py")+" script "+str(r/"matrix_client.py")
raw=run("166",["docker","exec","glm52-single","bash","-c",shell],"matrix_client",timeout=1000)
acks=[json.loads(l)for l in raw.decode().splitlines()if l.startswith('{"event":')];assert acks[0]["event"]=="task_acl_init"and acks[-1]["event"]=="task_acl_finalize"and acks[0]["returncode"]==acks[-1]["returncode"]==0
atomic_json(r/"matrix_client_SDK.json",dict(at=utc(),SDKinit_finalize0=True,summary=ref(r/"matrix_summary.json")))
