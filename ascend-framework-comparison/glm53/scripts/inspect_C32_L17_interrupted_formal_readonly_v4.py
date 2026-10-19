#!/usr/bin/env python3
"""Single bounded original-client read-only probe; no original-launcher replay."""
import ast,base64,datetime,hashlib,importlib.util,json,pathlib,shlex,stat,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1]
RID="20261019T-L17-sglang-formal-C32-native-v1"
OUT=ROOT/"runs"/"20261019T-L17-formal-interruption-readonly-recovery-v4"
HOST_SOURCE="scripts/inspect_C32_L17_interrupted_formal_readonly_v2.py"
HOST_SOURCE_SHA="62da025d99d91ac689f6dd059655804c07cbcdefd5c522ca5e6be86ca22272b4"
HOST_SHA="f1b21bec9981dcab0a47dfe320bef3efa99900a2b0633cf3d8d454fa1dc63700"
FRESH_CONTROL_SOCKET=pathlib.Path("/Users/wio/work/sglang-glm53-pd/ssh-control-l17-recovery-20261019")
INVENTORY="scripts/inventory_readonly.py"
INVENTORY_SHA="e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777"
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def host_plan():
 p=ROOT/HOST_SOURCE;assert sha(p)==HOST_SOURCE_SHA
 tree=ast.parse(p.read_text())
 host=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=="HOST" for x in n.targets))
 assert hashlib.sha256(host.encode()).hexdigest()==HOST_SHA
 ast.parse(host);return host
def main():
 host=host_plan()
 p=ROOT/INVENTORY;assert sha(p)==INVENTORY_SHA
 spec=importlib.util.spec_from_file_location("l17_readonly_inventory",p);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
 assert FRESH_CONTROL_SOCKET.exists() and stat.S_ISSOCK(FRESH_CONTROL_SOCKET.stat().st_mode),"A freshly authenticated SSH master is required; do not reuse the stalled original master"
 b.SSH=list(b.SSH)
 socket_index=b.SSH.index("-S")+1
 assert b.SSH[socket_index]=="/Users/wio/work/sglang-glm53-pd/ssh-control"
 b.SSH[socket_index]=str(FRESH_CONTROL_SOCKET)
 OUT.mkdir(parents=True,exist_ok=False)
 (OUT/"host-code.py").write_text(host)
 receipt={"project_id":"glm53","original_run":RID,"read_only":True,"model_requests":0,"NPU_calls":0,"SDK_calls":0,"remote_files_written":0,"original_launcher_replayed":False,"formal_counts_booked":False,"remote_state":"UNKNOWN","launcher_sha256":sha(__file__),"host_code_sha256":HOST_SHA,"control_socket":str(FRESH_CONTROL_SOCKET),"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"stage":"MAPPING"}
 def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2)+"\n")
 try:
  pod,hostname=b.NODES[43]
  mapping=b.invoke(["kubectl get pod -n default "+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
  save("mapping-wire.json",mapping)
  assert mapping["rc"]==0 and mapping["stdout"]=="172.16.12.43","Mapping unavailable or mismatched; host probe not executed"
  receipt["stage"]="ORIGINAL_CLIENT_READONLY"
  code="import socket\nassert socket.gethostname()=="+repr(hostname)+"\n"+host
  wire=b.invoke([shlex.join(["kubectl","exec","-i","-n","default",pod,"-c","hostctl","--","nsenter","-t","1","-m","--","python3","-"])],code,timeout=120)
  save("node-43-wire.json",wire)
  assert wire["rc"]==0,"Original client read-only probe failed"
  result=json.loads(wire["stdout"]);assert result["project_id"]=="glm53" and result["hostname"]==hostname
  for rel,item in result["files"].items():
   if "base64" in item:
    raw=base64.b64decode(item["base64"]);assert hashlib.sha256(raw).hexdigest()==item["sha256"]
    target=OUT/"artifacts"/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
  receipt["stage"]="ORIGINAL_ARTIFACTS_CAPTURED_PENDING_PRIMARY_REVIEW"
  receipt["remote_state"]="ORIGINAL_ARTIFACTS_CAPTURED; not assumed qualified"
  receipt["wire_sha256"]=sha(OUT/"node-43-wire.json")
 except BaseException as error:
  receipt["error"]=type(error).__name__+": "+str(error)
  if isinstance(error,subprocess.TimeoutExpired):
   receipt["timeout_seconds"]=error.timeout
   save("transport-timeout.json",{"timeout_seconds":error.timeout,"stdout":(error.stdout or b"").decode("utf-8","replace") if isinstance(error.stdout,bytes) else error.stdout,"stderr":(error.stderr or b"").decode("utf-8","replace") if isinstance(error.stderr,bytes) else error.stderr})
  raise
 finally:
  save("receipt.json",receipt)
  print(json.dumps(receipt,indent=2),flush=True)
if __name__=="__main__":main()

