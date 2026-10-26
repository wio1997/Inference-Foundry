#!/usr/bin/env python3
"""Bounded read-only recovery probe for the original interrupted L17 client."""
import base64, datetime, hashlib, importlib.util, json, pathlib
ROOT=pathlib.Path("/Users/wio/work/sglang-glm53-33-41-deploy/two-sdma-project-chats-20261017/workspaces/glm53/repo/ascend-framework-comparison/glm53")
RID="20261019T-L17-sglang-formal-C32-native-v1"
OUT=ROOT/"runs"/"20261019T-L17-formal-interruption-readonly-recovery-v1"
HOST=r'''
import base64,datetime,hashlib,json,pathlib,subprocess,socket
assert socket.gethostname()=="B7-23U-Atlas900RCKA3-43"
root=pathlib.Path("/data/wio/inference-foundry/glm53/runs/20261019T-L17-sglang-formal-C32-native-v1")
assert root.is_dir()
result={"project_id":"glm53","hostname":socket.gethostname(),"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"run_root":str(root),"files":{}}
for rel in ["controller-status.json","host-settings.json","client-settings.json","client.cid","host-controller.once","output/formal-client.once","output/formal-progress.json","output/formal-result.json","output/need-cache-gate.json","output/cache-gate.json"]:
 p=root/rel
 if p.is_file():
  assert not p.is_symlink()
  raw=p.read_bytes();item={"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"mtime_ns":p.stat().st_mtime_ns}
  if len(raw)<=16*1024**2:item["base64"]=base64.b64encode(raw).decode()
  else:item["content_not_transferred"]="exceeds16MiB"
  result["files"][rel]=item
 else:result["files"][rel]={"exists":False}
for rel in ["client-stdout.txt","client-stderr.txt"]:
 p=root/rel
 if p.is_file():
  with p.open("rb") as f:
   f.seek(max(0,p.stat().st_size-65536));raw=f.read(65536)
  result["files"][rel]={"bytes":p.stat().st_size,"tail_base64":base64.b64encode(raw).decode()}
result["artifact_stats"]=[{"path":str(p.relative_to(root)),"bytes":p.stat().st_size,"mtime_ns":p.stat().st_mtime_ns} for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink()][:150]
name="glm53-sglang-C32-L17-native-formal-v1-20261019-43"
fmt='{"Id":{{json .Id}},"Name":{{json .Name}},"Image":{{json .Image}},"Labels":{{json .Config.Labels}},"State":{{json .State}}}'
r=subprocess.run(["docker","inspect",name,"--format",fmt],capture_output=True,text=True,timeout=20)
result["owned_container"]={"rc":r.returncode,"stdout":r.stdout,"stderr":r.stderr}
if r.returncode==0:
 c=json.loads(r.stdout)
 assert c["Name"]=="/"+name and c["Image"]=="sha256:008d2ebcd28a18f4dfbc855d23d763a2bdc63c4ce82c6b7b761580ad6d15d0a7"
 assert c["Labels"].get("project_id")=="glm53" and c["Labels"].get("run_id")=="20261019T-L17-sglang-formal-C32-native-v1"
state=root/"controller-status.json"
if state.is_file():
 pid=json.loads(state.read_text()).get("host_docker_client_pid")
 if type(pid) is int:
  proc=pathlib.Path("/proc")/str(pid)
  item={"pid":pid,"exists":proc.exists()}
  if proc.exists():
   item["comm"]=(proc/"comm").read_text().strip()
   item["cmdline_contains_owned_name"]=name.encode() in (proc/"cmdline").read_bytes()
  result["original_docker_client_process"]=item
print(json.dumps(result))
'''
def main():
 OUT.mkdir(exist_ok=False,parents=True)
 p=ROOT/"scripts/stage_sglang_C32_packed_normal_and_rollback_v1.py"
 assert hashlib.sha256(p.read_bytes()).hexdigest()=="13545bf1c90368d57113dc810738da12c64d3efa5c01ba3fd38057e21cb1724f"
 spec=importlib.util.spec_from_file_location("readonly_stage_transport",p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 (OUT/"host-code.py").write_text(HOST)
 wire=mod.invoke(43,HOST,timeout=120)
 (OUT/"node-43-wire.json").write_text(json.dumps(wire,indent=2)+"\n")
 assert wire["rc"]==0,wire["stderr"][:1000]
 result=json.loads(wire["stdout"])
 for rel,item in result["files"].items():
  if "base64" in item:
   raw=base64.b64decode(item["base64"]);assert hashlib.sha256(raw).hexdigest()==item["sha256"]
   dest=OUT/"artifacts"/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
 receipt={"project_id":"glm53","original_run":RID,"read_only":True,"model_requests":0,"NPU_calls":0,"SDK_calls":0,"original_launcher_replayed":False,"remote_files_written":0,"wire_sha256":hashlib.sha256((OUT/"node-43-wire.json").read_bytes()).hexdigest(),"utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (OUT/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
 summary=dict(receipt)
 summary["controller"]=json.loads((OUT/"artifacts/controller-status.json").read_text()) if (OUT/"artifacts/controller-status.json").is_file() else None
 for rel in ["output/formal-progress.json","output/formal-result.json"]:
  p=OUT/"artifacts"/rel
  if p.is_file():
   v=json.loads(p.read_text());summary[rel]={"phase":v.get("phase"),"error":v.get("error"),"warmup_rows":len(v.get("warmup",[])),"warmup_passes":sum(x.get("pass") is True for x in v.get("warmup",[])),"backend_HTTP_attempts":v.get("backend_HTTP_attempts"),"formal_present":v.get("formal") is not None,"gateway_rows":len(v.get("gateway",[]))}
 summary["container"]=result["owned_container"]
 summary["docker_client_process"]=result.get("original_docker_client_process")
 summary["artifact_stats"]=result["artifact_stats"]
 print(json.dumps(summary,indent=2))
if __name__=="__main__":main()

