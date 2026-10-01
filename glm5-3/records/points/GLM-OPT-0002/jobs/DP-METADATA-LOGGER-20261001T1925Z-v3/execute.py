import pathlib,json,hashlib,subprocess
j=pathlib.Path(__file__).parent;job=json.loads((j/"job.json").read_text())
probe="""import json,logging,pathlib
from vllm.logger import init_logger
a=init_logger('coupled_dp_metadata');b=init_logger('vllm.glm_coupled_dp_metadata')
print(json.dumps({'external':{'level':a.getEffectiveLevel(),'INFO_enabled':a.isEnabledFor(logging.INFO)},'native_namespace':{'level':b.getEffectiveLevel(),'INFO_enabled':b.isEnabledFor(logging.INFO)},'root_level':logging.getLogger().getEffectiveLevel(),'inference_calls':0,'worker_constructed':False}))
"""
p=subprocess.run(["docker","exec","-i","glm52-single","python3","-"],input=probe.encode(),capture_output=True,timeout=30);p.check_returncode();(j/"probe.stdout").write_bytes(p.stdout);(j/"probe.stderr").write_bytes(p.stderr);data=json.loads(p.stdout.decode().splitlines()[-1]);assert not data["external"]["INFO_enabled"] and data["native_namespace"]["INFO_enabled"]
data["limits"]=["CPU logging configuration and existing missing install lines, not live worker-object introspection","Explicit Worker path/source and successful native init provide source-conditioned installation inference; no observed shape reduction"]
p=j/"reduction.json";p.write_text(json.dumps(data,indent=2));raw=p.read_bytes()
(j/"result.json").write_text(json.dumps({"schema_version":1,"job_id":job["job_id"],"status":"completed","summary":"Default logging filters external prototype INFO; vllm namespace INFO is enabled. No inference/worker/model created","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[{"kind":"fact","text":"CPU default logger coupled_dp_metadata suppressesINFO, vllm.glm_coupled_dp_metadata enablesINFO; Run28 missinginstall telemetry is not native rejection evidence","scope":{"CPU_only":True},"evidence_ids":["logging"]}],"evidence":[{"id":"logging","path":str(p),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"locator":"default native logging filter configurations"}],"unknowns":data["limits"],"decision_request":None,"next_check_at":None},indent=2))
print(json.dumps(data))
