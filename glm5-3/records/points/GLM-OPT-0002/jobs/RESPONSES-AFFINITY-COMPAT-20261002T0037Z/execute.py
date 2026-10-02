import pathlib,json,hashlib,subprocess
j=pathlib.Path(__file__).parent;job=json.loads((j/"job.json").read_text());repo=j.parents[4]
script="""import sys,unittest
sys.path[:0]=['/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime','/data/tiankuan/wio/Inference-Foundry/glm5-3/tests']
import response_affinity_gateway
sys.modules['replica_gateway']=response_affinity_gateway
sys.modules['persistent_coupled_gateway']=response_affinity_gateway
suite=unittest.defaultTestLoader.loadTestsFromNames(['test_response_affinity','test_persistent_gateway_wire','test_durable_coupled_group','test_coupled_fault_domain','test_replica_gateway','test_inband_errors','test_prefill_placement','test_stream_audit','test_protocol_receipts'])
result=unittest.TextTestRunner(verbosity=2).run(suite)
print(__import__('json').dumps({'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'native_inference_calls':0}))
raise SystemExit(0 if result.wasSuccessful() else 1)
"""
p=subprocess.run(["docker","exec","glm52-single","python3","-c",script],capture_output=True,timeout=180);(j/"tests.stdout").write_bytes(p.stdout);(j/"tests.stderr").write_bytes(p.stderr);assert p.returncode==0,p.stderr.decode()[-2000:]
d=json.loads(p.stdout.decode().splitlines()[-1]);assert d["failures"]==d["errors"]==0 and d["tests"]==48
sources=[]
for name in ["runtime/response_affinity.py","runtime/response_affinity_gateway.py","tests/test_response_affinity.py","tests/test_persistent_gateway_wire.py","runtime/persistent_coupled_placement.py","runtime/persistent_coupled_gateway.py","tests/test_durable_coupled_group.py","runtime/coupled_placement.py","runtime/coupled_gateway.py","tests/test_coupled_fault_domain.py","runtime/placement.py","runtime/replica_gateway.py"]:
 path=repo/name;raw=path.read_bytes();sources.append({"path":str(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
out={"valid":True,"cpu_tests":d,"source_identities":sources,"scope":"New Responses affinity factory exercises all42existing gateway/fault/wire/lease/protocol contracts with disabledfeature plus6new native39fixture owner contracts;0native model calls","native_error_replay":"actualRun32 native error payload with reconstructed SSE framing, CPU fixture only","limits":["No enabledstore/continuation/background nativeE2E; CPU fixture/transport contracts only","CPU journal durability only; no realnative E2E new gateway or OS/powerloss/filesystem-disaster proof","Logical draining leaves physical ranks alive; rebuild/recovery must be uniquecontroller-owned with new epoch"]}
r=j/"reduction.json";r.write_text(json.dumps(out,indent=2));raw=r.read_bytes()
result={"schema_version":1,"job_id":job["job_id"],"status":"completed","summary":"Responses affinity factory all48newandlegacy CPU contracts passed; no native inference","execution":{"inner_exit_code":0,"acceptance":"passed","processes":[]},"findings":[{"kind":"fact","text":"Actual process exits and file persistence verify conservative sameepoch quarantine; nativeRun32 error payload retained CPU fixture; no GPU inference","scope":{"cpu_only":True},"evidence_ids":["reduction"]}],"evidence":[{"id":"reduction","path":str(r),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"locator":"CPU contract count/source hashes/scope and limits"}],"unknowns":out["limits"],"decision_request":None,"next_check_at":None};(j/"result.json").write_text(json.dumps(result,indent=2));print(json.dumps(d))
