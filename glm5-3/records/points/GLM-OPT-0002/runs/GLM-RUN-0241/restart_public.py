from common import *
old=p/"runs/GLM-RUN-0240";proof=json.loads((old/"restored/public_service_proof.json").read_text())
pending=json.loads((r/"pending_response.json").read_text());assert pending["status"]in["queued","in_progress"]
with http.open("http://172.16.10.167:9900/metrics",timeout=10)as response:raw=response.read()
(r/"restart_native_live.metrics").write_bytes(raw);v=re.findall(r"^vllm:num_requests_running(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",raw.decode(),re.M);assert v and sum(float(x)for x in v)>0
snap=request("/control/replicas");assert all(not x["active_requests"]for x in snap["replicas"])
retire(proof,old/"restored/public.gateway.log","retire240_background")
for _ in range(30):
 guard()
 if(old/"restored/identity_observer/terminal.json").exists():break
 time.sleep(1)
else:raise RuntimeError("old public observer did not end")
start_public(r/"restored","public241")
m=json.loads((r/"restored/native_engines_resident.json").read_text())
for e in m["engines"]:
 for key,o in e["roots"].items():fresh(o,e["members"][key],"restart_"+e["replica_id"])
atomic_json(r/"restart_summary.json",dict(at=utc(),native_background_live_before=True,HTTP_leases0=True,old_public_clean_SDK0=True,new_public_V14=True,native32_same=True,native_model_signals=0,STORE_replication=False))
