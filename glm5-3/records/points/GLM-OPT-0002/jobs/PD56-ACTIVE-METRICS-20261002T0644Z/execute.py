from pathlib import Path
import json,urllib.request,hashlib,re,sys
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0056";http=urllib.request.build_opener(urllib.request.ProxyHandler({}));owners=json.loads((r/"adopted_model_identities.json").read_text());rows={}
for key,o in owners.items():
 with http.open("http://172.16.10."+o["host"]+":"+str(o["port"])+"/metrics",timeout=10)as z:assert z.status==200;b=z.read()
 f=j/(key+".metrics");f.write_bytes(b)
 rows[key]=dict(raw=dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()),selected=[l for l in b.decode().splitlines()if not l.startswith("#")and any(t in l for t in["num_requests_running","num_requests_waiting","prompt_tokens_total","generation_tokens_total","request_success_total","kv_cache_usage_perc","num_preemptions_total","prefix_cache_hits_total","prefix_cache_queries_total","external_prefix_cache"])])
out=dict(at=utc(),controller_state=json.loads((r/"state.json").read_text()),native_metrics=rows,inference_attempts=json.loads((r/"attempts.json").read_text()),signals=0,new_requests=0,models=0,limits=["One readonly metrics sample, exported stats may lag request progression","GET metrics cannot establish terminal transfer or allshard lifetime/functional completion/capacity","Controller/raw ledgers authoritative, no intervention/replay"])
atomic_json(j/"reduction.json",out);b=(j/"reduction.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="Readonly actualRun56 fourAPI metrics duringfirstlongP request; no newrequest/signals/model changes",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="actualnativeHTTPmetrics/state/attempts")],unknowns=out["limits"],decision_request=None,next_check_at=None));print(json.dumps(rows))

