from pathlib import Path
import subprocess,json,sys,hashlib
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;p=j.parents[1];new=p.parents[2]/"runtime/aisbench_slo.py";old=Path("/data/tiankuan/wio/glm52-pd/deploy/scripts/analyze_slo.py");contracts=[]
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
sources=[ref(old),ref(new)]
cases={}
for label,name in [("128","AUDIT-RUN128-20261003T0020Z"),("152","AUDIT-RUN152-20261003T0545Z")]:cases[label]=json.loads((p/"jobs"/name/"slo.json").read_text())
def run(label,script,case,concurrency,expected=None,code=0):
 out=j/(label+".json");args=["python3",str(script),"--perf_csv",case["perf_csv"],"--details_jsonl",case["details_jsonl"],"--expect_output_len","61440","--concurrency",str(concurrency),"--out_json",str(out)]
 if expected is not None:args+=["--expected_requests",str(expected)]
 z=subprocess.run(args,capture_output=True,timeout=30);(j/(label+".stdout")).write_bytes(z.stdout);(j/(label+".stderr")).write_bytes(z.stderr);assert z.returncode==code
 return json.loads(out.read_text())if code==0 else None
for label,concurrency in [("128",2),("152",4)]:
 a=run(label+"_old",old,cases[label],concurrency);b=run(label+"_newdefault",new,cases[label],concurrency);assert a==b;contracts.append("legacy"+label+"_default_json_identical_all6thresholds")
original=cases["152"];v=run("152_expected4",new,original,4,4);assert v["all_requests_succeeded"]and v["expected_requests"]==4and v["n_success"]==4and v["output_len_ok"]and not v["slo_all_pass"]and v["slo"]==original["slo"]and v["ttft_ms"]==original["ttft_ms"]and v["tpot_ms"]==original["tpot_ms"];contracts.append("actual4closedc4_countvalid_originalTTFTP50FAIL_unchangedlatencies")
w=run("152_expected8",new,original,4,8);assert w["expected_requests"]==8and not w["all_requests_succeeded"]and w["slo"]==v["slo"];contracts.append("declared8actual4_rejectcount_unchanged6thresholds")
for n in [0,-1]:
 run("invalid"+str(n),new,original,4,n,2);contracts.append("invalid_nonpositive_"+str(n)+"_argparse_exit2_beforeartifact")
assert [ref(old),ref(new)]==sources
out=dict(at=utc(),valid=True,contracts=contracts,sources=sources,native_inference=0,native_models_workers_signals=0,measurement_outputs=0,limits=["CPU observer regression only, reuse historicalperf/details/no actualinference/performancecredit","Same legacydefault2*concurrency, optionalactualpositivecount; six original latency thresholds/statistics unchanged"])
atomic_json(j/"reduction.json",out);atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="AISBenchSLOcount CPU6contracts/default128and152jsonidentical/explicit4correct/originalTTFTP50FAIL retained/no nativeoperations",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",**ref(j/"reduction.json"),locator="actualCPU legacy/default/explicitrequestcount")],unknowns=out["limits"],decision_request=None,next_check_at=None));print(json.dumps(dict(valid=True,contracts=len(contracts))))
