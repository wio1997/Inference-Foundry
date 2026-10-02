import hashlib,json,subprocess,sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
r=Path(__file__).parent;formal=r/"formal";bench=formal/"benchmark";summary=json.loads((formal/"formal_summary.json").read_text());assert summary["valid"]
phase=json.loads((r/"formal.phase.json").read_text());assert phase["status"]=="succeeded" and phase["exit_code"]==0
accept=json.loads((bench/"full_acceptance.json").read_text());details=Path(accept["details_path"]);perf=details.with_name("gsm8k.csv");assert perf.exists()
argv=["python3","/data/tiankuan/wio/glm52-pd/deploy/scripts/analyze_slo.py","--perf_csv",str(perf),"--details_jsonl",str(details),"--expect_output_len","61440","--concurrency","2","--out_json",str(r/"slo.json"),"--out_md",str(r/"slo.md")]
p=subprocess.run(argv,capture_output=True,text=True);(r/"slo_analyzer.stdout").write_text(p.stdout);(r/"slo_analyzer.stderr").write_text(p.stderr);assert p.returncode==0
slo=json.loads((r/"slo.json").read_text());assert slo["all_requests_succeeded"] and slo["output_len_ok"]
wire=json.loads((bench/"full_wire_acceptance.json").read_text());assert wire["measurement_valid"] and wire["effective_output_tokens"]==245760
full=json.loads((bench/"full_execution.json").read_text());warm=json.loads((bench/"warmup_execution.json").read_text());assert full["exit_code"]==0 and warm["exit_code"]==0
wall=(datetime.fromisoformat(full["finished_at"])-datetime.fromisoformat(full["started_at"])).total_seconds()
for node,name in [("166","TP_run42_0.log"),("167","TP_run42_1.log")]:
 argv=["cat","/data/tiankuan/wio/glm52-pd/deploy/logs/"+name]
 if node=="167":argv=["ssh","-o","BatchMode=yes","root@172.16.10.167",__import__("shlex").join(argv)]
 result=subprocess.run(argv,capture_output=True);result.check_returncode();(r/name).write_bytes(result.stdout)
assert json.loads((bench/"warmup_wire_acceptance.json").read_text())["effective_output_tokens"]==1
artifacts=[]
for p in sorted(r.rglob("*")):
 if p.is_file() and p.name not in ["manifest.json","state.json","controller.log","finalize.log","finalize.phase.json","artifact_index.json"]:
  artifacts.append({"path":str(p),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
index=r/"artifact_index.json";atomic_json(index,artifacts)
reduction={"run_id":r.name,"kind":"formal_e2e","measurement_valid":True,"functional_acceptance":True,"verdict":"INCONCLUSIVE","requests":wire["requests"],"effective_output_tokens":245760,"full_cli_phase_wall_s":wall,"effective_tps_full_cli_phase":245760/wall,"slo":slo,"phase_exit_codes":{"warmup":0,"full":0,"formal":0},"limits":summary["limits"]}
atomic_json(r/"reduction.json",reduction)
m=json.loads((r/"manifest.json").read_text());m.update(status="completed",completed_at=utc(),valid=True,verdict="INCONCLUSIVE",results=reduction,artifact_index={"path":str(index),"bytes":index.stat().st_size,"sha256":hashlib.sha256(index.read_bytes()).hexdigest(),"entries":len(artifacts)},artifacts=[x for x in artifacts if Path(x["path"]).name in ["controller_spec.json","adopted_model_identities.json","identity.json","formal_events.json","benchmark_command.json","reset_P166.json","reset_D167.json","full_wire_acceptance.json","warmup_wire_acceptance.json","full_acceptance.json","warmup_acceptance.json","full_execution.json","warmup_execution.json","slo.json","formal_summary.json","router_trace.jsonl","final_placement.json"]]);atomic_json(r/"manifest.json",m)
(r/"summary.md").write_text("# "+r.name+"\n\nOne formal closed-concurrency2 round:4/4 native requests81932→61440,245760 effective outputs. Single native DP1TP32DCP1EP32/16384/.80/K5/FULL/atomicMQ control, exactRun42 twohost cohort retained; no model reload/signals. Resident cache explicitly observed without reset; one native prefix warmup required; native wire replay/usage/finish/DONE/lease and terminal idle validated. Full CLI TPS="+str(245760/wall)+", SLA all_pass="+str(slo["slo_all_pass"])+". Single round INCONCLUSIVE, no formal repeated KEEP/stable capacity/global bounds. Canonical dataset/4full requests/closedconcurrency2/AISBench timers retained; physical/scheduler topology/cache/globalMTP trajectory differs vsRun21, not isolated code gain. NearKV208768 planning-token prediction assumes common prefix sharing, actual highwater/preempt/counters needed; startupKV is not hardwarecapacity certificate. Raw source/commands/all attempts hash indexed on server.\n")
print(json.dumps({"valid":True,"full_cli_tps":245760/wall,"slo_all_pass":slo["slo_all_pass"]}))
