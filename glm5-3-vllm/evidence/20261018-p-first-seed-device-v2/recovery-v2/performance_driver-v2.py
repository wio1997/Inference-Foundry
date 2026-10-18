from pathlib import Path
import datetime,json,subprocess,time
r=Path('/Users/wio/work/Inference-Foundry-execution-plan');e=r/'glm5-3-vllm/evidence/20261018-p-first-seed-device-v2';rec=e/'recovery-v2';control='/tmp/glm53-active-20261018.sock'
end=time.monotonic()+180
while not list((e/'P-profile-all16-repaired').rglob('result.json')) or not list((rec/'perf-prepare33').rglob('metrics.json')):
 assert time.monotonic()<end;time.sleep(3)
profile=json.loads(next((e/'P-profile-all16-repaired').rglob('metrics.json')).read_text());assert profile['all16_D2H_PCIE_DMA'] is True
assert json.loads(next((rec/'perf-prepare32').rglob('metrics.json')).read_text())['all3_exact_before_timing']
assert json.loads(next((rec/'perf-prepare33').rglob('metrics.json')).read_text())['D_trace_writes_disabled']
# Confirm the CPU exporter and model/profile related weight/hash transfers have
# ended before the measured request batch starts. No raw backup runs in parallel.
ssh=['ssh','-S',control,'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34'];code="import json,subprocess\np=subprocess.run(['docker','inspect','glm53-seed-p-rank15-repair-v1','--format','{{json .State.Running}}'],capture_output=True,text=True)\nassert p.returncode!=0 or p.stdout.strip()=='false'\nprint(json.dumps({'exporter_running':False}))"
p=subprocess.run(ssh+['kubectl exec -i -n default sglang-node-access-32 -- nsenter -t 1 -m -- python3 -'],input=code,capture_output=True,text=True,timeout=30);assert p.returncode==0,p.stderr
(rec/'performance-no-export-interference-gate.json').write_text(json.dumps({'checked_client_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'profile_export_complete':True,'exporter_running':False,'D_trace_disabled':True,'P_export_log_overhead_included':True},indent=2)+'\n')
def batch(label,order):
 args=['--execute','--output','/data/wio/glm5-3-vllm-execution/20261018-state-tail/seed-live1-20261018/node32/recovery-v2/'+label,'--max-tokens','64','--order',order,'--cases-file','/data/wio/glm5-3-vllm-execution/20261018-state-tail/seed-live1-20261018/node32/performance-cases-v1/cases.json']
 cmd=['python3',str(r/'glm5-3-vllm/tools/collect_node.py'),'--node','32','--script',str(r/'glm5-3-vllm/runtime-patches/p-first-seed-v2/run_pairs.py'),'--label','seed_live2_'+label,'--control',control,'--output',str(rec/label),'--timeout','600']
 for x in args:cmd+=['--script-arg='+x]
 p=subprocess.run(cmd,capture_output=True,text=True,cwd=r,timeout=660);(rec/(label+'-collector.txt')).write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr;j=json.loads(next((rec/label).rglob('metrics.json')).read_text());assert j['completed'] and j['passed'];print(json.dumps({'label':label,'rows':len(j['rows']),'exact_pass':True}),flush=True)
batch('perf-c1-warmup','control,adopt,seed')
batch('perf-c1-formal',','.join(['control','adopt','seed','seed','adopt','control']*3))
