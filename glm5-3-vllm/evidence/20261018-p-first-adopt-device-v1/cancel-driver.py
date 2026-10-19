from pathlib import Path
import collections,json,subprocess,time
root=Path('/Users/wio/work/Inference-Foundry-execution-plan');e=root/'glm5-3-vllm/evidence/20261018-p-first-adopt-device-v1';end=time.monotonic()+700
while not list((e/'native-quality24').glob('*/metrics.json')):
 assert time.monotonic()<end;time.sleep(2)
native=json.loads(next((e/'native-quality24').glob('*/metrics.json')).read_text());assert native['completed'],native.get('failure')
pairs=json.loads(next((e/'quality24-complete').glob('*/metrics.json')).read_text());assert pairs['completed']
summary={'numeric_scores':{'PD_control':sum(r['numeric_pass'] for r in pairs['rows'] if r['mode']=='control'),'PD_adopt':sum(r['numeric_pass'] for r in pairs['rows'] if r['mode']=='adopt')},'exact_PD_control_adopt':sum(r['matches_native_control'] for r in pairs['rows'] if r['mode']=='adopt'),'total_cases':24,'full64_exact_gate_passed':pairs['passed'],'native_path_exact':{}}
for n in (32,33):
 rs=[r for r in native['rows'] if r['node']==n];summary['numeric_scores']['native_'+str(n)]=sum(r['numeric_pass'] for r in rs);summary['native_path_exact'][n]={m:sum(r['matches_PD_'+m] for r in rs) for m in ('control','adopt')}
summary['scope']='24 frozen cases; numeric oracle and full64 exact sequence are separate. No default adoption/production precision acceptance.'
(e/'quality-matrix-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
assert all(v==24 for v in summary['numeric_scores'].values()),'Numerical regression; inspect before next stage'
c=['python3',str(root/'glm5-3-vllm/tools/collect_node.py'),'--node','32','--script',str(root/'glm5-3-vllm/runtime-patches/p-first-adopt-v1/cancel_reuse.py'),'--label','adopt_cancel_reuse_v1','--control','/tmp/glm53-active-20261018.sock','--output',str(e/'cancel-reuse'),'--timeout','390']
r=subprocess.run(c,cwd=root,capture_output=True,text=True,timeout=450);(e/'cancel-reuse-collector.txt').write_text(r.stdout+r.stderr)
reports=list((e/'cancel-reuse').glob('*/metrics.json'));report=json.loads(reports[0].read_text()) if reports else {}
print(json.dumps({'collector_rc':r.returncode,'completed':report.get('completed'),'failure':report.get('failure'),'reuse_passes':sum(r.get('reuse_pass',False) for r in report.get('rows',[])),'abort_delta':report.get('abort_metric_delta')}),flush=True)
