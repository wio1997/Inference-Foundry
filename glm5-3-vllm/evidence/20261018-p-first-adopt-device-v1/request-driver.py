from pathlib import Path
import json,subprocess,time
root=Path('/Users/wio/work/Inference-Foundry-execution-plan');e=root/'glm5-3-vllm/evidence/20261018-p-first-adopt-device-v1';outcome=e/'startup-outcome-v2.json';end=time.monotonic()+1000
while not outcome.exists():
 assert time.monotonic()<end,'observer missing';time.sleep(5)
assert json.loads(outcome.read_text())['status']=='both_ready',outcome.read_text()
base='/data/wio/glm5-3-vllm-execution/20261018-state-tail/tail-c1-live1-20261018/node32'
summary=[]
for label,maximum,order,cases in [('smoke',2,'control,adopt',False),('max1',1,'control,adopt',False),('full64',64,'control,adopt,adopt,control',False),('quality24',64,'control,adopt',True)]:
 c=['python3',str(root/'glm5-3-vllm/tools/collect_node.py'),'--node','32','--script',str(root/'glm5-3-vllm/runtime-patches/p-first-adopt-v1/run_pairs.py'),'--label','adopt_live1_'+label,'--control','/tmp/glm53-active-20261018.sock','--output',str(e/label),'--timeout','390','--script-arg=--execute','--script-arg=--output','--script-arg='+base+'/client-adopt-'+label+'-v1','--script-arg=--max-tokens','--script-arg='+str(maximum),'--script-arg=--order','--script-arg='+order]
 if cases:c+=['--script-arg=--cases-file','--script-arg='+base+'/adopt-cases-v1/cases.json']
 (e/(label+'-command.json')).write_text(json.dumps(c,indent=2)+'\n');r=subprocess.run(c,cwd=root,capture_output=True,text=True,timeout=450);(e/(label+'-collector.txt')).write_text(r.stdout+r.stderr)
 reports=list((e/label).glob('*/metrics.json'));report=json.loads(reports[0].read_text()) if reports else {}
 row={'batch':label,'collector_rc':r.returncode,'completed':report.get('completed'),'rows':len(report.get('rows',[])),'failure':report.get('failure')};summary.append(row);(e/'request-batches.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(row),flush=True)
 if not report.get('completed'):raise SystemExit(2)
