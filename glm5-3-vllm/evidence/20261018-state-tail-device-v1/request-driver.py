from pathlib import Path
import json,subprocess,time
root=Path('/Users/wio/work/Inference-Foundry-execution-plan');e=root/'glm5-3-vllm/evidence/20261018-state-tail-device-v1'
outcome=e/'startup-outcome-v2.json';deadline=time.monotonic()+600
while not outcome.exists():
 assert time.monotonic()<deadline,'readiness observer did not complete'
 time.sleep(5)
r=json.loads(outcome.read_text());assert r['status']=='both_ready',r['status']
base='/data/wio/glm5-3-vllm-execution/20261018-state-tail/tail-c1-live1-20261018/node32'
c=['python3',str(root/'glm5-3-vllm/tools/collect_node.py'),'--node','32','--script',str(root/'glm5-3-vllm/runtime-patches/state-tail-probe-v1/run_requests.py'),'--label','tail_live1_native_requests','--control','/tmp/glm53-active-20261018.sock','--output',str(e/'requests'),'--timeout','390','--script-arg=--execute-functional','--script-arg=--p-url','--script-arg=http://172.16.12.32:29481','--script-arg=--d-url','--script-arg=http://172.16.12.33:29482','--script-arg=--output','--script-arg='+base+'/client-native-v1','--script-arg=--p-probe-dir','--script-arg='+base+'/state-tail-probe']
(e/'request-command.json').write_text(json.dumps(c,indent=2)+'\n');p=subprocess.run(c,cwd=root,capture_output=True,text=True,timeout=450)
(e/'request-collector.txt').write_text(p.stdout+p.stderr)
print(json.dumps({'collector_rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr}),flush=True)
reports=list((e/'requests').glob('*/metrics.json'))
if reports:
 d=json.loads(reports[0].read_text());print(json.dumps({'completed':d.get('completed'),'failure':d.get('failure'),'rows':[{'input_tokens':r['input_tokens'],'functional_pass':r.get('functional_pass'),'P_tail_precheck':r.get('P_tail_precheck')} for r in d['rows']]}),flush=True)
