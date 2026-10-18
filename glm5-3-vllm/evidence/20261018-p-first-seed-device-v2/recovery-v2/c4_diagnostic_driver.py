from pathlib import Path
import json,subprocess,time
r=Path('/Users/wio/work/Inference-Foundry-execution-plan');e=r/'glm5-3-vllm/evidence/20261018-p-first-seed-device-v2/recovery-v2';control='/tmp/glm53-active-20261018.sock';end=time.monotonic()+400
while not list((e/'first-proxy-formal').rglob('metrics.json')):
 assert time.monotonic()<end;time.sleep(3)
j=json.loads(next((e/'first-proxy-formal').rglob('metrics.json')).read_text());assert j['completed'] and all(v['exact_pass'] for v in j['rows'])
def collect(node,script,label,args=(),timeout=300):
 c=['python3',str(r/'glm5-3-vllm/tools/collect_node.py'),'--node',str(node),'--script',str(r/'glm5-3-vllm/runtime-patches/p-first-seed-v2'/script),'--label','seed_live2_'+label,'--output',str(e/label),'--control',control,'--timeout',str(timeout)]
 for x in args:c+=['--script-arg='+x]
 p=subprocess.run(c,capture_output=True,text=True,cwd=r,timeout=timeout+80);(e/(label+'-collector.txt')).write_text(p.stdout+p.stderr);print(json.dumps({'label':label,'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr}),flush=True);assert p.returncode==0,p.stderr
 return json.loads(next((e/label).rglob('metrics.json')).read_text())
collect(33,'enable_D_trace.py','c4-enable-trace',timeout=60)
base='/data/wio/glm5-3-vllm-execution/20261018-state-tail/seed-live1-20261018/node32/recovery-v2/'
for label,order in [('c4-control-repeat','control,control,control,control'),('c4-matched-pairs','control,seed,seed,control')]:
 collect(32,'run_c4_diagnostic.py',label,['--execute','--diagnostic','--max-tokens','64','--cases','/data/wio/glm5-3-vllm-execution/20261018-state-tail/seed-live1-20261018/node32/performance-cases-v1/cases.json','--output',base+label,'--order',order])
