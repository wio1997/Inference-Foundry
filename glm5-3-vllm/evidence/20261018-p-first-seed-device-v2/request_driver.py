from pathlib import Path
import json,subprocess,time
root=Path('/Users/wio/work/Inference-Foundry-execution-plan');e=root/'glm5-3-vllm/evidence/20261018-p-first-seed-device-v2';control='/tmp/glm53-active-20261018.sock'
end=time.monotonic()+1000
while not (e/'startup-outcome-v1.json').exists():
 assert time.monotonic()<end;time.sleep(3)
assert json.loads((e/'startup-outcome-v1.json').read_text())['status']=='both_ready'
base='/data/wio/glm5-3-vllm-execution/20261018-state-tail/seed-live1-20261018/node32'
def batch(label,max_tokens,order='control,adopt,seed',extra=()):
 args=['--execute','--output',base+'/'+label,'--max-tokens',str(max_tokens),'--order',order,*extra]
 cmd=['python3',str(root/'glm5-3-vllm/tools/collect_node.py'),'--node','32','--script',str(root/'glm5-3-vllm/runtime-patches/p-first-seed-v2/run_pairs.py'),'--label','seed_live1_'+label,'--control',control,'--output',str(e/label),'--timeout','600']
 for a in args:cmd+=['--script-arg='+a]
 p=subprocess.run(cmd,capture_output=True,text=True,cwd=root,timeout=660);(e/(label+'-collector.txt')).write_text(p.stdout+p.stderr)
 print(json.dumps({'batch':label,'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr}),flush=True)
 j=json.loads(next((e/label).rglob('metrics.json')).read_text());(e/(label+'-summary.json')).write_text(json.dumps({'completed':j.get('completed'),'passed':j.get('passed'),'failure':j.get('failure'),'rows':[{k:v for k,v in r.items() if k in ('input_tokens','mode','seed_packet_valid','matches_native_control','functional_pass','TTFT_ms','last_token_ms','numeric_pass')} for r in j['rows']]},indent=2)+'\n')
 assert p.returncode==0 and j['completed'] and j['passed'],j.get('failure')
batch('smoke2',2)
batch('max1',1)
batch('original64',64,'control,seed,seed,control')
batch('quality24',64,'control,adopt,seed', ['--cases-file','/data/wio/glm5-3-vllm-execution/20261018-state-tail/tail-c1-live1-20261018/node32/adopt-cases-v1/cases.json','--continue-mismatch'])
