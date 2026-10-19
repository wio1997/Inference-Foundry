from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import time

repo=Path('/Users/wio/work/Inference-Foundry-execution-plan');r=repo/'glm5-3-vllm/evidence/20261018-p-first-seed-device-v2/recovery-v2'
start=time.monotonic()
while not (r/'mtp-moe-graph-profile-gate-v1.json').exists():
    ready=r/'mtp-moe-graph-startup-outcome.json'
    if ready.exists():assert json.loads(ready.read_text())['both_ready']
    assert time.monotonic()-start<2400;time.sleep(10)
assert json.loads((r/'mtp-moe-graph-profile-gate-v1.json').read_text())['all16_captured_D2D_SDMA']
assert json.loads((r/'mtp-moe-graph-functional-outcome.json').read_text())['completed']
for node in [32,33]:
    name='mtp-moe-graph-preflight-'+str(node)
    q=subprocess.run([sys.executable,'glm5-3-vllm/tools/collect_node.py','--node',str(node),'--script',str(r/'preflight_mtp_moe.py'),'--label','mtp_moe_graph_preflight_'+str(node)+'_v1','--output',str(r/name),'--control','/tmp/glm53-active-20261018.sock','--timeout','90']+(['--script-arg=--disable-D-trace'] if node==33 else []),cwd=repo,capture_output=True,text=True,timeout=240)
    (r/(name+'-collector.txt')).write_text(q.stdout+q.stderr);assert q.returncode==0,q.stdout+q.stderr
    x=json.loads(next((r/name).glob('*/metrics.json')).read_text());print(json.dumps({'phase':'preflight','node':node,'remaining_s':x['remaining_s']}),flush=True)
spec=importlib.util.spec_from_file_location('finite_mtp_functional',r/'run_mtp_moe_functional.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
c1=m.run('c1')
c4=m.collect(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/validate_sfa_replay_order.py','mtp_moe_graph_c4_balanced_v1','mtp-moe-graph-c4',('--execute','--phase','c4','--probe-kind','mtp_moe_graph','--run-dir',m.root,'--expected-D-CID',m.cid,'--cases',m.root+'/prefill-padding-cases-v1.json','--rotate-cases','--c4-cycles','3','--run-tag','balanced1'),timeout=1200)
assert c4['completed'] and c4['numeric_all_pass']
(r/'mtp-moe-graph-formal-outcome.json').write_text(json.dumps({'completed':True,'native_marker_restored':True,'finite_C1_C4_only':True,'formal_C1':c1['summary'],'formal_C4':c4['summary'],'no_general_quality_or_capacity_acceptance':True},indent=2)+'\n')
print(json.dumps({'phase':'formal_completed','C1':c1['summary'],'C4':c4['summary']}),flush=True)
