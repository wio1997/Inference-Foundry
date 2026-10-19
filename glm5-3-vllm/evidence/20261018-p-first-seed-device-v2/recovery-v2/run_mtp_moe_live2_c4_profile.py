from pathlib import Path
import json
import subprocess
import sys
import time

repo=Path('/Users/wio/work/Inference-Foundry-execution-plan');r=repo/'glm5-3-vllm/evidence/20261018-p-first-seed-device-v2/recovery-v2'
start=time.monotonic()
while not (r/'mtp-moe-graph-live2-functional-outcome.json').exists():
    ready=r/'mtp-moe-graph-live2-startup-outcome.json'
    if ready.exists():assert json.loads(ready.read_text())['both_ready']
    assert time.monotonic()-start<1800;time.sleep(10)
assert json.loads((r/'mtp-moe-graph-live2-functional-outcome.json').read_text())['completed']
jobs=[('profile_mtp_moe_live2_c4.py','mtp_moe_live2_c4_profile_v1','mtp-moe-live2-c4-profile',600),
      ('export_mtp_moe_live2_c4.py','mtp_moe_live2_c4_export_v1','mtp-moe-live2-c4-export',1100),
      ('analyze_mtp_moe_live2_c4_copy.py','mtp_moe_live2_c4_copy_v1','mtp-moe-live2-c4-copy',600)]
for script,label,name,limit in jobs:
    q=subprocess.run([sys.executable,'glm5-3-vllm/tools/collect_node.py','--node','33','--script',str(r/script),'--label',label,'--output',str(r/name),'--control','/tmp/glm53-active-20261018.sock','--timeout',str(limit)],cwd=repo,capture_output=True,text=True,timeout=limit+180)
    (r/(name+'-collector.txt')).write_text(q.stdout+q.stderr);print(name,q.returncode,q.stdout,flush=True)
    assert q.returncode==0,q.stderr
    x=json.loads(next((r/name).glob('*/metrics.json')).read_text())
    assert x['completed']
    if name.endswith('copy'):
        assert x['all16_added_copy_SDMA']
        (r/'mtp-moe-graph-live2-physical-gate-v1.json').write_text(json.dumps({'completed':True,'all16_added_copy_SDMA':True,'ranks':[{k:z[k] for k in ['rank','candidate_input_copies','observed_selected_shapes','all_copy_chains_complete','all_added_copy_data_tasks_SDMA','data_task_types']} for z in x['ranks']],'scope':x['scope']},indent=2)+'\n')
        print('all16_C4_copy_SDMA_pass',flush=True)
