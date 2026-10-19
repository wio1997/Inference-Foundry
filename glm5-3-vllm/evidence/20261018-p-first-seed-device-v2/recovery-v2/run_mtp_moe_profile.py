from pathlib import Path
import json
import subprocess
import sys
import time

repo=Path('/Users/wio/work/Inference-Foundry-execution-plan');r=repo/'glm5-3-vllm/evidence/20261018-p-first-seed-device-v2/recovery-v2'
start=time.monotonic()
while not (r/'mtp-moe-graph-functional-outcome.json').exists():
    assert time.monotonic()-start<1800;time.sleep(10)
assert json.loads((r/'mtp-moe-graph-functional-outcome.json').read_text())['completed']
for script,label,name,timeout in [
    ('profile_mtp_moe_graph.py','mtp_moe_graph_profile_v1','mtp-moe-graph-profile',420),
    ('export_mtp_moe_graph.py','mtp_moe_graph_export_v1','mtp-moe-graph-export',720),
]:
    q=subprocess.run([sys.executable,'glm5-3-vllm/tools/collect_node.py','--node','33','--script',str(r/script),'--label',label,'--output',str(r/name),'--control','/tmp/glm53-active-20261018.sock','--timeout',str(timeout)],cwd=repo,capture_output=True,text=True,timeout=timeout+180)
    (r/(name+'-collector.txt')).write_text(q.stdout+q.stderr)
    print(name,q.returncode,q.stdout,flush=True)
    assert q.returncode==0,q.stderr
    result=json.loads(next((r/name).glob('*/metrics.json')).read_text())
    assert result['completed']
    if name.endswith('export'):
        summary={'all16_captured_D2D_SDMA':result['all16_captured_D2D_SDMA'],
                 'ranks':[{k:x[k] for k in ['rank','D2D_count','D2D_device_task_types','all_captured_D2D_SDMA','rdma_named_X_events']} for x in result['ranks']],
                 'scope':result['scope'],'all16_raw_bytes':result['all16_raw_bytes'],'all16_raw_archive_sha256':result['all16_raw_archive_sha256']}
        (r/'mtp-moe-graph-profile-gate-v1.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary),flush=True)
