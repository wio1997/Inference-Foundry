#!/usr/bin/env python3
"""Read-only TASK/PMU ID multiplicity for two saved EP8 MC2 records."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parents[2]
input_paths = [root / 'diagnostic-20261029/target-layer-mc2-16rank.json', root / 'agents/diagnosis/existing-mc2-pc-capture-audit.json']
expected_audit = '8363012053e686d5e06303194d039b3394f5f038796311b25ba77a03a323dfb6'
assert hashlib.sha256(input_paths[1].read_bytes()).hexdigest() == expected_audit
ranks, audit = [json.loads(p.read_text()) for p in input_paths]
ep8s = [r['rank'] for r in ranks['ranks'] if r['ep'] == 8]
assert len(ep8s) == 1
targets = [{k: x[k] for k in ('target', 'exact_global_task_id', 'end_ns', 'pmu_distinct_timestamp_ns')} for x in audit['ep8_targets']]
assert sorted(x['target']['layer'] for x in targets) == [14, 39]
base = '/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces'
code = r'''import json,pathlib,sqlite3,sys
base,rank,targets=sys.argv[1],sys.argv[2],json.loads(sys.argv[3])
db=pathlib.Path(base)/rank/'ASCEND_PROFILER_OUTPUT'/'ascend_pytorch_profiler_8.db'
con=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
res=[]
for t in targets:
 gid=t['exact_global_task_id']
 rows=con.execute('select globalTaskId,taskId,streamId,modelId,startNs,endNs from TASK where globalTaskId=? order by startNs,endNs',(gid,)).fetchall()
 pmu=con.execute('select timestampNs,count(*) from TASK_PMU_INFO where globalTaskId=? group by timestampNs order by timestampNs',(gid,)).fetchall()
 res.append({'target':t,'task_count_global_id':len(rows),'task_rows':[{'globalTaskId':r[0],'taskId':r[1],'streamId':r[2],'modelId':r[3],'startNs':r[4],'endNs':r[5]} for r in rows], 'pmu_timestamp_groups':[{'timestampNs':r[0],'count':r[1]} for r in pmu]})
print(json.dumps({'results':res, 'total_task_rows':con.execute('select count(*) from TASK').fetchone()[0]},ensure_ascii=False))'''
cmd = shlex.join(['docker', 'exec', 'wio-vllm-v41-repro169-stage-20261028', 'python3', '-c', code, base, ep8s[0], json.dumps(targets)])
r = subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh', cmd], check=True, capture_output=True, text=True, timeout=50)
out = json.loads(r.stdout)
out['input_sha256'] = [hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths]
out['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
out['scope'] = 'EP8 existing profiler database; read-only TASK and TASK_PMU_INFO; no NPU'
for x in out['results']:
    target=x['target']
    assert any(y['startNs'] == target['target']['start_ns'] and y['endNs'] == target['end_ns'] for y in x['task_rows'])
    print('layer',target['target']['layer'],'gid',target['exact_global_task_id'],'task_count',x['task_count_global_id'],'pmu_groups',x['pmu_timestamp_groups'])
p = root/'agents/diagnosis/mc2-ep8-globaltaskid-uniqueness.json'
p.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print('total_TASK',out['total_task_rows'],'output_sha256',hashlib.sha256(p.read_bytes()).hexdigest())
