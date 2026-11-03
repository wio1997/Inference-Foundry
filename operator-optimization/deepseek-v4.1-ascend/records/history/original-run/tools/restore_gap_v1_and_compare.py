"""Restore the v1 aggregate from its unchanged remote copy, compare v2."""
from pathlib import Path
import csv,json,shlex,subprocess
T=Path(__file__).resolve().parents[1]
R='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41'
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';C='wio-vllm-v41-repro169-stage-20261028'
for name in ['summary.json','rank_gaps.csv','task_families.csv']:
 code='from pathlib import Path;p=Path('+repr(R+'/inter-forward-gap-analysis/'+name)+');assert p.stat().st_size<2*1024*1024;print(p.read_text(),end="")'
 r=subprocess.run([SSH,shlex.join(['docker','exec',C,'python3','-c',code])],capture_output=True,timeout=90);assert r.returncode==0,r.stderr.decode()
 p=T/'inter-forward-gap-analysis'/name;p.parent.mkdir(exist_ok=True);assert not p.exists();p.write_bytes(r.stdout)
a=json.loads((T/'inter-forward-gap-analysis/summary.json').read_text());b=json.loads((T/'inter-forward-gap-analysis-v2/summary.json').read_text())
assert len(a['paired'])==len(b['paired'])
rows=[]
for x,y in zip(a['paired'],b['paired']):
 assert x['target_forward_ordinal']==y['target_forward_ordinal']
 rows.append({'forward_ordinal':x['target_forward_ordinal'],
              'dp0_task_union_change_us':y['dp0_known_device_task_union_us_mean']-x['dp0_known_device_task_union_us_mean'],
              'dp1_task_union_change_us':y['dp1_known_device_task_union_us_mean']-x['dp1_known_device_task_union_us_mean'],
              'gap_duration_unchanged':x['dp0_local_gap_us_mean']==y['dp0_local_gap_us_mean'] and x['dp1_local_gap_us_mean']==y['dp1_local_gap_us_mean']})
doc={'status':'V1_RESTORED_V2_BOUNDARY_REVIEWED','rows':rows,
     'local_file_correction':'Original v2 retrieval wrapper mistakenly saved the new files under the local v1 directory. The three v2 files were moved to their proper directory; unchanged v1 files were re-fetched from the existing authorized remote CPU-stage directory. The old wrapper is preserved under evidence.',
     'original_remote_trace_and_aggregates_modified':False}
(T/'evidence/inter-forward-gap-version-comparison.json').write_text(json.dumps(doc,indent=2)+'\n');print(json.dumps(doc,indent=2))
