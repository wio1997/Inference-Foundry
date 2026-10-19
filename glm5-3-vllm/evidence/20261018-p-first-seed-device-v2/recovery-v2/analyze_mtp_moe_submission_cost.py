"""CPU-only reduction of preserved diagnostic trace; durations are not additive service costs."""
from pathlib import Path
from bisect import bisect_left,bisect_right
from collections import Counter,defaultdict
import hashlib,json,statistics
r=Path(__file__).resolve().parent
p=Path('/Users/wio/work/glm53-execution-evidence-backup/20261018-p-first-seed-device-v2/recovery-v2/mtp-moe-live1-profile/rank0-trace_view.json')
x=json.loads(p.read_text());events=x['traceEvents'] if isinstance(x,dict) else x
e=[z for z in events if z.get('ph')=='X'];ops=sorted((z for z in e if z.get('name')=='vllm::moe_forward_shared'),key=lambda z:float(z['ts']))
assert len(ops)==150
main=sorted((z for z in e if z.get('pid')==ops[0]['pid'] and z.get('tid')==ops[0]['tid']),key=lambda z:float(z['ts']));ts=[float(z['ts']) for z in main]
groups=defaultdict(list)
for op in ops:
 begin=float(op['ts']);end=begin+float(op['dur']);kids=main[bisect_left(ts,begin):bisect_right(ts,end)]
 mode='native' if any('GroupedMatmul' in z.get('name','') for z in kids) else 'graph_replay'
 groups[mode].append(op)
def stats(values):
 v=sorted(values);return {'n':len(v),'median_us':statistics.median(v),'min_us':min(v),'max_us':max(v),'sum_us':sum(v)}
proof=json.loads(next((r/'mtp-moe-copy-tasks-v2').glob('*/metrics.json')).read_text());rank=proof['ranks'][0];assert rank['all_added_copy_data_tasks_SDMA']
# Match same 50 candidate op timestamps using the retained exact input-copy proof.
# The full copy-chain file is in the SHA-verified artifact archive; metric only has a sample.
sample=rank['sample'][0]
copy_tasks=sample['data_tasks']
report={'completed':True,'source':{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'diagnostic':'six true P->D8 requests, four native including arm, two candidate; profiler enabled'},'opaque_MTP_core_CPU_submission':{mode:stats([float(z['dur']) for z in arr]) for mode,arr in groups.items()},'candidate_copy_sample_physical_data_task_us':[float(z['dur']) for z in copy_tasks],'sample_copy_task_types':[z['args']['Task Type'] for z in copy_tasks],'observed_core_calls':{mode:len(z) for mode,z in groups.items()},'interpretation':'Opaque core CPU duration measures host submission/dispatch, including asynchronous work submission; it is not isolated end-to-end device service time. The private graph removes native eager operator submissions. Physical copy sample duration is sub-microsecond but may overlap other streams; do not add host/device durations. Actual target/MTP calls and accept/reject schedules may differ across modes and require native counters; no accepted-token count is inferred from five draft calls.','unknown':['Unprofiled per-mode target-verification critical path','Per-request accepted speculative tokens','GPU-visible host-supply idle gaps attributable specifically to core rather than target/attention/KV/queueing','Maximum sustained SLO capacity'],'next':'Use completed observer-off C1/balanced C4 as performance decision; reuse this trace only as mechanism/cost support, not as throughput prediction.'}
(r/'mtp-moe-submission-cost-v1.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['opaque_MTP_core_CPU_submission','candidate_copy_sample_physical_data_task_us','observed_core_calls']}))
