"""Link the two D2H operations per step; test the source-inferred phase labels."""
from pathlib import Path
from bisect import bisect_left, bisect_right
from collections import defaultdict
import argparse, hashlib, json, statistics

p = argparse.ArgumentParser(); p.add_argument('--mode', choices=['native', 'adaptive'], required=True); a = p.parse_args()
r = Path(__file__).resolve().parent
base = Path('/Users/wio/work/glm53-execution-evidence-backup/20261019-prefix-count-c4-supply-rank0/artifacts')/a.mode
path = base/'rank0/trace_view.json'; raw = json.loads(path.read_text()); events = raw['traceEvents'] if isinstance(raw, dict) else raw
events = [z for z in events if z.get('ph') == 'X']
dq = {(z['name'], z.get('args', {}).get('correlation_id')): z for z in events if z.get('cat') == 'dequeue'}
apis = defaultdict(list); tasks = defaultdict(list)
for z in events:
    if z['name'].startswith(('AscendCL@', 'Node@')): apis[z['tid']].append(z)
    if 'Task Type' in z.get('args', {}) and 'connection_id' in z['args']: tasks[z['args']['connection_id']].append(z)
for v in apis.values(): v.sort(key=lambda z: float(z['ts']))
stamps = {tid: [float(z['ts']) for z in v] for tid, v in apis.items()}
copies = []
for enq in sorted([z for z in events if z['name'] == 'Enqueue@acl_memcpy_device_to_host'], key=lambda z: float(z['ts'])):
    deq = dq[(enq['name'].replace('Enqueue@', 'Dequeue@'), enq['args']['correlation_id'])]
    lo = float(deq['ts']); hi = lo+float(deq['dur']); tid = deq['tid']; linked = []
    for api in apis[tid][bisect_left(stamps[tid], lo):bisect_right(stamps[tid], hi)]:
        linked.extend(z for z in tasks.get(api.get('args', {}).get('connection_id'), []) if z['args']['Task Type'] in ['PCIE_DMA_SQE', 'SDMA_SQE'])
    assert linked, enq
    copies.append(dict(enqueue_ts=float(enq['ts']), enqueue_us=enq['dur'], device_start=min(float(z['ts']) for z in linked), device_end=max(float(z['ts'])+float(z['dur']) for z in linked), tasks=[dict(name=z['name'], task_type=z['args']['Task Type'], ts=z['ts'], dur=z['dur']) for z in linked]))
replays = sorted([z for z in events if z['name'] == 'AscendCL@aclmdlRIExecuteAsync'], key=lambda z: float(z['ts']))
frames = sorted([z for z in json.loads((base/'metadata.json').read_text()) if z['rank'] == 0], key=lambda z: z['monotonic_ns'])
shapes = [g['tokens'] for z in frames if z['phase'] == 'execute_model' for g in z['graphs']]
assert len(shapes) == len(replays) and len(copies) == 2*len(replays)
mlas = sorted([z for z in events if z['name'] == 'vllm::mla_forward'], key=lambda z: float(z['ts']))
heads = sorted([z for z in events if z['name'] == 'vllm::unquantized_gemm'], key=lambda z: float(z['ts']))
assert len(mlas) == len(heads) == 5*len(replays)
rows = []
for index, replay in enumerate(replays):
    lo = float(replay['ts']); hi = float(replays[index+1]['ts']) if index+1 < len(replays) else float('inf')
    chosen = [z for z in copies if lo <= z['enqueue_ts'] < hi]; assert len(chosen) == 2
    early, late = chosen
    before_mtp = early['enqueue_ts'] < float(mlas[5*index]['ts'])
    after_mtp = late['enqueue_ts'] > float(heads[5*index+4]['ts'])+float(heads[5*index+4]['dur'])
    assert before_mtp and after_mtp
    rows.append(dict(index=index, N=shapes[index], early_count_copy=early, late_sample_copy=late,
                     late_device_end_minus_early_device_end_us=late['device_end']-early['device_end'], source_phase_order_checked=True))
values=[z['late_device_end_minus_early_device_end_us'] for z in rows]
result=dict(completed=True, mode=a.mode, source_trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), steps=rows,
            gap_us=dict(min=min(values), median=statistics.median(values), max=max(values)),
            scope='Per-step enqueue/dequeue/API/physical-task links plus matching five MTP cores validate pre-draft versus post-draft D2H phase order. The early operation is the existing accepted-count copy, not a hypothetical early token copy; its time difference is a diagnostic proxy, not a service gain or retirement-safety proof. Moving publication also needs original GPU source, pinned-input and KV retirement ownership preserved.')
(r/('prefix-count-output-fence-'+a.mode+'-v1.json')).write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(dict(mode=a.mode, steps=len(rows), gap_us=result['gap_us'], by_step=[dict(N=z['N'], gap_us=z['late_device_end_minus_early_device_end_us']) for z in rows]), indent=2))
