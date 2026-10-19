"""Link actual early sample, count and native late sample D2H in one shadow profile."""
from pathlib import Path
from bisect import bisect_left, bisect_right
from collections import defaultdict
import argparse, hashlib, json, statistics

r = Path(__file__).resolve().parent
base = Path('/Users/wio/work/glm53-execution-evidence-backup/20261019-early-sample-shadow-c4/artifacts/adaptive')
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
frames = sorted([z for z in json.loads((base/'metadata.json').read_text()) if z['rank'] == 0], key=lambda z: z['native_get_begin_ns'])
assert frames and all(z['bitwise_equal'] and z['native_event_kept'] and not z['early_publication'] for z in frames)
mlas = sorted([z for z in events if z['name'] == 'vllm::mla_forward'], key=lambda z: float(z['ts']))
heads = sorted([z for z in events if z['name'] == 'vllm::unquantized_gemm'], key=lambda z: float(z['ts']))
assert len(mlas) == len(heads) == 5*len(replays)
rows = []; metadata_index = 0
for index, replay in enumerate(replays):
    lo=float(replay['ts']); hi=float(replays[index+1]['ts']) if index+1<len(replays) else float('inf')
    chosen=[z for z in copies if lo<=z['enqueue_ts']<hi]
    assert len(chosen) in [2,3], (index,len(chosen))
    row=dict(index=index,D2H_copies=len(chosen))
    if len(chosen)==3:
        early,count,late=chosen
        frame=frames[metadata_index];metadata_index+=1
        assert early['enqueue_ts']<count['enqueue_ts']<float(mlas[5*index]['ts'])
        assert late['enqueue_ts']>float(heads[5*index+4]['ts'])+float(heads[5*index+4]['dur'])
        assert late['device_end']>=early['device_end']
        row.update(sample_output_shape=frame['shape'],request_ids=frame['req_ids'],
                   early_sample_copy=early,accepted_count_copy=count,native_late_sample_copy=late,
                   late_minus_early_sample_device_end_us=late['device_end']-early['device_end'],
                   bitwise_early_late_equal=frame['bitwise_equal'],source_phase_order_checked=True)
    else:
        row.update(scope='No selected width6 shadow in this step; original two-copy path retained.')
    rows.append(row)
assert metadata_index==len(frames) and metadata_index>0
values=[z['late_minus_early_sample_device_end_us'] for z in rows if 'late_minus_early_sample_device_end_us' in z]
result=dict(completed=True,source_trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            metadata_sha256=hashlib.sha256((base/'metadata.json').read_bytes()).hexdigest(),steps=rows,
            observed_shadow_steps=len(values),
            early_to_native_late_sample_device_end_us=dict(min=min(values),median=statistics.median(values),max=max(values)),
            scope='Actual read-only early sample D2H and original native late D2H, linked through enqueue/dequeue/API connection_id to physical tasks. Matching FIFO rank0 shadow metadata proves the two host buffers bitwise equal in these finite steps. Existing accepted-count copy remains a separate operation. Instrumented synchronized4xD8 fragment only; data readiness gap is not a service gain, all-rank critical path or safe KV/input retirement proof. Native late publication and InputBatch event remain unchanged.')
(r/'early-sample-shadow-output-fence-v1.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(completed=True,steps=len(rows),shadow_steps=len(values),device_gap_us=result['early_to_native_late_sample_device_end_us'])))
