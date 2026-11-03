"""One indexed CPU-only pass over the existing selected operator opportunities."""
import collections
import hashlib
import json
import os
import sqlite3
import statistics
import time
from pathlib import Path

A = Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis')
P = A / 'joint169'
O = A / 'joint170'
O.mkdir(exist_ok=True)
symbols = {
    'AscendDSAV41MetadataBuilder.build',
    'AscendDSAV41MetadataBuilder.build.build_c2_metadata',
    'AscendDSAV41MetadataBuilder.build.build_smla_metadata',
    'compressed_slot_mapping',
    'AscendSpecDecodeBaseProposer._run_merged_draft', 'DSparkMarkovHead.embed',
    'DeepseekV41Model.embed_input_ids', 'DeepseekV41DSparkModel.forward',
    'AscendEngram.forward', 'engram_gate',
}
idx = json.loads((P / 'PROFILE-SOURCE-SYMBOL-INDEX.json').read_text())['source_frame_index']
catalog = json.loads((P / 'OPERATOR-CALLSITE-CATALOG.json').read_text())['catalog']
catalog_by_key = {(r['CPU_operator'], tuple(r['native_kernel_multiset']), json.dumps(r['source_anchor'], sort_keys=True)): i for i, r in enumerate(catalog, 1)}
state = {'state': 'RUNNING', 'pid': os.getpid(), 'start_ticks': Path('/proc/'+str(os.getpid())+'/stat').read_text().rsplit(')', 1)[1].split()[19], 'started_unix': time.time(), 'completed_profiles': 0, 'CPU_only': True}


def save_state():
    (O / 'STATUS.json').write_text(json.dumps(state, indent=2)+'\n')


def stat(v):
    return {'n': len(v), 'sum': sum(v), 'min': min(v), 'median': statistics.median(v), 'max': max(v)}


save_state()
groups = {}
profile_results = []
for path in sorted((P / 'startup-creators').glob('*_ascend_pt.json')):
    d = json.loads(path.read_text())
    calls = collections.defaultdict(list)
    for x in d['executions']:
        assert len(x['CPU_rows']) == 1
        calls[x['CPU_rows'][0]].append(x)
    selected = {}
    for rowid, ops in calls.items():
        cpu = d['CPU_rows'][str(rowid)]
        frames = [idx[f['text']] for f in d['callchains'][str(cpu['callchain_id'])] if f['text'] in idx]
        models = [f for f in frames if any(t in f['path'] for t in ['/models/', '/attention/', '/spec_decode/'])]
        anchor = models[0] if models else (frames[0] if frames else None)
        if anchor and anchor['enclosing_qualified_symbol'] in symbols:
            key = (cpu['name'], tuple(sorted(o['native_name'] for o in ops)), json.dumps(anchor, sort_keys=True))
            selected[rowid] = (cpu, anchor, ops, 'OG169-'+str(catalog_by_key[key]).zfill(3))
    c = sqlite3.connect('file:'+d['summary']['database']+'?mode=ro&immutable=1', uri=True)
    c.row_factory = sqlite3.Row
    s = dict(c.execute('select id,value from STRING_IDS'))
    task = {r['TASK_ROW']: dict(r) for r in c.execute('select rowid TASK_ROW,* from TASK')}
    info = {r['globalTaskId']: dict(r) for r in c.execute('select * from COMPUTE_TASK_INFO')}
    cpu_times = {}
    if selected:
        cpu_times = {r['CPU_ROW']: dict(r) for r in c.execute('select rowid CPU_ROW,startNs,endNs from PYTORCH_API where rowid in ('+','.join('?' for _ in selected)+')', list(selected))}
    seen = 0
    for rowid, (cpu, anchor, ops, gid) in selected.items():
        rs = sorted(ops, key=lambda x: (task[x['TASK_ROW']]['startNs'], x['TASK_ROW']))
        typed = []
        for op in rs:
            t = task[op['TASK_ROW']]
            x = info[t['globalTaskId']]
            fields = {k: s.get(x[k], x[k]) for k in ['inputDataTypes', 'outputDataTypes', 'inputShapes', 'outputShapes', 'inputFormats', 'outputFormats', 'taskType', 'opType']}
            typed.append(dict(native_name=op['native_name'], **fields))
        key = (gid, json.dumps(typed, sort_keys=True))
        if key not in groups:
            groups[key] = {'callsite_group': gid, 'CPU_operator': cpu['name'], 'source_anchor': anchor, 'native_sequence': typed, 'calls': 0, 'native_executions': 0, 'profiles': set(), 'CPU_duration_ns': [], 'native_work_ns': [], 'native_span_ns': [], 'per_sequence_kernel_duration_ns': [[] for _ in typed], 'distinct_globalTaskIds_per_call': collections.Counter(), 'same_stream_per_call': collections.Counter(), 'sample_calls': []}
        r = groups[key]
        durations = [task[o['TASK_ROW']]['endNs']-task[o['TASK_ROW']]['startNs'] for o in rs]
        r['calls'] += 1
        r['native_executions'] += len(rs)
        r['profiles'].add(path.stem)
        r['CPU_duration_ns'].append(int(cpu_times[rowid]['endNs'])-int(cpu_times[rowid]['startNs']))
        r['native_work_ns'].append(sum(durations))
        r['native_span_ns'].append(max(task[o['TASK_ROW']]['endNs'] for o in rs)-min(task[o['TASK_ROW']]['startNs'] for o in rs))
        for v, duration in zip(r['per_sequence_kernel_duration_ns'], durations):
            v.append(duration)
        r['distinct_globalTaskIds_per_call'][len({task[o['TASK_ROW']]['globalTaskId'] for o in rs})] += 1
        r['same_stream_per_call'][len({task[o['TASK_ROW']]['streamId'] for o in rs}) == 1] += 1
        if len(r['sample_calls']) < 2:
            r['sample_calls'].append({'profile': path.stem, 'CPU_ROW': rowid, 'CPU_duration_ns': int(cpu_times[rowid]['endNs'])-int(cpu_times[rowid]['startNs']), 'native': [{'TASK_ROW': o['TASK_ROW'], 'globalTaskId': task[o['TASK_ROW']]['globalTaskId'], 'stream': task[o['TASK_ROW']]['streamId'], 'taskType': s.get(task[o['TASK_ROW']]['taskType']), 'model': task[o['TASK_ROW']]['modelId'], 'startNs': task[o['TASK_ROW']]['startNs'], 'endNs': task[o['TASK_ROW']]['endNs']} for o in rs]})
        seen += 1
    profile_results.append({'profile': path.stem, 'database_sha256_reused': d['summary']['database_sha256'], 'selected_CPU_calls': seen})
    c.close()
    state['completed_profiles'] += 1
    save_state()
rows = []
for r in groups.values():
    r['profiles'] = sorted(r['profiles'])
    for field in ['CPU_duration_ns', 'native_work_ns', 'native_span_ns']:
        r[field] = stat(r[field])
    r['per_sequence_kernel_duration_ns'] = [stat(v) for v in r['per_sequence_kernel_duration_ns']]
    r['distinct_globalTaskIds_per_call'] = dict(r['distinct_globalTaskIds_per_call'])
    r['same_stream_per_call'] = {str(k): v for k, v in r['same_stream_per_call'].items()}
    rows.append(r)
result = {'status': 'SELECTED_ACTUAL_OPERATOR_TYPED_FRONTIER_RETURNED', 'profiles': profile_results, 'selected_source_symbols': sorted(symbols), 'callsite_groups': len({r['callsite_group'] for r in rows}), 'typed_regimes': len(rows), 'public_CPU_calls': sum(r['calls'] for r in rows), 'native_executions': sum(r['native_executions'] for r in rows), 'regimes': sorted(rows, key=lambda r: -r['native_work_ns']['sum']), 'scope': 'Existing startup dummy/warmup/creation domain only. Public calls are associated by exact native connection. Native work/span and CPU times are not formalC8 critical path or production utilization. No target source reread, no device work, no new capture.', 'overall_complete': False}
target = O / 'SELECTED-OPERATOR-TYPED-FRONTIER.json'
target.write_text(json.dumps(result, indent=2)+'\n')
state.update(state='RETURNED', elapsed_seconds=time.time()-state['started_unix'], result=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest())
save_state()
print(json.dumps({k: v for k, v in result.items() if k not in ['profiles', 'regimes']}))
