from pathlib import Path
from collections import Counter
import json, socket, hashlib

assert socket.gethostname() == 'bm-jn-zs-zone1-910C-64G-10-105'
p = Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031/ep3-hc-boundary-task-stack-evidence.json')
before = p.stat()
raw = p.read_bytes()
d = json.loads(raw)
assert d['database_sha256'] == '084b8718888e33271435ec79d7573c95ca1615f3e7484c807ea24590fad234c4' and d['database_stat_unchanged']


def short(r):
    vals = {k:r[k] for k in ['taskId','startNs','endNs','streamId','op_type','input_shapes','output_shapes','input_dtypes','output_dtypes']}
    vals['api'] = []
    for a in r.get('api_candidates', []):
        relevant = [x for x in a['frames'] if any(s in x['frame'] for s in ['model.py(', 'dspark', 'engram', 'hc_', 'layernorm', 'vocab_parallel', 'logits_processor'])]
        vals['api'].append(dict(name=a['name'], callchainId=a['callchainId'], source_frames=relevant[:3], source_frame_count=len(relevant)))
    return vals


summary = []
display = []
for x in d['items']:
    named = [r for r in x['task_rows'] if r['kernel_name']]
    summary.append(dict(label=x['label'], record_count=x['record_count'], compute_records=x['compute_records'],
                        compute_API_count_hist=dict(Counter(len(r.get('api_candidates',[])) for r in named)),
                        communication_ops=x['communication_ops'], op_type_counts=x['op_type_counts']))
    show = x['label'] == 'CROSS_db852bff_L0_L1' or x['label'].startswith(('LEADING_', 'TAIL_')) or '_L36_L37' in x['label']
    if show:
        display.append(dict(label=x['label'], interval_ns=x['interval_ns'], compute_rows=[short(r) for r in named]))
    if x['label'].startswith('INTRA_'):
        display.append(dict(label=x['label'], interval_ns=x['interval_ns'], anchor_rows=[short(r) for r in x['anchor_rows']]))
after = p.stat()
assert before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
result = dict(source_file=str(p), source_file_sha256=hashlib.sha256(raw).hexdigest(), database_sha256=d['database_sha256'],
              source_file_stat_unchanged=True, source_created_utc=d['created_utc'], summaries=summary, display=display,
              original_query_completed_despite_local_controller_timeout=True, no_database_requery_or_device_work=True)
print(json.dumps(result, ensure_ascii=False))
