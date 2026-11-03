from pathlib import Path
from collections import Counter
import socket, json, hashlib

assert socket.gethostname() == 'bm-jn-zs-zone1-910C-64G-10-105'
p = Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031/ep3-hc-tail20ms-stack-evidence.json')
raw = p.read_bytes()
d = json.loads(raw)
out = dict(source_file=str(p), source_sha256=hashlib.sha256(raw).hexdigest(), database_sha256=d['database_sha256'], items=[], no_database_requery_or_device_work=True)
for x in d['items']:
    rows = []
    for r in x['task_rows']:
        if not r['kernel_name']:
            continue
        v = {k:r[k] for k in ['taskId','startNs','endNs','streamId','op_type','input_shapes','output_shapes','input_dtypes','output_dtypes']}
        v['api'] = []
        for a in r.get('api_candidates', []):
            relevant = [f for f in a['frames'] if any(t in f['frame'] for t in ['model.py(', 'model_runner', 'dspark', 'engram', 'hc_', 'layernorm', 'vocab_parallel', 'logits_processor'])]
            v['api'].append(dict(name=a['name'], callchainId=a['callchainId'], frames=relevant[:3]))
            if r['taskId'] in (41613, 45309):
                v['full_index_callchain'] = a['frames']
        rows.append(v)
    out['items'].append(dict(label=x['label'], interval_ns=x['interval_ns'], compute_rows=rows,
                            unique_API_count_hist=dict(Counter(len(r.get('api',[])) for r in rows))))
print(json.dumps(out, ensure_ascii=False))
