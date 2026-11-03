from pathlib import Path
import socket, json, hashlib
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
p=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031/ep3-dspark-and-moe-linked-tasks-v7-v2.json')
raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()=='e806fc67206b77a7d90d1d70f289eab84656e78fb813c1636509c901d51c4a98'
d=json.loads(raw);rows=[]
for kind in ['GatherV2','hcom_allReduce_']:
    for t in d['tasks']:
        fs=[f for a in t['api_candidates'] for f in d['callchains'][str(a['callchainId'])]]
        if t['op_type']==kind and any('dspark.py(249): forward' in f['frame'] for f in fs):
            rows.append(dict(taskId=t['taskId'],op_type=kind,input_shapes=t['input_shapes'],output_shapes=t['output_shapes'],
                             API=t['api_candidates'],retained_frames=fs));break
assert len(rows)==2
print(json.dumps(dict(source_sha256=hashlib.sha256(raw).hexdigest(),rows=rows,no_database_or_device_work=True),ensure_ascii=False))
