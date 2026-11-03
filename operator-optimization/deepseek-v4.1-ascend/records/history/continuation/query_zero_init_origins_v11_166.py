from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib, json, re, socket

assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
A=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/profile-readonly-20261031')
OUT=A/'ep3-vector-zero-initialization-origins-v11.json'
assert not OUT.exists()
p=A/'ep3-all-compute-origins-v9-v2.json';b=p.read_bytes()
assert hashlib.sha256(b).hexdigest()=='fdce699c6cf2fdf15b9ba85c5e9e5a5cd7893d2b34a0f69b0321e719a9ef6b9d'
x=json.loads(b);apis={a['api_rowid']:a for a in x['apis']}
records=[]
for t in x['tasks']:
    if t['op_type']!='ZerosLike' or not re.fullmatch(r'"?\d+"?',t['output_shapes'].strip()):continue
    aa=[]
    for aid in t['api_rowids']:
        a=apis[aid];fs=x['callchains'].get(str(a['callchainId']),[])
        aa.append(dict(**a,full_frames=[dict(stackDepth=f['stackDepth'],frame=x['frame_dictionary'][str(f['frame_id'])]) for f in fs]))
    records.append(dict(task=t,apis=aa))
assert len(records)<2000
by_leaf=Counter()
for r in records:
    for a in r['apis']:
        fs=[f['frame'] for f in a['full_frames'] if '/vllm_ascend/' in f['frame'] or '/vllm/vllm/' in f['frame']]
        by_leaf[fs[0] if fs else 'NO_PROJECT_FRAME']+=1
full=dict(created_utc=datetime.now(timezone.utc).isoformat(),records=records,summary=dict(records=len(records),linked_records=sum(bool(r['apis']) for r in records),source_leaf_counts=dict(by_leaf),no_new_source_read=True,no_inference_NPU_or_service_mutation=True),limits=['Only vector-output ZerosLike records in the pinned EP3 capture. Same type/shape is not a join to replay or proof of an identical field. Full original callchains retained.'],overall_complete=False)
payload=json.dumps(full,ensure_ascii=False,separators=(',',':'))+'\n';OUT.write_text(payload)
public=dict(summary=full['summary'],full_result=str(OUT),result_sha256=hashlib.sha256(payload.encode()).hexdigest(),linked_examples=[r for r in records if r['apis']][:3])
assert len(json.dumps(public).encode())<30000
print(json.dumps(public,ensure_ascii=False))
