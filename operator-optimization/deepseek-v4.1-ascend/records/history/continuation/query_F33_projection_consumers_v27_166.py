from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib, json, socket

assert socket.gethostname() == 'bm-jn-zs-zone1-910C-64G-10-105'
P = Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis')
E=P/'native-execution-identity-v14'; C=P/'native-communications-v15'; M=P/'native-PMU-complete-v18'
OUT=M/'F33-C1-SWA-projection-order-and-consumer-surfaces-v27.json'
assert not OUT.exists()

def pin(p, sha):
    b=p.read_bytes(); assert hashlib.sha256(b).hexdigest()==sha, str(p)
    return json.loads(b)

roles=pin(C/'communication-roles-and-projection-streams-v16.json','bb5db68a578e4802304450f01c90460fe33f590fab850d284ef4a7b80210a2b1')
gr=pin(E/'graph-blocks-PMU-and-MC2-v14.json','a035c4de08191656141ad0727e8305a6444cc77e6184d4951f86136021af0991')
ix=pin(E/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4')
fs=pin(E/'native-compute-fullchains-v14-v2.json','6c7bf4bd3a7dcf557327aeadc4a37ec4b9b23cb1ba675353bbd6e5d56a4e1ca0')
comm=pin(C/'communication-and-queue-review-v15.json','c196d5083904a75a794c81fadfaf9d528d11461fdd5d81a6e941cad1e9693d6c')
gmap={(g['database_index'],g['graph_ordinal']):g for g in gr['graph_executions']}
bydb=defaultdict(list)
for r in roles['projection_streams']: bydb[r['database_index']].append(r)
api_bydb={d['database_index']:d['API_records'] for d in fs['databases']}
records=[]; native=[]; chains={}
for ref in ix['databases']:
    di=ref['database_index']; d=pin(Path(ref['path']),ref['sha256'])
    tm={t['task_record_rowid']:t for t in d['TASK_records']}
    defs={c['globalTaskId']:c for c in d['COMPUTE_definitions']}
    def wrap(t):
        return dict(TASK=t,definition=defs.get(t['globalTaskId']))
    selected=[]
    for a in api_bydb[di]:
        text='\n'.join(a.get('project_frames',[]))
        if 'dsa_v41.py' in text and ('compressor.py(71)' in text or 'dsa_v41.py(475)' in text or 'dsa_v41.py(487)' in text):
            selected.append(a)
            if a.get('chain_key'): chains[a['chain_key']]=fs['canonical_fullchains'][a['chain_key']]
    aa={a['api_rowid']:a for a in selected}
    for t in d['TASK_records']:
        keys=sorted(set(t.get('native_linked_API_rowids',[])) & aa.keys())
        if keys: native.append(dict(database_index=di,**wrap(t),APIs=[aa[k] for k in keys]))
    for r in bydb[di]:
        g=gmap[(di,r['graph_ordinal'])]
        ab=r['ambiguous_blocks']; assert len(ab)==1 and ab[0]['block_ordinal']==20
        ab=ab[0]; candidates=ab['candidates']['KV']; assert len(candidates)==2
        swa=ab['candidates_on_unique_unambiguous_KV_stream']; assert len(swa)==1
        swa_id=swa[0]['task_record_rowid']
        c1_id=next(t['task_record_rowid'] for t in candidates if t['task_record_rowid']!=swa_id)
        ts,tc=tm[swa_id],tm[c1_id]
        b=next(b for b in g['blocks'] if b['block_ordinal']==20)
        tasks=sorted([tm[k] for k in b['task_record_rowids']],key=lambda t:(t['startNs'],t['task_record_rowid']))
        compute=[wrap(t) for t in tasks if t['globalTaskId'] in defs]
        norms=[v for v in compute if 'rmsnorm' in v['definition']['opType_text'].lower() and '24,512' in v['definition']['inputShapes_text']]
        block_comm=[v for v in comm['communication_operations'] if v['database_index']==di and len(v['graph_candidates'])==1 and v['graph_candidates'][0]['graph_ordinal']==g['graph_ordinal'] and 20 in v['contained_marker_block_ordinals']]
        overlap=max(0,min(ts['endNs'],tc['endNs'])-max(ts['startNs'],tc['startNs']))
        records.append(dict(database_index=di,graph_ordinal=g['graph_ordinal'],prepare_root_keys=g['prepare_root_keys'],phase_records=g['phase_records'],block_ordinal=20,
            role_evidence='SOURCE_STREAM_CONSISTENT;NOT_TENSOR_POINTER_PROOF',
            SWA_projection=wrap(ts),C1_projection=wrap(tc),
            projection_overlap_ns=overlap,C1_start_minus_SWA_end_ns=tc['startNs']-ts['endNs'],
            independent_512_norm_candidates=norms,block_compute=compute,block_communication=block_comm,
            all_block_TASK_type_counts=dict(Counter(t.get('task_type_name','UNNAMED') for t in tasks))))

assert len(records)==80
def distribution(values):
    v=sorted(values)
    return dict(min=v[0],median=(v[(len(v)-1)//2]+v[len(v)//2])/2,max=v[-1],zero=sum(x==0 for x in v),positive=sum(x>0 for x in v),negative=sum(x<0 for x in v))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),graph_pairs=len(records),
    source_stream_consistent_pairs=80,
    projection_overlap_ns=distribution([r['projection_overlap_ns'] for r in records]),
    C1_start_minus_SWA_end_ns=distribution([r['C1_start_minus_SWA_end_ns'] for r in records]),
    independent_512_norm_candidate_counts=dict(Counter(len(r['independent_512_norm_candidates']) for r in records)),
    block20_compute_types=dict(Counter(v['definition']['opType_text'] for r in records for v in r['block_compute'])),
    native_selected_TASKs=len(native),native_selected_fullchains=len(chains),
    native_selected_op_types=dict(Counter(r['definition']['opType_text'] for r in native if r['definition'])),
    no_new_source_DB_model_device_or_service_work=True,overall_complete=False,
    limits=['Graphroles are constrained by saved source/stream evidence,not native Python stacks or pointer identities.',
            'Interval intersections and gaps describe the recorded schedule only,not causal dependencies or removable request latency.',
            'Typed norm/consumer candidates are retained;no nearest-time assignment or tensor alias invented.',
            'Native caller records include all matching contexts;they are not declared layer20 merely by dtype or shape.'])
result=dict(summary=summary,graph_records=records,native_records=native,native_canonical_chains=chains)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
sample=records[0]
public=dict(summary=summary,first_graph=dict(database_index=sample['database_index'],graph_ordinal=sample['graph_ordinal'],
    projections=[{k:v for k,v in sample[role].items()} for role in ['SWA_projection','C1_projection']],
    norms=sample['independent_512_norm_candidates'],
    compute_sequence=[dict(task_record_rowid=v['TASK']['task_record_rowid'],streamId=v['TASK']['streamId'],startNs=v['TASK']['startNs'],endNs=v['TASK']['endNs'],
        opType=v['definition']['opType_text'],name=v['definition']['name_text'],inputShapes=v['definition']['inputShapes_text'],inputDtypes=v['definition']['inputDataTypes_text'],outputShapes=v['definition']['outputShapes_text']) for v in sample['block_compute']],
    communication=sample['block_communication']),
    native_chain_headers=[dict(key=k,frames=[f['frame'] for f in v if '/vllm' in f['frame']]) for k,v in chains.items()],
    full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest())
print(json.dumps(public,ensure_ascii=False))
