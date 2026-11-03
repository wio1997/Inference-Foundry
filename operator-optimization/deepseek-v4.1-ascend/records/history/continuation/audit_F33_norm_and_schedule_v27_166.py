from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib,json,socket

assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18')
p=M/'F33-C1-SWA-projection-order-and-consumer-surfaces-v27.json'
b=p.read_bytes();assert hashlib.sha256(b).hexdigest()=='da20a123d7cf2c113efbc93d82979e2a404e337319062819160ae60f26711321'
d=json.loads(b);out=M/'F33-norm-publication-and-overlap-audit-v27.json';assert not out.exists()

def one(rows,pred):
    vs=[v for v in rows if pred(v)]
    assert len(vs)==1,len(vs)
    return vs[0]

def overlap(a,b):
    a=a['TASK'];b=b['TASK']
    return max(0,min(a['endNs'],b['endNs'])-max(a['startNs'],b['startNs']))

def dist(xs):
    v=sorted(xs)
    return dict(min=v[0],median=(v[(len(v)-1)//2]+v[len(v)//2])/2,max=v[-1],positive=sum(x>0 for x in v),zero=sum(x==0 for x in v))

rows=[]
for r in d['graph_records']:
    tasks=r['block_compute'];s=r['SWA_projection'];c=r['C1_projection']
    ss=s['TASK']['streamId'];cs=c['TASK']['streamId'];assert ss!=cs
    norm_s=one(r['independent_512_norm_candidates'],lambda v:v['TASK']['streamId']==ss)
    norm_c=one(r['independent_512_norm_candidates'],lambda v:v['TASK']['streamId']==cs)
    index=one(tasks,lambda v:v['definition']['opType_text']=='MatMulV2'and v['definition']['inputShapes_text']=='"24,512;128,512"')
    rope_s=one(tasks,lambda v:v['definition']['opType_text']=='InplacePartialRotaryMul'and v['TASK']['streamId']==ss and v['definition']['inputShapes_text'].startswith('"24,1,1,512;'))
    rope_c=one(tasks,lambda v:v['definition']['opType_text']=='InplacePartialRotaryMul'and v['TASK']['streamId']==cs and v['definition']['inputShapes_text'].startswith('"24,1,1,512;'))
    scatters=[v for v in tasks if v['definition']['opType_text']=='ScatterNdUpdateSk']
    pub_s=one(scatters,lambda v:v['TASK']['streamId']==ss and v['definition']['outputShapes_text']=='"17306,128,512"')
    pub_c=one(scatters,lambda v:v['TASK']['streamId']==cs and v['definition']['outputShapes_text']=='"17306,128,512"')
    pub_index=one(scatters,lambda v:v['definition']['outputShapes_text']=='"17306,128,128"')
    pub_scale=one(scatters,lambda v:v['definition']['outputShapes_text']=='"17306,128,1"')
    qnorm=one(tasks,lambda v:v['definition']['opType_text']=='RmsNorm'and v['definition']['inputShapes_text']=='"3,1280;1280"')
    qb=one(tasks,lambda v:v['definition']['opType_text']=='QuantBatchMatmulV3'and v['definition']['outputShapes_text']=='"3,32768"')
    iq=one(tasks,lambda v:v['definition']['opType_text']=='_quantize_indexer_query_kernel')
    sequence=[s,norm_s,rope_s,pub_s,c,norm_c,index,pub_index,pub_scale,rope_c,pub_c]
    assert all(a['TASK']['endNs']<=z['TASK']['startNs'] for a,z in zip(sequence,sequence[1:]))
    rows.append(dict(database_index=r['database_index'],graph_ordinal=r['graph_ordinal'],
        role_evidence=r['role_evidence'],
        ordered_TASK_rowids=[v['TASK']['task_record_rowid'] for v in sequence],
        ordered_role_names=['SWA_projection','SWA_norm','SWA_RoPE','SWA_publish','C1_projection','C1_norm','index_key_projection','index_publish','index_scale_publish','C1_RoPE','long_publish'],
        overlap_ns=dict(SWA_projection_with_Q_norm=overlap(s,qnorm),SWA_norm_with_Q_B=overlap(norm_s,qb),C1_projection_with_index_query_quantization=overlap(c,iq)),
        C1_projection_after_SWA_publish_ns=c['TASK']['startNs']-pub_s['TASK']['endNs']))
native_roles=Counter();native_shapes=Counter();native_ambiguous=[]
for r in d['native_records']:
    role_keys=set()
    for a in r['APIs']:
        text='\n'.join(a['project_frames'])
        primitive=r['definition']['opType_text']
        context='DRAFT_PROPOSE_CALLER'if 'propose_draft_token_ids'in text else 'TARGET_MODEL_EXECUTE_CALLER'
        if 'compressor.py(71)'in text:role='C1_norm'if primitive=='RmsNorm'else'C1_projection'
        elif 'dsa_v41.py(475)'in text:role='SWA_projection'
        elif 'dsa_v41.py(487)'in text:role='SWA_norm'
        else:role='UNKNOWN'
        role_keys.add((context,role))
    if len(role_keys)!=1:native_ambiguous.append(dict(database_index=r['database_index'],TASK_rowid=r['TASK']['task_record_rowid'],roles=sorted(role_keys)))
    for context,role in role_keys:
        native_roles[(context,role)]+=1
        native_shapes[(context,role,r['definition']['inputShapes_text'],r['definition']['inputDataTypes_text'])]+=1
assert len(rows)==80 and not native_ambiguous
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),profile_source=dict(path=str(p),sha256=hashlib.sha256(b).hexdigest()),
    graphs_reviewed=80,source_stream_consistent_order_sequences=80,
    observed_overlap_ns={key:dist([r['overlap_ns'][key]for r in rows])for key in rows[0]['overlap_ns']},
    C1_projection_after_SWA_publish_ns=dist([r['C1_projection_after_SWA_publish_ns']for r in rows]),
    native_role_counts=[dict(context=k[0],role=k[1],TASKs=v)for k,v in native_roles.items()],
    native_role_shape_counts=[dict(context=k[0],role=k[1],inputShapes=k[2],inputDataTypes=k[3],TASKs=v)for k,v in native_shapes.items()],
    native_ambiguous_roles=native_ambiguous,
    caveats=['This is exact recorded TASK order with source/stream-consistent role labels;it is not tensor-pointer alias or a proof every order edge is semantically required.',
             'Actual C1 andSWA projections do not overlap,but movingC1 earlier may change other observed vector/Cube overlap.',
             'SWA andlong publication share typed output shapes,not proven physical tensor addresses.',
             'All native matching contexts stay split;do not assign every SWA operation to source layer20.'],
    no_new_source_DB_model_device_or_service_work=True,overall_complete=False)
result=dict(summary=summary,graph_records=rows)
blob=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();out.write_bytes(blob)
print(json.dumps(dict(summary=summary,full_result=str(out),result_sha256=hashlib.sha256(blob).hexdigest()),ensure_ascii=False))
