from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,socket

assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis')
M=P/'native-PMU-complete-v18';OUT=M/'Engram-exact-root-shape-and-parameter-role-audit-v30.json'
assert not OUT.exists()
def pin(path,sha):
    blob=path.read_bytes();assert hashlib.sha256(blob).hexdigest()==sha,str(path);return json.loads(blob)
d=pin(M/'Engram-allrank-native-graph-and-collective-scope-v30.json','f4daeb471f3bda1cec7e89a89b1ffe3791150f5e06b976fbcef4388f13d01f6f')
prep=pin(P/'prepare-field-origins-v12.json','c23b75f3fbd3ae3d3d7152f6882524e899bade53001ea2f1befa09ce943f091f')
roots={r['root_key']:r for r in prep['prepare_roots']};bydb=defaultdict(list);markers=defaultdict(list)
for r in roots.values():bydb[r['database_index']].append(r)
for b in prep['bindings']:markers[b['database_index']].append(b['marker'])
def links(di,a):
    direct=[r['root_key']for r in bydb[di]if r['root']['globalTid']==a['globalTid']and r['root']['startNs']<=a['startNs']and a['endNs']<=r['root']['endNs']]
    mm=[m for m in markers[di]if m['globalTid']==a['globalTid']and m['startNs']<=a['startNs']and a['endNs']<=m['endNs']]
    execute=sorted({r['root_key']for m in mm for r in bydb[di]if r['root']['globalTid']==m['globalTid']and m['startNs']<=r['root']['startNs']and r['root']['endNs']<=m['endNs']})
    return dict(direct_prepare_roots=direct,same_execute_marker_prepare_roots=execute)
def shape(c):
    return dict(op=c['opType_text'],input=c.get('inputShapes_text'),input_types=c.get('inputDataTypes_text'),output=c.get('outputShapes_text'),output_types=c.get('outputDataTypes_text'))
def census(rows):
    count=Counter(json.dumps(shape(x),sort_keys=True)for x in rows)
    return [dict(signature=json.loads(k),TASKs=v)for k,v in sorted(count.items())]
def frames(a):return '\n'.join(a['project_frames'])
apis=[]
for row in d['native_APIs']:
    a=row['API'];apis.append(dict(database_index=row['database_index'],api_rowid=a['api_rowid'],chain_key=a['chain_key'],**links(row['database_index'],a)))
am={(r['database_index'],r['api_rowid']):r for r in apis}
native=[];selected_roles=defaultdict(list)
for row in d['native_TASKs']:
    c=row['definition'];op=c['opType_text'];ff='\n'.join(frames(a)for a in row['source_APIs'])
    rr=sorted({k for a in row['source_APIs']for k in am[(row['database_index'],a['api_rowid'])]['same_execute_marker_prepare_roots']})
    role='other_preparation_or_gate'
    if op=='ZerosLike'and'embedding.py(343)'in ff:role='lookup_prezeros'
    elif op=='_engram_host_uva_gather_dequant_kernel':role='UVA_lookup'
    elif op in['_engram_req_index_kernel','_write_hash_cache_kernel','_hash_ids_kernel']:role=op
    elif op in['Cast','Mul']and'layer.py(53)'in ff:role='qk_parameter_preparation'
    elif op=='MatMulV2'and'common.py(64)'in ff:role='block32_rotation'
    elif op=='MatMulV2'and'layer.py(41)'in ff:role='WKV'
    x=dict(database_index=row['database_index'],TASK_rowid=row['TASK']['task_record_rowid'],API_rowids=[a['api_rowid']for a in row['source_APIs']],role=role,prepare_root_candidates=rr,signature=shape(c))
    native.append(x);selected_roles[role].append(c)
comms=[]
for row in d['collectives']:
    ll=[dict(API_rowid=s['API']['api_rowid'],**links(row['database_index'],s['API']))for s in row['source_candidates']]
    rr=sorted({k for x in ll for k in x['same_execute_marker_prepare_roots']})
    comms.append(dict(database_index=row['database_index'],operation=row['operation'],source_links=ll,prepare_root_candidates=rr))
segments=[];g_roles=Counter();g_shapes=defaultdict(list)
for s in d['graph_segments']:
    cc=[r['definition']for r in s['compute']]
    # Ordinal roles are accepted only after checking the corresponding typed shapes.
    assert len(cc)==38
    assert shape(cc[0])['op']=='MatMulV2'and'25600,6144'in cc[0]['inputShapes_text']
    for c in cc[1:3]:assert c['opType_text']=='Cast'and c['inputShapes_text'].strip('"')=='4,5120'and c['inputDataTypes_text']=='DT_BF16'and c['outputDataTypes_text']=='DT_FLOAT'
    assert cc[3]['opType_text']=='Mul'and cc[3]['inputShapes_text'].strip('"')=='4,5120;4,5120'and cc[3]['inputDataTypes_text']=='DT_FLOAT;DT_FLOAT'
    rot=[c for c in cc if c['opType_text']=='MatMulV2'and';32,32'in c['inputShapes_text']]
    assert len(rot)==1
    for role,cs in[('WKV',cc[:1]),('qk_parameter_preparation',cc[1:4]),('ordinary_gate_including_block_rotation',cc[4:])]:
        g_roles[role]+=len(cs);g_shapes[role]+=cs
    segments.append(dict(database_index=s['database_index'],graph_ordinal=s['graph_ordinal'],next_block=s['next_block'],prepare_root_keys=s['prepare_root_keys'],phase_records=s['phase_records'],WKV=shape(cc[0]),rotation=shape(rot[0]),TASK_rowids=[r['task']['task_record_rowid']for r in s['compute']]))
root_rows=[]
for key,r in roots.items():
    nn=[x for x in native if x['prepare_root_candidates']==[key]];co=[x for x in comms if x['prepare_root_candidates']==[key]];ss=[x for x in segments if x['prepare_root_keys']==[key]]
    root_rows.append(dict(root_key=key,database_index=r['database_index'],schedule=r.get('schedule'),metadata=r.get('metadata'),native_roles=dict(Counter(x['role']for x in nn)),native_TASKs=len(nn),TP_collectives=[dict(COMM_OP_rowid=x['operation']['COMM_OP_rowid'],count=x['operation']['count'],rankSize=x['operation']['rankSize'],dtype=x['operation']['dtype'])for x in co],graph_segments=len(ss)))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),native_TASKs=len(native),native_unbound_or_ambiguous=sum(len(x['prepare_root_candidates'])!=1 for x in native),native_roles={k:len(v)for k,v in selected_roles.items()},native_role_shapes={k:census(v)for k,v in selected_roles.items()if k!='other_preparation_or_gate'},source_bound_TP_collectives=len(comms),collectives_unbound_or_ambiguous=sum(len(x['prepare_root_candidates'])!=1 for x in comms),prepare_roots=len(root_rows),roots_with_hash=sum(r['native_roles'].get('_hash_ids_kernel',0)>0 for r in root_rows),root_role_regimes=[dict(native_roles=json.loads(k[0]),TP_counts=json.loads(k[1]),graph_segments=k[2],roots=v)for k,v in Counter((json.dumps(r['native_roles'],sort_keys=True),json.dumps(sorted(x['count']for x in r['TP_collectives'])),r['graph_segments'])for r in root_rows).items()],graph_segments=len(segments),graph_segments_unbound_or_ambiguous=sum(len(x['prepare_root_keys'])!=1 for x in segments),graph_roles=dict(g_roles),graph_role_shapes={k:census(v)for k,v in g_shapes.items()},all_jobs_cpu_only=True,overall_complete=False)
result=dict(summary=summary,source=dict(path=str(M/'Engram-allrank-native-graph-and-collective-scope-v30.json'),sha256='f4daeb471f3bda1cec7e89a89b1ffe3791150f5e06b976fbcef4388f13d01f6f'),native_API_root_links=apis,native_TASK_role_rows=native,collective_links=comms,graph_segment_roles=segments,prepare_root_roles=root_rows,limits=['Containment is same-thread execute association,notnearesttime;allcandidatekeysretained.','Graphrolesaretyped-shapeauditedandSOURCE_CONSISTENT,nottensorpointer/nativePythonstackproof.','Thisancestryselection isnotanautomaticreplacementforv20functionalPMUattribution.','Noactualdeadmask/historyhit/slotcontents/paddingvalues/alias/readinessgenerationinferencefromcounts.','Noelapsed/counter/bytecountisturnedintophysicalpeakorrequestgain.'])
blob=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(blob)
print(json.dumps(dict(summary=summary,full_result=str(OUT),result_sha256=hashlib.sha256(blob).hexdigest()),ensure_ascii=False))
