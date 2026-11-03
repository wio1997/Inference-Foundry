"""Saved first draft initialization / HC / embedding / attention publication frontiers."""
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json,sqlite3
from finalize_origin_coverage_v9 import N,O,read,write


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
    assert not (O/'REVIEW-CHECKPOINT-v132.json').exists()
    old=read(O/'context-sp-profile-frontiers-v130.json');frontiers=old['records']
    inits={(r['database_index'],r['stages'][0]['TASK_rowid']):r for r in read(O/'embedding-initialization-native-chains-v67.json')['draft_initializations']}
    comms=read(O/'communication-native-operator-population-v66.json')['operations'];records=[];refs=[]
    rawrefs={r['database_index']:r for r in read(O/'host-launch-original-API-coverage-v126.json')['databases']}
    for ref in old['databases']:
        di=ref['database_index'];data=read(Path(ref['path']));tasks=data['TASK_records'];defs={r['globalTaskId']:r for r in data['COMPUTE_definitions']};apis={r['api_rowid']:r for r in data['linked_PYTORCH_APIs']}
        raw=rawrefs[di];assert pin(raw['path'])['sha256']==raw['sha256']
        db=sqlite3.connect(Path(raw['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row
        strings={r['id']:r['value']for r in db.execute('SELECT *FROM STRING_IDS')};cidmap=defaultdict(set)
        for x in db.execute('SELECT *FROM CONNECTION_IDS'):cidmap[x['connectionId']].add(x['id'])
        def native(t):
            api=[apis[i]for i in t.get('native_linked_API_rowids',[])];cc={a['callchainId']for a in api if a['callchainId']is not None};frames=[]
            if cc:
                frames=[dict(x,frame=strings[x['stack']])for x in db.execute('SELECT *FROM PYTORCH_CALLCHAINS WHERE id IN ('+','.join('?'for _ in cc)+')',list(cc))]
            return dict(TASK=t,definition=defs[t['globalTaskId']],APIs=api,frames=frames)
        for f in [x for x in frontiers if x['database_index']==di]:
            tile=f['draft_repeat'];init=inits[(di,tile['TASK_rowid'])];g=f['layers'][0]['draft_SP_allGather'];end=f['layers'][1]['draft_SP_allGather']['startNs']
            between=[t for t in tasks if init['stages'][-1]['endNs']<=t['startNs']<=t['endNs']<=g['startNs'] and t['globalTaskId']in defs]
            hc=[t for t in between if defs[t['globalTaskId']]['opType_text']=='HcPre'];norm=[t for t in between if defs[t['globalTaskId']]['opType_text']=='RmsNorm'];assert len(hc)==len(norm)==1
            assert init['local_rows']*5120==g['count'] and hc[0]['endNs']<=norm[0]['startNs'] and norm[0]['endNs']<=g['startNs']
            qa=[t for t in tasks if g['endNs']<=t['startNs']<=t['endNs']<=end and t['globalTaskId']in defs and defs[t['globalTaskId']]['inputShapes_text'].endswith(';1280,5120"')];assert len(qa)<=1
            ar=[x for x in comms if x['database_index']==di and x['semantic_role']=='VOCAB_EMBEDDING_ALLREDUCE' and f['layers'][-1]['context_cycle']['endNs']<=x['startNs']<=x['endNs']<=tile['startNs']];assert len(ar)==1;ar=ar[0]
            assert ar['count']==init['N']*5120 and ar['rankSize']==8 and ar['dtype']=='BFP16'
            ids=cidmap[ar['connectionId']];aa=[dict(x)for x in db.execute('SELECT rowid AS api_rowid,*FROM PYTORCH_API WHERE connectionId IN ('+','.join('?'for _ in ids)+')',list(ids))];cc={x['callchainId']for x in aa if x['callchainId']is not None}
            af=[dict(x,frame=strings[x['stack']])for x in db.execute('SELECT *FROM PYTORCH_CALLCHAINS WHERE id IN ('+','.join('?'for _ in cc)+')',list(cc))]if cc else[]
            assert any('/dspark.py(249): forward'in x['frame']for x in af)
            kv=f['layers'][0]['query_KV'];zero=f['layers'][0]['dummy_zero_output_witness'];kfull=native(kv['TASK'])if kv else None
            if zero:domain='POSITIVE_METADATA_NONE_ZERO'
            elif qa:domain='POSITIVE_QUERY_QA_AND_GLOBAL_KV'
            else:
                assert any('/dsa_v41.py(354): _preprocess'in x['frame']for x in kfull['frames']);domain='POSITIVE_LOCAL_QUERY_EMPTY_GLOBAL_KV'
            records.append(dict(database_index=di,source_order_ordinal=f['source_order_ordinal'],source_context=f['source_context'],branch=domain,
                initialization=init,embedding_allReduce=ar,embedding_native_APIs=aa,embedding_native_frames=af,
                first_HcPre=native(hc[0]),first_RmsNorm=native(norm[0]),first_SP_allGather=g,
                first_query_QA=native(qa[0])if qa else None,first_global_KV=kfull,positive_dummy_zero=zero,
                first_attention_window_end=end,N_valid_from_Tile=init['N'],A_gather_padded=init['local_rows']*8,
                actual_SP_CP_row_map_tensor_alias_parameter_or_request_identity_bound=False,
                grade='EXACT_SAVED_TASK_COMM_KEYS_PLUS_SOURCE_ORDER_FRONTIER_AND_POSITIVE_CPU_BRANCH;NOT_TENSOR_EDGE'))
        refs.append(dict(**pin(ref['path']),database_index=di,raw_database=pin(raw['path'])));db.close()
    counts=Counter(x['branch']for x in records);assert counts=={'POSITIVE_METADATA_NONE_ZERO':16,'POSITIVE_QUERY_QA_AND_GLOBAL_KV':69,'POSITIVE_LOCAL_QUERY_EMPTY_GLOBAL_KV':19}
    assert len(records)==104
    write(O/'init-sp-native-frontiers-v132.json',dict(records=records,databases=refs,source_order_frontiers=104,branch_counts=dict(counts),
        observed_geometries=dict(Counter(str((r['N_valid_from_Tile'],r['A_gather_padded']))for r in records)),
        no_new_native_operand_or_graph_binding=True,new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='FIRST_DRAFT_INIT_SP_FRONTIERS_BOUND',frontiers=104,branch_counts=dict(counts),new_device_work=False)))


if __name__=='__main__':main()
