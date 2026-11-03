"""Reuse exact saved profile keys for context/Tile/draft-SP/query-KV windows."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,re,sqlite3
from finalize_origin_coverage_v9 import N,O,read,write

CACHE=N/'original-profile-cache-v53/analysis/native-execution-identity-v14'


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
    assert not (O/'REVIEW-CHECKPOINT-v130.json').exists()
    fronts=read(O/'draft-cache-join-native-frontiers-v113.json')['frontiers']
    population=read(O/'communication-native-operator-population-v66.json')
    role=read(O/'communication-role-contract-v16.json')
    keys={c['chain_key']for c in role['communication_chains']if c['semantic_role']=='ATTENTION_INPUT_SP_ALLGATHER'and c['source_path']in ['DRAFT_PROPOSE_SOURCE','EXECUTE_DUMMY_BATCH_MERGED_DRAFT']}
    bydb=defaultdict(list);gathers=defaultdict(list);main_gathers=defaultdict(list);seeds=defaultdict(set)
    for r in fronts:bydb[r['database_index']].append(r)
    for r in population['operations']:
        if r.get('chain_key')in keys:gathers[r['database_index']].append(r)
        if r['semantic_role']=='COMBINE_HIDDEN_COLUMN_LINEAR_ALLGATHER':main_gathers[r['database_index']].append(r)
    for a in read(O/'verify-metadata-profile-scope-v129.json')['selected_APIs']:
        if 'SEED_PUBLICATION'in a['roles']:seeds[a['database_index']].update(a['native_TASK_rowids'])
    previous={r['path']:r for r in read(O/'draft-cache-join-inputs-v113.json')['inputs']};dbrefs=[];records=[];used_comm=set();used_query=set();dummy_zero=set()
    rawrefs={r['database_index']:r for r in read(O/'host-launch-original-API-coverage-v126.json')['databases']}
    for di,ff in bydb.items():
        p=CACHE/f'db-{di:02d}.json';identity=pin(p);assert identity['sha256']==previous[str(p)]['sha256']
        data=read(p);ts={r['task_record_rowid']:r for r in data['TASK_records']};defs={r['globalTaskId']:r for r in data['COMPUTE_definitions']}
        apis={r['api_rowid']:r for r in data['linked_PYTORCH_APIs']}
        rawp=Path(rawrefs[di]['path']);con=sqlite3.connect(rawp.as_uri()+'?mode=ro&immutable=1',uri=True)
        zero_chains={r[0]for r in con.execute("SELECT DISTINCT c.id FROM PYTORCH_CALLCHAINS c JOIN STRING_IDS s ON s.id=c.stack WHERE s.value LIKE '%/attention/dsa_v41.py(698): forward'")}
        seed_tasks=sorted([ts[k]for k in seeds[di]],key=lambda r:r['startNs']);ff.sort(key=lambda r:r['draft_repeat']['startNs'])
        for i,f in enumerate(ff):
            lo=f['draft_repeat']['endNs'];hi=ff[i+1]['draft_repeat']['startNs']if i+1<len(ff)else 2**63
            gs=sorted([r for r in gathers[di]if lo<=r['startNs']and r['endNs']<=hi],key=lambda r:r['startNs'])
            seed=[r for r in seed_tasks if lo<=r['startNs']<hi];assert len(gs)==3 and len(seed)==1,(di,i,len(gs),len(seed))
            layers=[]
            for j,g in enumerate(gs):
                key=(di,g['COMM_OP_rowid']);assert key not in used_comm;used_comm.add(key)
                assert g['dtype']=='BFP16'and g['rankSize']==8 and g['count']%5120==0
                end=gs[j+1]['startNs']if j<2 else seed[0]['startNs'];kv=[]
                for t in ts.values():
                    d=defs.get(t['globalTaskId'],{})
                    if g['endNs']<=t['startNs']and t['endNs']<=end and d.get('opType_text')=='MatMulV2'and re.fullmatch(r'"\d+,5120;512,5120"',d.get('inputShapes_text','')):
                        kv.append((t,d))
                zero=[]
                if not kv:
                    assert f['source_context']=='CALLER_DUMMY',(di,i,j,'unexpected non-dummy missing KV')
                    for t0 in ts.values():
                        d0=defs.get(t0['globalTaskId'],{})
                        if g['endNs']<=t0['startNs']and t0['endNs']<=end and d0.get('opType_text')=='ZerosLike':
                            ids=[a for a in t0.get('native_linked_API_rowids',[])if a in apis and apis[a]['callchainId']in zero_chains]
                            if ids:zero.append(dict(TASK=t0,definition=d0,API_ids=ids,callchain_ids=[apis[a]['callchainId']for a in ids],source_frame='/attention/dsa_v41.py(698): forward'))
                    assert len(zero)==1,(di,i,j,'no positive None-metadata zero witness')
                    dummy_zero.add((di,zero[0]['TASK']['task_record_rowid']));query=None
                else:
                    assert len(kv)==1,(di,i,j,len(kv))
                    t,d=kv[0];qrows=int(d['inputShapes_text'].strip('"').split(',')[0]);assert qrows==f['query_global_rows']
                    assert d['inputDataTypes_text']=='DT_BF16;DT_BF16'and d['outputDataTypes_text']=='DT_BF16'
                    qkey=(di,t['task_record_rowid']);assert qkey not in used_query;used_query.add(qkey);query=dict(TASK=t,definition=d)
                layers.append(dict(layer_source_order_ordinal=j,context_cycle=f['context_cycles'][j],
                    draft_SP_allGather=g,query_KV=query,dummy_zero_output_witness=zero[0]if zero else None,
                    per_sender_hidden_rows_from_count=g['count']//5120,
                    uniform_gather_padded_rows_from_count=8*(g['count']//5120),
                    grade='EXACT_SAVED_KEYS_TYPED_ORDER_WINDOW;NOT_LAYER_INSTANCE_WEIGHT_CACHE_OR_REQUEST_BINDING'))
            mg=[]
            if f['prior_main']:
                m=f['prior_main'];mg=[r for r in main_gathers[di]if m['GEMM']['endNs']<=r['startNs']and r['endNs']<=m['gather_transpose']['startNs']]
                assert len(mg)==1 and mg[0]['count']==640*f['context_rows']and mg[0]['dtype']=='BFP16'and mg[0]['rankSize']==8
            records.append(dict(database_index=di,source_order_ordinal=i,context_rows=f['context_rows'],query_KV_rows=f['query_global_rows'],
                source_context=f['source_context'],draft_repeat=f['draft_repeat'],seed_TASK=seed[0],layers=layers,
                preceding_main_projection=f['prior_main'],retained_main_feature_allGather=mg[0]if mg else None,
                exact_tensor_request_cache_weight_identity_bound=False))
        con.close();dbrefs.append(dict(**identity,database_index=di,raw_database_sha256=data['database_sha256']))
    assert len(records)==104 and len(used_comm)==312 and len(used_query)==264 and len(dummy_zero)==48 and sum(r['retained_main_feature_allGather']is not None for r in records)==88
    regimes=Counter((r['context_rows'],r['query_KV_rows'],tuple(l['per_sender_hidden_rows_from_count']for l in r['layers']),r['source_context'])for r in records)
    write(O/'context-sp-profile-frontiers-v130.json',dict(records=records,databases=dbrefs,frontiers=104,context_cycles=312,draft_SP_collectives=312,
        query_KV_typed_TASKs=264,positive_None_metadata_zero_output_TASKs=48,retained_main_feature_gathers=88,
        regimes=[dict(context_rows=k[0],query_KV_rows=k[1],sender_rows=list(k[2]),source_context=k[3],frontiers=v)for k,v in regimes.items()],
        source_filter='Only DIRECT_NATIVE_FULLCHAIN draft/dummy SP calls;target eager and captured graph SP populations remain separate.',
        scope='Original task/COMM keys and GPU source-order windows,not formalrequest/layerinstance/pointer/weight/cache ownership orcausalbenefit.',
        no_new_native_promotions=True,no_remote_model_Torch_CANN_or_NPU_work=True,overall_complete=False))
    print(json.dumps(dict(status='DRAFT_CONTEXT_SP_SCHEMA_RECONCILED',frontiers=104,SP_collectives=312,query_KV_TASKs=264,positive_dummy_zero_TASKs=48,main_gathers_retained=88,regimes=[dict(context=k[0],query=k[1],local_rows=k[2],frontiers=v)for k,v in regimes.items()],new_device_work=False)))


if __name__=='__main__':main()
