"""Bind QB/query-RoPE/SWA frontiers and reuse precise C1/C2 old-latent chains."""
from pathlib import Path
from collections import defaultdict,Counter
import sqlite3,hashlib,json
from finalize_origin_coverage_v9 import O,read,write


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


SPECS=[('QB','/attention/dsa_v41.py',491,'QuantBatchMatmulV3'),('QUERY_ROPE','/attention/dsa_v41.py',374,'InplacePartialRotaryMul'),
    ('SWA_NORM','/attention/dsa_v41.py',487,'RmsNorm'),('SWA_ROPE','/attention/dsa_v41.py',488,'InplacePartialRotaryMul'),('SWA_STORE','/attention/dsa_v41.py',308,'ScatterNdUpdateSk')]


def main():
    assert not(O/'REVIEW-CHECKPOINT-v140.json').exists()
    old=read(O/'hc-sparse-eager-native-frontiers-v138.json');refs={r['database_index']:r for r in old['databases']};groups=defaultdict(list);out=[];pins=[]
    index={(r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']):r for r in read(O/'hc-index-query-weight-native-producers-v139.json')['records']}
    for r in old['records']:groups[r['database_index']].append(r)
    for di,rows in sorted(groups.items()):
        ref=refs[di];db=sqlite3.connect(Path(ref['raw_database']['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;ss={r['id']:r['value']for r in db.execute('SELECT *FROM STRING_IDS')};sidroles=defaultdict(set)
        for sid,v in ss.items():
            for role,path,line,op in SPECS:
                if path+f'({line}):'in v:sidroles[sid].add(role)
        fs=defaultdict(list);rids=defaultdict(set)
        for f in db.execute('SELECT *FROM PYTORCH_CALLCHAINS'):
            if f['stack']in sidroles:
                fs[f['id']].append(dict(f,value=ss[f['stack']]))
                for role in sidroles[f['stack']]:rids[role].add(f['id'])
        wanted=set().union(*rids.values());apis=[dict(a,name_text=ss[a['name']])for a in db.execute('SELECT rowid AS api_rowid,*FROM PYTORCH_API')if a['callchainId']in wanted]
        data=read(Path(ref['cached_native']['path']));defs={d['globalTaskId']:d for d in data['COMPUTE_definitions']};byapi=defaultdict(list)
        for t in data['TASK_records']:
            for aid in t.get('native_linked_API_rowids',[]):byapi[aid].append(t)
        for r in rows:
            p=r['eager_HC_SP_parent'];ag,rs=p['SP_allGather']['API'],p['output_reduceScatter']['API'];inside=[a for a in apis if a['globalTid']==ag['globalTid']and int(ag['endNs'])<=int(a['startNs'])<=int(a['endNs'])<=int(rs['startNs'])];st={}
            for role,path,line,op in SPECS:
                hits=[]
                for a in inside:
                    if a['callchainId']not in rids[role]:continue
                    for t in byapi[a['api_rowid']]:
                        d=defs.get(t['globalTaskId'])
                        if d and d['opType_text']==op:hits.append((a,t,d))
                assert len(hits)==1,(di,r['source_block_ordinal'],role,len(hits));a,t,d=hits[0]
                st[role]=dict(API=a,TASK=t,definition={k:d[k]for k in ['compute_record_rowid','globalTaskId','opType_text','inputShapes_text','outputShapes_text','inputDataTypes_text','outputDataTypes_text','blockNum','mixBlockNum','hf32Eligible']},source_frames=fs[a['callchainId']])
            key=di,r['forward_API_rowid'],r['source_block_ordinal'];ix=index.get(key);qb,qr,sw=st['QB']['TASK'],st['QUERY_ROPE']['TASK'],st['SWA_STORE']['TASK']
            out.append(dict(database_index=di,forward_API_rowid=r['forward_API_rowid'],source_block_ordinal=r['source_block_ordinal'],eager_HC_sparse_parent=r,stages=st,
                recorded_SWA_store_end_minus_QB_end_ns=sw['endNs']-qb['endNs'],recorded_positive_SWA_wait_window_ns=max(0,sw['endNs']-qb['endNs']),
                query_RoPE_TASK_starts_after_QB_and_SWA_store=qr['startNs']>=max(qb['endNs'],sw['endNs']),
                query_RoPE_before_IndexQ_TASK=(qr['endNs']<=ix['producers']['INDEX_Q']['TASK']['startNs'])if ix else None,
                actual_CS_alias_or_new_published_row_native_ABI_or_counterfactual_gain_bound=False))
        pins.append(dict(database_index=di,raw_database=pin(ref['raw_database']['path']),cached_native=pin(ref['cached_native']['path'])));db.close()
    assert len(out)==960
    latent=[]
    for name,typ,norm in [('hc-c1-native-consumers-v135.json','C1','C1_NORM'),('hc-c2-native-consumers-v136.json','C2','POOL_NORM')]:
        for r in read(O/name)['records']:
            c=r['consumers'];sel={k:c[k]for k in ['INDEX_WK','INDEX_NORM','INDEX_ROPE','INDEX_QUANT','INDEX_KEY_STORE','INDEX_SCALE_CAST','INDEX_SCALE_STORE','LONG_ROPE','LONG_STORE']};sel['LATENT_NORM']=c[norm]
            wk,lr,ks,ss=[c[k]['TASK']for k in ['INDEX_WK','LONG_ROPE','INDEX_KEY_STORE','INDEX_SCALE_STORE']]
            latent.append(dict(database_index=r['database_index'],forward_API_rowid=r['forward_API_rowid'],source_block_ordinal=r['source_block_ordinal'],mode=typ,
                reused_artifact=name,consumers=sel,IndexWK_end_before_LongRoPE=wk['endNs']<=lr['startNs'],
                recorded_LongRoPE_start_minus_IndexWK_end_ns=lr['startNs']-wk['endNs'],recorded_LongRoPE_after_both_index_stores=lr['startNs']>=max(ks['endNs'],ss['endNs']),
                source_grade_only_no_new_native_or_alias_promotion=True))
    assert len(latent)==96
    write(O/'hc-rope-latent-native-frontiers-v140.json',dict(query_records=out,latent_records=latent,databases=pins,query_frontiers=960,query_stage_TASK_bindings=4800,reused_latent_frontiers=96,reused_latent_TASK_bindings=960,
        positive_SWA_wait_windows=sum(r['recorded_positive_SWA_wait_window_ns']>0 for r in out),recorded_wait_window_is_not_removable_critical_time=True,
        modes=dict(Counter(r['mode']for r in latent)),current_query_RoPE_already_before_indexer=True,new_native_promotions=0,new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='HC_QUERY_ROPE_SWA_AND_OLD_LATENT_FRONTIERS_BOUND',query_frontiers=960,query_TASKs=4800,latent_frontiers=96,reused_latent_TASKs=960,positive_recorded_SWA_wait_windows=sum(r['recorded_positive_SWA_wait_window_ns']>0 for r in out),new_device_work=False)))


if __name__=='__main__':main()
