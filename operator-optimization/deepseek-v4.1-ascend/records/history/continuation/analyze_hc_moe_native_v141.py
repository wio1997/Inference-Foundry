"""Join eager FFN router/MC2/shared/merge to HC and exact next-layer guards."""
from pathlib import Path
from collections import defaultdict,Counter
from bisect import bisect_right
import hashlib,json,sqlite3
from finalize_origin_coverage_v9 import O,read,write


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


SPECS=[('FFN_NORM','/models/deepseek_v41/model.py',846,['RmsNormCast']),('ROUTER','/ops/fused_moe/fused_moe.py',408,['MatMulV2','MatMulV3']),
    ('TOPK','/ops/fused_moe/router/fused_topk_router.py',204,['MoeGatingTopKHash']),('MC2','/ops/fused_moe/moe_comm_method.py',591,['DispatchFFNCombine']),
    ('SHARED_QUANT','/ops/fused_moe/shared_experts.py',429,['DynamicQuant']),('SHARED_UP','/ops/fused_moe/shared_experts.py',475,['QuantBatchMatmulV3']),
    ('SHARED_ACT','/ops/fused_moe/shared_experts.py',534,['DequantSwigluQuant']),('SHARED_DOWN','/ops/fused_moe/shared_experts.py',557,['QuantBatchMatmulV3']),
    ('MERGE','/ops/fused_moe/fused_moe.py',240,['ForeachAddList'])]


def main():
    assert not(O/'REVIEW-CHECKPOINT-v141.json').exists()
    p=read(O/'hc-sp-eager-native-frontiers-v134.json');groups=defaultdict(list);refs={r['database_index']:r for r in p['databases']};out=[];bridges=[];pins=[]
    for r in p['records']:groups[r['database_index']].append(r)
    for di,rows in sorted(groups.items()):
        ref=refs[di];db=sqlite3.connect(Path(ref['raw_database']['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;ss={x['id']:x['value']for x in db.execute('SELECT *FROM STRING_IDS')};sidroles=defaultdict(set)
        for sid,v in ss.items():
            for role,path,line,ops in SPECS:
                if path+f'({line}):'in v:sidroles[sid].add(role)
        fs=defaultdict(list);rids=defaultdict(set)
        for f in db.execute('SELECT *FROM PYTORCH_CALLCHAINS'):
            if f['stack']in sidroles:
                fs[f['id']].append(dict(f,value=ss[f['stack']]))
                for role in sidroles[f['stack']]:rids[role].add(f['id'])
        data=read(Path(ref['cached_native']['path']));defs={d['globalTaskId']:d for d in data['COMPUTE_definitions']};byapi=defaultdict(list)
        for t in data['TASK_records']:
            for aid in t.get('native_linked_API_rowids',[]):byapi[aid].append(t)
        roots=defaultdict(list)
        for r in rows:roots[r['forward_API_rowid']].append(r)
        gap_by_tid=defaultdict(list)
        for forward,rr in roots.items():
            rr=sorted(rr,key=lambda x:x['source_block_ordinal']);assert [x['source_block_ordinal']for x in rr]==list(range(40))
            for prev,nxt in zip(rr,rr[1:]):
                assert nxt['previous_FFN_Post']['TASK']==prev['stages']['FFN_POST']['TASK']and nxt['previous_FFN_Pre']['TASK']==prev['stages']['FFN_PRE']['TASK']
                a,b=prev['stages']['FFN_POST']['API'],nxt['stages']['HC']['API'];g=dict(database_index=di,forward_API_rowid=forward,from_block=prev['source_block_ordinal'],to_block=nxt['source_block_ordinal'],
                    previous_FFN_Pre=prev['stages']['FFN_PRE'],previous_FFN_Post=prev['stages']['FFN_POST'],next_HC=nxt['stages']['HC'],
                    next_Engram_positive_APIs=nxt['Engram_positive_APIs'],next_aux_mean=nxt['aux_mean'],next_aux_allGather=nxt['aux_allGather'],
                    CPU_gap=dict(globalTid=a['globalTid'],startNs=a['endNs'],endNs=b['startNs']),source_linked_intervening_TASKs=[])
                gap_by_tid[a['globalTid']].append(g);bridges.append(g)
        for tid,gg in gap_by_tid.items():gg.sort(key=lambda g:int(g['CPU_gap']['startNs']))
        starts={tid:[int(g['CPU_gap']['startNs'])for g in gg]for tid,gg in gap_by_tid.items()};wanted=set().union(*rids.values());apis=[]
        for raw in db.execute('SELECT rowid AS api_rowid,*FROM PYTORCH_API'):
            a=dict(raw,name_text=ss[raw['name']])
            if a['callchainId']in wanted:apis.append(a)
            tid=a['globalTid']
            if tid not in gap_by_tid:continue
            j=bisect_right(starts[tid],int(a['startNs']))-1
            if j<0:continue
            gap=gap_by_tid[tid][j]
            if not int(gap['CPU_gap']['startNs'])<=int(a['startNs'])<=int(a['endNs'])<=int(gap['CPU_gap']['endNs']):continue
            for t in byapi[a['api_rowid']]:
                d=defs.get(t['globalTaskId'])
                if d:gap['source_linked_intervening_TASKs'].append(dict(API=a,TASK=t,definition={k:d[k]for k in ['compute_record_rowid','globalTaskId','opType_text','inputShapes_text','outputShapes_text','inputDataTypes_text','outputDataTypes_text']}))
        for r in rows:
            a0,a1=r['stages']['FFN_PRE']['API'],r['stages']['FFN_POST']['API'];inside=[a for a in apis if a['globalTid']==a0['globalTid']and int(a0['endNs'])<=int(a['startNs'])<=int(a['endNs'])<=int(a1['startNs'])];st={}
            for role,path,line,ops in SPECS:
                hits=[]
                for a in inside:
                    if a['callchainId']not in rids[role]:continue
                    for t in byapi[a['api_rowid']]:
                        d=defs.get(t['globalTaskId'])
                        if d and d['opType_text']in ops:hits.append((a,t,d))
                if not hits:
                    aa=[a for a in inside if a['callchainId']in rids[role]and a['name_text'].startswith('aclnn')]
                    assert len(aa)==1,(di,r['forward_API_rowid'],r['source_block_ordinal'],role,'API',len(aa))
                    a=aa[0];assert not byapi[a['api_rowid']]
                    st[role]=dict(API=a,TASK=None,definition=None,source_frames=fs[a['callchainId']],evidence_grade='SOURCE_API_ONLY_NO_LINKED_NATIVE_TASK',native_TASK_or_Compute_inferred_by_time=False)
                    continue
                assert len(hits)==1,(di,r['forward_API_rowid'],r['source_block_ordinal'],role,len(hits));a,t,d=hits[0]
                st[role]=dict(API=a,TASK=t,definition={k:d[k]for k in ['compute_record_rowid','globalTaskId','opType_text','inputShapes_text','outputShapes_text','inputDataTypes_text','outputDataTypes_text','blockNum','mixBlockNum','hf32Eligible']},source_frames=fs[a['callchainId']])
            out.append(dict(database_index=di,forward_API_rowid=r['forward_API_rowid'],source_block_ordinal=r['source_block_ordinal'],HC_parent=r,stages=st,
                current_shared_after_MC2_same_stream=(st['SHARED_QUANT']['TASK']['startNs']>=st['MC2']['TASK']['endNs']and st['SHARED_QUANT']['TASK']['streamId']==st['MC2']['TASK']['streamId'])if st['SHARED_QUANT']['TASK']is not None and st['MC2']['TASK']is not None else None,
                actual_router_IDs_counts_or_gate_params_or_MC2_internal_services_bound=False))
        pins.append(dict(database_index=di,raw_database=pin(ref['raw_database']['path']),cached_native=pin(ref['cached_native']['path'])));db.close()
    assert len(out)==960 and len(bridges)==936
    for g in bridges:
        unique={t['TASK']['task_record_rowid']:t for t in g['source_linked_intervening_TASKs']};g['source_linked_intervening_TASKs']=sorted(unique.values(),key=lambda x:x['TASK']['task_record_rowid']);g['all_readers_or_mutation_alias_closed']=False
    counts=Counter((int(r['stages']['FFN_NORM']['definition']['outputShapes_text'].strip('"').split(',')[0]),int(r['stages']['TOPK']['definition']['outputShapes_text'].strip('"').split(',')[0]))for r in out)
    write(O/'hc-moe-native-frontiers-v141.json',dict(records=out,bridges=bridges,databases=pins,frontiers=960,consumer_roles_per_frontier=9,source_bound_TASKs=sum(s['TASK']is not None for r in out for s in r['stages'].values()),API_only_frontiers=[dict(role=role,database_index=r['database_index'],forward_API_rowid=r['forward_API_rowid'],source_block_ordinal=r['source_block_ordinal'],API_rowid=s['API']['api_rowid'])for r in out for role,s in r['stages'].items()if s['TASK']is None],
        interlayer_bridges=936,bridge_Engram_guards=sum(bool(g['next_Engram_positive_APIs'])for g in bridges),bridge_aux_guards=sum(bool(g['next_aux_mean'])for g in bridges),
        recorded_no_intervening_compute_TASK_bridges=sum(not g['source_linked_intervening_TASKs']for g in bridges),
        source_and_router_padded_geometries=[dict(M=m,P=pp,frontiers=n)for (m,pp),n in sorted(counts.items())],new_native_promotions=0,new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='HC_MOE_ROUTER_MC2_SHARED_NEXTLAYER_BOUND',frontiers=960,TASKs=sum(s['TASK']is not None for r in out for s in r['stages'].values()),API_only=sum(s['TASK']is None for r in out for s in r['stages'].values()),bridges=936,Engram_guards=sum(bool(g['next_Engram_positive_APIs'])for g in bridges),aux_guards=sum(bool(g['next_aux_mean'])for g in bridges),no_intervening_compute=sum(not g['source_linked_intervening_TASKs']for g in bridges),new_device_work=False)))


if __name__=='__main__':main()
