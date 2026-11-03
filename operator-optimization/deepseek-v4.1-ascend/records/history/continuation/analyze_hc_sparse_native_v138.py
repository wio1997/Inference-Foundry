"""Join saved eager HC/SP fronts to SMLA, sparse-index publication and head layout."""
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json,sqlite3
from finalize_origin_coverage_v9 import O,read,write


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


SPECS=[('SMLA','/attention/dsa_v41.py',657,'SparseFlashMla'),('INVERSE_ROPE','/attention/dsa_v41.py',712,'InplacePartialRotaryMul'),
    ('SEND_LAYOUT','/attention/context_parallel/dsa_v41_cp.py',215,'Transpose'),('QLI','/models/deepseek_v41/indexer.py',275,'VllmAscendQuantLightningIndexerV2'),
    ('PREPARE','/models/deepseek_v41/indexer.py',291,'_prepare_indexer_indices_kernel'),('QUERY_QUANT','/models/deepseek_v41/indexer.py',179,'_quantize_indexer_query_kernel'),
    ('CANDIDATE_COPY','/attention/dsa_v41.py',610,'TensorMove'),('TOPK_COPY_API','/attention/dsa_v41.py',608,None)]


def main():
    assert not(O/'REVIEW-CHECKPOINT-v138.json').exists()
    p=read(O/'hc-sp-eager-native-frontiers-v134.json');refs={r['database_index']:r for r in p['databases']};groups=defaultdict(list);out=[];pins=[];counts=Counter()
    for r in p['records']:groups[r['database_index']].append(r)
    for di,rows in sorted(groups.items()):
        ref=refs[di];db=sqlite3.connect(Path(ref['raw_database']['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row
        ss={r['id']:r['value']for r in db.execute('SELECT *FROM STRING_IDS')};sidroles=defaultdict(set)
        for sid,v in ss.items():
            for name,path,line,op in SPECS:
                if path+f'({line}):'in v:sidroles[sid].add(name)
        frames=defaultdict(list);roleids=defaultdict(set)
        for f in db.execute('SELECT *FROM PYTORCH_CALLCHAINS'):
            if f['stack']in sidroles:
                frames[f['id']].append(dict(f,value=ss[f['stack']]))
                for role in sidroles[f['stack']]:roleids[role].add(f['id'])
        wanted=set().union(*roleids.values());apis=[dict(a,name_text=ss[a['name']])for a in db.execute('SELECT rowid AS api_rowid,*FROM PYTORCH_API')if a['callchainId']in wanted]
        cache=read(Path(ref['cached_native']['path']));defs={d['globalTaskId']:d for d in cache['COMPUTE_definitions']};byapi=defaultdict(list)
        for t in cache['TASK_records']:
            for aid in t.get('native_linked_API_rowids',[]):byapi[aid].append(t)
        for r in rows:
            ag,rs=r['SP_allGather']['API'],r['output_reduceScatter']['API'];inside=[a for a in apis if a['globalTid']==ag['globalTid']and int(ag['endNs'])<=int(a['startNs'])<=int(a['endNs'])<=int(rs['startNs'])];st={};block=r['source_block_ordinal'];isindex=block in [2,8,14,20,24,28,32,36]
            for role,path,line,op in SPECS:
                if role=='TOPK_COPY_API':
                    hits=[a for a in inside if a['callchainId']in roleids[role]and a['name_text']=='aclnnInplaceCopy'];assert len(hits)==int(isindex),(di,block,role,len(hits))
                    st[role]=[dict(API=a,source_frames=frames[a['callchainId']],exact_linked_TASKs=byapi[a['api_rowid']],grade='SOURCE_API_WITH_EXACT_CID_TASK_LIST;NO_ZERO_COST_OR_ALIAS_INFERENCE')for a in hits];continue
                hits=[]
                for a in inside:
                    if a['callchainId']not in roleids[role]:continue
                    for t in byapi[a['api_rowid']]:
                        d=defs.get(t['globalTaskId'])
                        if d and d['opType_text']==op:hits.append((a,t,d))
                expect=1 if role in ['SMLA','INVERSE_ROPE','SEND_LAYOUT']else int(block==20)if role=='CANDIDATE_COPY'else int(isindex)
                assert len(hits)==expect,(di,r['forward_API_rowid'],block,role,len(hits),expect)
                if hits:
                    a,t,d=hits[0];st[role]=dict(API=a,TASK=t,definition={k:d[k]for k in ['compute_record_rowid','globalTaskId','opType_text','inputShapes_text','outputShapes_text','inputDataTypes_text','outputDataTypes_text','blockNum','mixBlockNum','hf32Eligible']},source_frames=frames[a['callchainId']]);counts[role]+=1
                else:st[role]=None
            family='SWA_ONLY'if block<2 else'CSA_C2'if block<20 else'CSA_C1';mode='MODE3'if block in [2,8,14]else'MODE1'if block==20 else'MODE2'if isindex else'SHARED_TOPK'if block>=2 else'NONE'
            out.append(dict(database_index=di,forward_API_rowid=r['forward_API_rowid'],source_block_ordinal=block,eager_HC_SP_parent=r,stages=st,family=family,index_mode=mode,
                selected_ID_payload_or_crossrank_episode_or_new_partial_state_ABI_bound=False))
        pins.append(dict(database_index=di,raw_database=pin(ref['raw_database']['path']),cached_native=pin(ref['cached_native']['path'])));db.close()
    assert len(out)==960 and counts=={'SMLA':960,'INVERSE_ROPE':960,'SEND_LAYOUT':960,'QLI':192,'PREPARE':192,'QUERY_QUANT':192,'CANDIDATE_COPY':24}
    byforward=defaultdict(list)
    for r in out:byforward[r['database_index'],r['forward_API_rowid']].append(r)
    shared=[]
    for key,rows in sorted(byforward.items()):
        rows=sorted(rows,key=lambda r:r['source_block_ordinal']);assert [r['source_block_ordinal']for r in rows]==list(range(40))
        source=rows[20];consumers=[rows[b]for b in [24,28,32,36]]
        for r in rows[2:]:
            owner=max(b for b in [2,8,14,20,24,28,32,36]if b<=r['source_block_ordinal'])
            r['source_ordinal_shared_topk_producer']=owner
        shared.append(dict(database_index=key[0],forward_API_rowid=key[1],candidate_source_block=20,candidate_publish=source['stages']['CANDIDATE_COPY'],
            consumers=[dict(block=r['source_block_ordinal'],QLI=r['stages']['QLI'],HC=r['eager_HC_SP_parent']['stages']['HC'],
                candidate_publish_TASK_ends_before_consumer_HC_TASK_starts=source['stages']['CANDIDATE_COPY']['TASK']['endNs']<=r['eager_HC_SP_parent']['stages']['HC']['TASK']['startNs'])for r in consumers],
            source_role_membership_not_candidate_values_or_bank_epochs=True))
    write(O/'hc-sparse-eager-native-frontiers-v138.json',dict(records=out,databases=pins,forward_candidate_publish_frontiers=shared,frontier_count=960,CPU_forward_roots=24,stage_counts=dict(counts),source_bound_TASKs=sum(counts.values()),
        families=dict(Counter(r['family']for r in out)),index_modes=dict(Counter(r['index_mode']for r in out)),exact_TOPK_copy_APIs=192,
        no_linked_TOPK_copy_TASK_is_not_zero_cost_proof=True,new_native_promotions=0,new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='HC_SPARSE_EAGER_NATIVE_FRONTIERS_BOUND',frontiers=960,CPU_forwards=24,stage_counts=dict(counts),TASKs=sum(counts.values()),new_device_work=False)))


if __name__=='__main__':main()
