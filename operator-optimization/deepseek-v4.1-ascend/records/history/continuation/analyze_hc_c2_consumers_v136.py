"""Join saved C2 HC/SP parents to their source/CID native state consumers."""
from pathlib import Path
from collections import defaultdict,Counter
import sqlite3,json,hashlib
from finalize_origin_coverage_v9 import O,read,write


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


SPECS=[('INPUT_CAST','/attention/dsa_v41.py',526,'Cast'),
    ('SWA_NORM','/attention/dsa_v41.py',487,'RmsNorm'),('SWA_ROPE','/attention/dsa_v41.py',488,'InplacePartialRotaryMul'),
    ('POOL_ZERO','/compressor/compressor_triton.py',675,'ZerosLike'),
    ('POOL','/compressor/compressor_triton.py',684,'_pool_kernel'),
    ('RING_UPDATE','/compressor/compressor_triton.py',706,'_cache_update_kernel'),
    ('POOL_NORM','/models/deepseek_v41/compressor.py',67,'RmsNorm'),
    ('INDEX_WK','/models/deepseek_v41/indexer.py',110,'MatMulV2'),('INDEX_NORM','/models/deepseek_v41/indexer.py',110,'RmsNorm'),
    ('INDEX_ROPE','/models/deepseek_v41/indexer.py',111,'InplacePartialRotaryMul'),('INDEX_QUANT','/models/deepseek_v41/indexer.py',119,'DynamicQuant'),
    ('INDEX_KEY_STORE','/models/deepseek_v41/indexer.py',120,'ScatterNdUpdateSk'),('INDEX_SCALE_CAST','/models/deepseek_v41/indexer.py',124,'Cast'),
    ('INDEX_SCALE_STORE','/models/deepseek_v41/indexer.py',121,'ScatterNdUpdateSk'),('LONG_ROPE','/attention/dsa_v41.py',551,'InplacePartialRotaryMul'),
    ('LONG_STORE','/attention/dsa_v41.py',561,'ScatterNdUpdateSk')]


def main():
    assert not(O/'REVIEW-CHECKPOINT-v136.json').exists()
    old=read(O/'hc-sp-eager-native-frontiers-v134.json');refs={r['database_index']:r for r in old['databases']};groups=defaultdict(list);out=[];pins=[]
    for r in old['records']:
        if r['C2_KV']:groups[r['database_index']].append(r)
    for di,records in sorted(groups.items()):
        ref=refs[di];db=sqlite3.connect(Path(ref['raw_database']['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row
        strings={x['id']:x['value']for x in db.execute('SELECT *FROM STRING_IDS')};chains=defaultdict(dict);roleids=defaultdict(set)
        for f in db.execute('SELECT *FROM PYTORCH_CALLCHAINS'):
            value=strings[f['stack']]
            for role,path,line,op in SPECS:
                if path+f'({line}):'in value:
                    roleids[role].add(f['id']);chains[f['id']][f['id'],f['stackDepth'],f['stack']]=dict(f,value=value)
        data=read(Path(ref['cached_native']['path']));defs={d['globalTaskId']:d for d in data['COMPUTE_definitions']};byapi=defaultdict(list)
        for t in data['TASK_records']:
            for aid in t.get('native_linked_API_rowids',[]):byapi[aid].append(t)
        apis=[dict(a,name_text=strings[a['name']])for a in db.execute('SELECT rowid AS api_rowid,*FROM PYTORCH_API')]
        for r in records:
            ag,rs=r['SP_allGather']['API'],r['output_reduceScatter']['API'];inside=[a for a in apis if a['globalTid']==ag['globalTid']and int(ag['endNs'])<=int(a['startNs'])<=int(a['endNs'])<=int(rs['startNs'])];st={}
            for role,path,line,op in SPECS:
                hits=[]
                for a in inside:
                    if a['callchainId']not in roleids[role]:continue
                    for t in byapi[a['api_rowid']]:
                        d=defs.get(t['globalTaskId'])
                        if d and d['opType_text']==op:hits.append((a,t,d))
                assert len(hits)==1,(di,r['forward_API_rowid'],r['source_block_ordinal'],role,len(hits))
                a,t,d=hits[0];st[role]=dict(API=a,TASK=t,definition={k:d[k]for k in ['compute_record_rowid','globalTaskId','opType_text','inputShapes_text','outputShapes_text','inputDataTypes_text','outputDataTypes_text','blockNum','mixBlockNum','hf32Eligible']},source_frames=list(chains[a['callchainId']].values()))
            out.append(dict(database_index=di,forward_API_rowid=r['forward_API_rowid'],source_block_ordinal=r['source_block_ordinal'],eager_HC_SP_parent=r,consumers=st,
                current_pool_update_norm_TASK_order=st['POOL']['TASK']['endNs']<=st['RING_UPDATE']['TASK']['startNs']and st['RING_UPDATE']['TASK']['endNs']<=st['POOL_NORM']['TASK']['startNs'],
                current_index_old_read_before_long_write=st['INDEX_WK']['TASK']['endNs']<=st['LONG_ROPE']['TASK']['startNs'],
                metadata_values_page_generation_actual_aliases_or_new_owner_numeric_parity_bound=False,
                grade='EXACT_CPU_SOURCE_AND_CID_NATIVE_KEYS;OBSERVED_ORDER_IS_NOT_A_UNIVERSAL_BARRIER'))
        pins.append(dict(database_index=di,raw_database=pin(ref['raw_database']['path']),cached_native=pin(ref['cached_native']['path'])));db.close()
    assert len(out)==72 and Counter(r['source_block_ordinal']for r in out)=={2:24,8:24,14:24}
    write(O/'hc-c2-native-consumers-v136.json',dict(records=out,databases=pins,frontier_count=72,consumer_roles_per_frontier=len(SPECS),
        roles=[dict(role=r,path=p,line=l,op=o)for r,p,l,o in SPECS],source_bound_native_consumer_TASKs=len(out)*len(SPECS),
        observed_order_not_required_whole_operator_barrier=True,metadata_E_B_U_not_measured=True,new_native_promotion=False,new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='HC_C2_NATIVE_CONSUMERS_BOUND',frontiers=len(out),roles=len(SPECS),TASKs=len(out)*len(SPECS),source_blocks=[2,8,14],new_device_work=False)))


if __name__=='__main__':main()
