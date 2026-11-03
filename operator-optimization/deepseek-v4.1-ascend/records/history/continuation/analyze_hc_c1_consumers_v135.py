"""Bind the C1/SWA norm, old-latent index read, mutation and publication surfaces."""
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json,sqlite3
from finalize_origin_coverage_v9 import N,O,read,write


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
    assert not(O/'REVIEW-CHECKPOINT-v135.json').exists()
    old=read(O/'hc-sp-eager-native-frontiers-v134.json');refs={r['database_index']:r for r in old['databases']};bydb=defaultdict(list);out=[];pins=[]
    for r in old['records']:
        if r['C1']:bydb[r['database_index']].append(r)
    specs=[('SWA_NORM','/attention/dsa_v41.py',487,'RmsNorm'),('SWA_ROPE','/attention/dsa_v41.py',488,'InplacePartialRotaryMul'),
        ('C1_NORM','/models/deepseek_v41/compressor.py',71,'RmsNorm'),('INDEX_WK','/models/deepseek_v41/indexer.py',110,'MatMulV2'),
        ('INDEX_NORM','/models/deepseek_v41/indexer.py',110,'RmsNorm'),('INDEX_ROPE','/models/deepseek_v41/indexer.py',111,'InplacePartialRotaryMul'),
        ('INDEX_QUANT','/models/deepseek_v41/indexer.py',119,'DynamicQuant'),('INDEX_KEY_STORE','/models/deepseek_v41/indexer.py',120,'ScatterNdUpdateSk'),
        ('INDEX_SCALE_CAST','/models/deepseek_v41/indexer.py',124,'Cast'),('INDEX_SCALE_STORE','/models/deepseek_v41/indexer.py',121,'ScatterNdUpdateSk'),
        ('LONG_ROPE','/attention/dsa_v41.py',551,'InplacePartialRotaryMul'),('LONG_STORE','/attention/dsa_v41.py',561,'ScatterNdUpdateSk')]
    for di,records in sorted(bydb.items()):
        ref=refs[di];raw=ref['raw_database'];cache=ref['cached_native'];db=sqlite3.connect(Path(raw['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row
        strings={r['id']:r['value']for r in db.execute('SELECT *FROM STRING_IDS')};chains=defaultdict(list);roleids=defaultdict(set)
        for f in db.execute('SELECT *FROM PYTORCH_CALLCHAINS'):
            value=strings[f['stack']]
            for name,path,line,op in specs:
                if path+f'({line}):'in value:roleids[name].add(f['id']);chains[f['id']].append(dict(f,value=value))
        data=read(Path(cache['path']));defs={r['globalTaskId']:r for r in data['COMPUTE_definitions']};byapi=defaultdict(list)
        for t in data['TASK_records']:
            for aid in t.get('native_linked_API_rowids',[]):byapi[aid].append(t)
        api=[dict(a,name_text=strings[a['name']])for a in db.execute('SELECT rowid AS api_rowid,*FROM PYTORCH_API')]
        for r in records:
            ag,rs=r['SP_allGather']['API'],r['output_reduceScatter']['API'];inside=[a for a in api if a['globalTid']==ag['globalTid']and int(ag['endNs'])<=int(a['startNs'])<=int(a['endNs'])<=int(rs['startNs'])];st={}
            for name,path,line,op in specs:
                hits=[]
                for a in inside:
                    if a['callchainId']not in roleids[name]:continue
                    for t in byapi[a['api_rowid']]:
                        d=defs.get(t['globalTaskId'])
                        if d and d['opType_text']==op:hits.append((a,t,d))
                assert len(hits)==1,(di,r['forward_API_rowid'],name,len(hits));a,t,d=hits[0]
                unique_frames=list({(f['id'],f['stackDepth'],f['stack']):f for f in chains[a['callchainId']]}.values())
                st[name]=dict(API=a,TASK=t,definition={k:d[k]for k in ['compute_record_rowid','globalTaskId','opType_text','inputShapes_text','outputShapes_text','inputDataTypes_text','outputDataTypes_text','blockNum','mixBlockNum','hf32Eligible']},source_frames=unique_frames)
            out.append(dict(database_index=di,forward_API_rowid=r['forward_API_rowid'],source_block_ordinal=r['source_block_ordinal'],eager_HC_SP_parent=r,
                consumers=st,current_native_index_read_end_before_long_RoPE_start=st['INDEX_WK']['TASK']['endNs']<=st['LONG_ROPE']['TASK']['startNs'],
                input_parameter_pointer_or_new_owner_numeric_parity_bound=False,
                grade='EXACT_CPU_SOURCE_AND_CID_NATIVE_CONSUMER_KEYS;OLD_LATENT_WAR_IS_SOURCE_VALUE_CONTRACT'))
        pins.append(dict(database_index=di,raw_database=pin(raw['path']),cached_native=pin(cache['path'])));db.close()
    assert len(out)==24 and all(r['source_block_ordinal']==20 for r in out)
    types=Counter((name,r['consumers'][name]['definition']['outputDataTypes_text'])for r in out for name in ['INDEX_WK','INDEX_QUANT','INDEX_SCALE_CAST','LONG_ROPE'])
    assert types=={('INDEX_WK','DT_BF16'):24,('INDEX_QUANT','INT8;FLOAT'):24,('INDEX_SCALE_CAST','FLOAT16'):24,('LONG_ROPE','DT_BF16'):24}
    write(O/'hc-c1-native-consumers-v135.json',dict(records=out,databases=pins,frontier_count=24,consumer_roles_per_frontier=12,
        fallback_index_quant_key_scale_store_positive=True,captured_graph80_SWA_C1_ambiguities_retained=True,
        current_task_order_not_universal_whole_kernel_WAR_barrier=True,new_native_promotion=False,new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='C1_HC_INDEX_LONG_CONSUMER_FRONTIERS_BOUND',frontiers=24,roles=12,all_old_read_before_long_mutation=all(r['current_native_index_read_end_before_long_RoPE_start']for r in out),new_device_work=False)))


if __name__=='__main__':main()
