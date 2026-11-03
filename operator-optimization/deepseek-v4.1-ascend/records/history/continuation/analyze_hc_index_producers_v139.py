"""Source/CID producer joins for the192 saved eager index-query/weight fronts."""
from pathlib import Path
from collections import defaultdict,Counter
import sqlite3,hashlib,json
from finalize_origin_coverage_v9 import O,read,write


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


SPECS=[('QR_NORM','/attention/dsa_v41.py',477,'RmsNorm'),('QB_INPUT_QUANT','/attention/dsa_v41.py',478,'DynamicQuant'),
    ('INDEX_INPUT_QUANT','/models/deepseek_v41/indexer.py',168,'DynamicQuant'),('INDEX_Q','/models/deepseek_v41/indexer.py',168,'QuantBatchMatmulV3'),
    ('WEIGHT','/models/deepseek_v41/indexer.py',174,'MatMulV2'),('WEIGHT_FLOAT','/models/deepseek_v41/indexer.py',174,'Cast'),
    ('WEIGHT_SCALE','/models/deepseek_v41/indexer.py',174,'Mul'),('WEIGHT_FP16','/models/deepseek_v41/indexer.py',256,'Cast'),
    ('INDEX_ROPE','/models/deepseek_v41/indexer.py',171,'InplacePartialRotaryMul')]


def overlap(a,b):return max(0,min(a['endNs'],b['endNs'])-max(a['startNs'],b['startNs']))


def main():
    assert not(O/'REVIEW-CHECKPOINT-v139.json').exists()
    old=read(O/'hc-sparse-eager-native-frontiers-v138.json');groups=defaultdict(list);refs={r['database_index']:r for r in old['databases']};out=[];pins=[]
    for r in old['records']:
        if r['stages']['QLI']:groups[r['database_index']].append(r)
    for di,rows in sorted(groups.items()):
        ref=refs[di];db=sqlite3.connect(Path(ref['raw_database']['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;ss={r['id']:r['value']for r in db.execute('SELECT *FROM STRING_IDS')};sidroles=defaultdict(set)
        for sid,v in ss.items():
            for role,path,line,op in SPECS:
                if path+f'({line}):'in v:sidroles[sid].add(role)
        fs=defaultdict(dict);rids=defaultdict(set)
        for f in db.execute('SELECT *FROM PYTORCH_CALLCHAINS'):
            if f['stack']in sidroles:
                fs[f['id']][f['id'],f['stackDepth'],f['stack']]=dict(f,value=ss[f['stack']])
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
                assert len(hits)==1,(di,r['forward_API_rowid'],r['source_block_ordinal'],role,len(hits));a,t,d=hits[0]
                st[role]=dict(API=a,TASK=t,definition={k:d[k]for k in ['compute_record_rowid','globalTaskId','opType_text','inputShapes_text','outputShapes_text','inputDataTypes_text','outputDataTypes_text','blockNum','mixBlockNum','hf32Eligible']},source_frames=list(fs[a['callchainId']].values()))
            wt,qp,rope,quant=st['WEIGHT']['TASK'],st['INDEX_Q']['TASK'],st['INDEX_ROPE']['TASK'],r['stages']['QUERY_QUANT']['TASK']
            out.append(dict(database_index=di,forward_API_rowid=r['forward_API_rowid'],source_block_ordinal=r['source_block_ordinal'],eager_HC_sparse_parent=r,producers=st,
                observed=dict(IndexQ_TASK_end_before_Weight_TASK_start=qp['endNs']<=wt['startNs'],Weight_and_IndexQuant_same_stream=wt['streamId']==quant['streamId'],Weight_RoPE_TASK_interval_overlap_ns=overlap(wt,rope),
                    IndexQuant_C2KV_TASK_interval_overlap_ns=overlap(quant,p['C2_KV']['TASK'])if p['C2_KV']else None),
                actual_QR_quantizer_value_identity_parameter_rows_or_new_HC_schedule_bound=False))
        pins.append(dict(database_index=di,raw_database=pin(ref['raw_database']['path']),cached_native=pin(ref['cached_native']['path'])));db.close()
    assert len(out)==192
    write(O/'hc-index-query-weight-native-producers-v139.json',dict(records=out,databases=pins,frontiers=192,new_producer_roles_per_frontier=9,source_bound_producer_TASKs=1728,
        prior_QueryQuant_QLI_TASKs_reused=384,source_blocks=dict(Counter(r['source_block_ordinal']for r in out)),
        TASK_intervals_not_hardware_utilization_or_causal_gain=True,new_native_promotions=0,new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='HC_INDEX_QUERY_WEIGHT_PRODUCERS_BOUND',frontiers=192,roles=9,TASKs=1728,new_device_work=False)))


if __name__=='__main__':main()
