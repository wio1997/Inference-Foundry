"""Bind input-HC RMS and all positive eager QA-normalization/quantization branches."""
from pathlib import Path
from collections import defaultdict,Counter
import sqlite3,hashlib,json
from finalize_origin_coverage_v9 import N,O,read,write


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
    assert not(O/'REVIEW-CHECKPOINT-v143.json').exists()
    old=read(O/'hc-sp-eager-native-frontiers-v134.json');rope={(r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']):r for r in read(O/'hc-rope-latent-native-frontiers-v140.json')['query_records']};c1={(r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']):r for r in read(O/'hc-c1-native-consumers-v135.json')['records']};c2={(r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']):r for r in read(O/'hc-c2-native-consumers-v136.json')['records']};groups=defaultdict(list);refs={x['database_index']:x for x in old['databases']};out=[];pins=[];holes=[]
    for r in old['records']:groups[r['database_index']].append(r)
    for di,rr in sorted(groups.items()):
        ref=refs[di];db=sqlite3.connect(Path(ref['raw_database']['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;ss={x['id']:x['value']for x in db.execute('SELECT *FROM STRING_IDS')};fs=defaultdict(list)
        for x in db.execute('SELECT *FROM PYTORCH_CALLCHAINS'):
            v=ss.get(x['stack'])
            if v and('/attention/dsa_v41.py(477):'in v or'/attention/dsa_v41.py(478):'in v):fs[x['id']].append(dict(x,value=v))
        apis=[dict(x,name_text=ss[x['name']])for x in db.execute('SELECT rowid AS api_rowid,*FROM PYTORCH_API')if x['callchainId']in fs];data=read(Path(ref['cached_native']['path']));defs={x['globalTaskId']:x for x in data['COMPUTE_definitions']};byapi=defaultdict(list)
        for t in data['TASK_records']:
            for aid in t.get('native_linked_API_rowids',[]):byapi[aid].append(t)
        for r in rr:
            key=di,r['forward_API_rowid'],r['source_block_ordinal'];q=rope[key];st=dict(HC=r['stages']['HC'],INPUT_NORM=r['stages']['NORM'],QA=r['QA'],SWA_KV=r['SWA_KV'],**q['stages']);a0,a1=r['SP_allGather']['API'],r['output_reduceScatter']['API'];inside=[a for a in apis if a['globalTid']==a0['globalTid']and int(a0['endNs'])<=int(a['startNs'])<=int(a['endNs'])<=int(a1['startNs'])]
            for role,line,op,name in [('QA_NORM',477,'RmsNorm','aclnnRmsNorm'),('QA_QUANT',478,'DynamicQuant','aclnnDynamicQuantV2')]:
                aa=[a for a in inside if any(f'/attention/dsa_v41.py({line}):'in f['value']for f in fs[a['callchainId']])];hits=[]
                for a in aa:
                    for t in byapi[a['api_rowid']]:
                        d=defs.get(t['globalTaskId'])
                        if d and d['opType_text']==op:hits.append((a,t,d))
                if not hits:
                    candidates=[a for a in aa if a['name_text']==name];assert len(candidates)==1,(key,role,'API-only',len(candidates));a=candidates[0];assert not byapi[a['api_rowid']]
                    st[role]=dict(API=a,TASK=None,definition=None,selected_frames=fs[a['callchainId']],evidence_grade='SOURCE_API_ONLY_NO_LINKED_NATIVE_TASK',time_nearest_repair=False);holes.append(dict(key=list(key),role=role,API_rowid=a['api_rowid']));continue
                assert len(hits)==1,(key,role,len(hits));a,t,d=hits[0];st[role]=dict(API=a,TASK=t,definition={k:d[k]for k in ['compute_record_rowid','globalTaskId','opType_text','inputShapes_text','outputShapes_text','inputDataTypes_text','outputDataTypes_text','blockNum','mixBlockNum','hf32Eligible']},selected_frames=fs[a['callchainId']])
            if r['C1']:st.update(C1_PROJ=r['C1'],C1_NORM=c1[key]['consumers']['C1_NORM'])
            if r['C2_KV']:st.update(C2_INPUT_CAST=c2[key]['consumers']['INPUT_CAST'],C2_KV=r['C2_KV'],C2_SCORE=r['C2_SCORE'])
            out.append(dict(database_index=di,forward_API_rowid=key[1],source_block_ordinal=key[2],stages=st,SP_allGather=r['SP_allGather'],output_reduceScatter=r['output_reduceScatter'],parent_keys=dict(HC_SP='hc-sp-eager-native-frontiers-v134.json',RoPE='hc-rope-latent-native-frontiers-v140.json',C1='hc-c1-native-consumers-v135.json'if r['C1']else None,C2='hc-c2-native-consumers-v136.json'if r['C2_KV']else None),actual_input_tensor_gamma_rstd_or_weight_epoch_identity=False))
        pins.append(dict(database_index=di,raw_database=pin(ref['raw_database']['path']),cached_native=pin(ref['cached_native']['path'])));db.close()
    assert len(out)==960
    counts=Counter(role for r in out for role,s in r['stages'].items()if s['TASK']is not None)
    write(O/'hc-norm-projection-native-frontiers-v143.json',dict(records=out,databases=pins,frontiers=960,TASK_bindings=sum(counts.values()),native_role_counts=dict(counts),API_only_roles=holes,old_QR_roles192x2_reused=384,new_QR_source_roles=sum(counts[k]for k in ['QA_NORM','QA_QUANT']),all_input_normalizer_rstd_or_Z_readers_closed=False,new_native_promotions=0,new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='BOUND',frontiers=960,TASK_bindings=sum(counts.values()),roles=counts,API_only=len(holes),new_device_work=False,overall_complete=False)))


if __name__=='__main__':main()
