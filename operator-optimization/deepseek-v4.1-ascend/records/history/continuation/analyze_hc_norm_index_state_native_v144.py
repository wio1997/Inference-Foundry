"""Join fixed input-RMS/index/C1-C2 state source keys, retaining prior evidence grades."""
from pathlib import Path
from collections import Counter
import json
from finalize_origin_coverage_v9 import O,read,write


def main():
    assert not(O/'REVIEW-CHECKPOINT-v144.json').exists()
    key=lambda r:(r['database_index'],r['forward_API_rowid'],r['source_block_ordinal'])
    norm=read(O/'hc-norm-projection-native-frontiers-v143.json');nn={key(r):r for r in norm['records']};ii=read(O/'hc-index-query-weight-native-producers-v139.json');sparse={key(r):r for r in read(O/'hc-sparse-eager-native-frontiers-v138.json')['records']};cc1={key(r):r for r in read(O/'hc-c1-native-consumers-v135.json')['records']};cc2={key(r):r for r in read(O/'hc-c2-native-consumers-v136.json')['records']};records=[];roles=Counter();seen=set()
    for r in ii['records']:
        k=key(r);n=nn[k];sp=sparse[k];st={role:n['stages'][role]for role in ['HC','INPUT_NORM','QA','QA_NORM','QA_QUANT','QB','SWA_KV','QUERY_ROPE']}
        st.update({role:s for role,s in r['producers'].items()if role not in ['QR_NORM','QB_INPUT_QUANT']});st.update({role:sp['stages'][role]for role in ['QLI','PREPARE','QUERY_QUANT','SMLA']})
        mode='INDEX_QUERY_WEIGHT_ONLY'
        if k in cc1:
            mode='C1';st['C1_PROJ']=n['stages']['C1_PROJ'];st.update({'CACHE_'+role:s for role,s in cc1[k]['consumers'].items()if role not in ['SWA_NORM','SWA_ROPE']})
        if k in cc2:
            mode='C2';st.update({role:n['stages'][role]for role in ['C2_INPUT_CAST','C2_KV','C2_SCORE']});st.update({'CACHE_'+role:s for role,s in cc2[k]['consumers'].items()if role not in ['INPUT_CAST','SWA_NORM','SWA_ROPE']})
        for role,s in st.items():
            assert s['TASK']is not None,(k,role);tk=k[0],s['TASK']['task_record_rowid'];assert tk not in seen,(k,role,tk);seen.add(tk);roles[role]+=1
        records.append(dict(database_index=k[0],forward_API_rowid=k[1],source_block_ordinal=k[2],mode=mode,stages=st,parent_keys=dict(input_norm='hc-norm-projection-native-frontiers-v143.json',index='hc-index-query-weight-native-producers-v139.json',sparse='hc-sparse-eager-native-frontiers-v138.json',C1='hc-c1-native-consumers-v135.json'if k in cc1 else None,C2='hc-c2-native-consumers-v136.json'if k in cc2 else None),actual_tensor_data_edge_pointer_parameter_or_state_epoch_bound=False))
    assert len(records)==192 and len(seen)==5064
    write(O/'hc-norm-index-state-native-joins-v144.json',dict(records=records,databases=norm['databases'],frontiers=192,source_bound_TASK_bindings=len(seen),native_role_counts=dict(roles),modes=dict(Counter(r['mode']for r in records)),new_native_bindings_or_promotions=0,scope='Exact saved source/CID native task-key union across prior reviewed artifacts;new composition joins only. Not same tensor values,pointers,request/weight/state epochs or causal timing.',new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='JOINED',frontiers=192,TASKs=len(seen),modes=Counter(r['mode']for r in records),new_native_promotions=0,new_device_work=False)))


if __name__=='__main__':main()
