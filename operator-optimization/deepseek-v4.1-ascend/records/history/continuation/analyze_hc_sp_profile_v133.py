"""Exact saved graph/draft HC forks and SP carriers, retaining branch/role gaps."""
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json,sqlite3
from finalize_origin_coverage_v9 import N,O,read,write


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
    assert not(O/'REVIEW-CHECKPOINT-v133.json').exists()
    old=read(O/'HC-attention-fork-native-frontiers-v93.json');graph={(r['database_index'],r['graph_ordinal'],r['block_ordinal']):r for r in old['frontiers']}
    edges={(r['database_index'],r['graph_ordinal'],r['block_ordinal']):r for r in read(O/'HC-operator-profile-and-edges-v58.json')['exact_graph_role_edges']if r['kind']=='CROSSLAYER_POST_TO_PRE'}
    sp={(r['database_index'],r['graph_ordinal'],r['block_ordinal']):r for r in read(O/'communication-native-operator-population-v66.json')['operations']if r['semantic_role']=='ATTENTION_INPUT_SP_ALLGATHER'and r.get('graph_ordinal')is not None}
    draft=read(O/'context-sp-profile-frontiers-v130.json');records=[];refs=[];rawrefs={r['database_index']:r for r in read(O/'host-launch-original-API-coverage-v126.json')['databases']}
    for ref in draft['databases']:
        di=ref['database_index'];data=read(Path(ref['path']));ts={t['task_record_rowid']:t for t in data['TASK_records']};defs={d['globalTaskId']:d for d in data['COMPUTE_definitions']};apis={a['api_rowid']:a for a in data['linked_PYTORCH_APIs']}
        raw=rawrefs[di];db=sqlite3.connect(Path(raw['path']).as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row
        frames=defaultdict(list);rolechains=defaultdict(set)
        for x in db.execute("SELECT c.id,c.stackDepth,c.stack,s.value FROM PYTORCH_CALLCHAINS c JOIN STRING_IDS s ON c.stack=s.id WHERE s.value LIKE '%/models/deepseek_v41/model.py(889): forward' OR s.value LIKE '%/models/deepseek_v41/model.py(896): forward' OR s.value LIKE '%/models/deepseek_v41/model.py(902): forward' OR s.value LIKE '%/models/deepseek_v41/model.py(905): forward' OR s.value LIKE '%/models/deepseek_v41/model.py(919): forward'"):
            role=next(i for i in [889,896,902,905,919]if f'({i})'in x['value']);rolechains[role].add(x['id']);frames[x['id']].append(dict(x))
        def stage(tid):
            t=ts[tid];d=defs[t['globalTaskId']];aa=[apis[i]for i in t.get('native_linked_API_rowids',[])];ff=[f for a in aa for f in frames.get(a['callchainId'],[])]
            return dict(TASK=t,definition={k:d[k]for k in ['globalTaskId','compute_record_rowid','opType_text','inputShapes_text','outputShapes_text','inputDataTypes_text','outputDataTypes_text','blockNum','mixBlockNum','hf32Eligible']},APIs=aa,selected_model_frames=ff)
        for key,f in sorted(graph.items()):
            if key[0]!=di:continue
            g=sp[key];assert g['count']==15360 and g['rankSize']==8
            x=edges.get(key);prev=graph.get((di,key[1],key[2]-1));kind='INITIAL_PREMIX_SOURCE_ONLY'if not prev else 'AFTER_ENGRAM_WRITE'if key[2]in [1,14]else 'AUX_READER_STILL_LIVE'if key[2]in [37,38,39]else'NO_INTERVENING_COMPUTE_RECORDED'
            stages={name:stage(f[field])for name,field in [('HC','attention_HcPre_TASK'),('NORM','input_norm_TASK'),('POST','attention_HcPost_TASK'),('NEXT_FFN_PRE','FFN_HcPre_TASK')]}
            assert stages['HC']['TASK']['endNs']<=stages['NORM']['TASK']['startNs']<=stages['NORM']['TASK']['endNs']<=g['startNs']<=g['endNs']<=stages['POST']['TASK']['startNs']
            records.append(dict(domain='TARGET_GRAPH',database_index=di,graph_ordinal=key[1],block_ordinal=key[2],root_bound=f['root_bound'],stages=stages,SP_allGather=g,
                previous_FFN_Pre=stage(prev['FFN_HcPre_TASK'])if prev else None,previous_FFN_Post=stage(x['producer_task_rowid'])if x else None,
                crosslayer_record=x,previous_input_guard=kind,SWA_role_unique=f['SWA_role_unique'],KV_BF16_candidates=f['KV_BF16_candidates'],
                exact_pointer_premix_values_or_new_handoff_bound=False,grade='EXACT_NATIVE_KEYS_GRAPH_CONTROL_AND_SOURCE_CONTRACT;NOT_TENSOR_BINDING'))
        def role(t,line):return any(apis[i]['callchainId']in rolechains[line]for i in t.get('native_linked_API_rowids',[]))
        for f in [r for r in draft['records']if r['database_index']==di]:
            previous=None
            for j,l in enumerate(f['layers']):
                g=l['draft_SP_allGather'];start=f['draft_repeat']['endNs']if j==0 else f['layers'][j-1]['draft_SP_allGather']['endNs'];end=f['layers'][j+1]['draft_SP_allGather']['startNs']if j<2 else f['seed_TASK']['startNs']
                before=[t for t in ts.values()if start<=t['startNs']<=t['endNs']<=g['startNs'] and t['globalTaskId']in defs]
                after=[t for t in ts.values()if g['endNs']<=t['startNs']<=t['endNs']<=end and t['globalTaskId']in defs]
                chosen={}
                for name,line,ops,items in [('HC',889,'HcPre',before),('NORM',896,'RmsNorm',before),('POST',902,'HcPost',after),('NEXT_FFN_PRE',905,'HcPre',after),('FFN_POST',919,'HcPost',after)]:
                    hits=[t for t in items if defs[t['globalTaskId']]['opType_text']==ops and role(t,line)];assert len(hits)==1,(di,j,name,len(hits));chosen[name]=stage(hits[0]['task_record_rowid'])
                records.append(dict(domain='DRAFT_SOURCE_ORDER',database_index=di,source_order_ordinal=f['source_order_ordinal'],draft_layer_ordinal=j,source_context=f['source_context'],
                    stages=chosen,SP_allGather=g,previous_FFN_Pre=previous['stages']['NEXT_FFN_PRE']if previous else None,previous_FFN_Post=previous['stages']['FFN_POST']if previous else None,
                    previous_input_guard='INITIAL_PREMIX_SOURCE_ONLY'if previous is None else'NO_DRAFT_ENGRAM_IN_PINNED_FORWARD',
                    positive_query_KV=l['query_KV'],positive_metadata_None_zero=l['dummy_zero_output_witness'],
                    exact_pointer_premix_values_or_new_handoff_bound=False,grade='EXACT_NATIVE_KEYS_POSITIVE_MODEL_CALLCHAINS_AND_SOURCE_ORDER;NOT_TENSOR_BINDING'))
                previous=records[-1]
        refs.append(dict(**pin(ref['path']),database_index=di,raw_database=pin(raw['path'])));db.close()
    graphs=[r for r in records if r['domain']=='TARGET_GRAPH'];drafts=[r for r in records if r['domain']=='DRAFT_SOURCE_ORDER'];guards=Counter(r['previous_input_guard']for r in graphs)
    assert len(graphs)==3200 and len(drafts)==312 and sum(x['root_bound']for x in graphs)==2560
    assert guards=={'INITIAL_PREMIX_SOURCE_ONLY':80,'AFTER_ENGRAM_WRITE':160,'AUX_READER_STILL_LIVE':240,'NO_INTERVENING_COMPUTE_RECORDED':2720}
    assert sum(bool(r['positive_query_KV'])for r in drafts)==264 and sum(bool(r['positive_metadata_None_zero'])for r in drafts)==48
    write(O/'hc-sp-native-frontiers-v133.json',dict(records=records,databases=refs,target_graph_frontiers=3200,root_bound_graph_frontiers=2560,phase_unbound_graph_frontiers=640,
        graph_SWA_C1_role_ambiguities_retained=80,draft_frontiers=312,draft_positive_query_KV=264,draft_positive_metadata_None_zero=48,target_previous_input_guards=dict(guards),
        no_new_native_promotion_or_crossrank_group_alignment=True,new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='HC_SP_GRAPH_AND_DRAFT_FORKS_RECONCILED',target=3200,draft=312,target_guards=dict(guards),new_device_work=False)))


if __name__=='__main__':main()
