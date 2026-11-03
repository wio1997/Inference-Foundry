"""Bind saved checkpoint auxiliary/SP collectives and input-selector callers."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,sqlite3
from finalize_origin_coverage_v9 import N,O,read,write


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
    assert not (O/'REVIEW-CHECKPOINT-v131.json').exists()
    pop=read(O/'communication-native-operator-population-v66.json')['operations'];aux=read(O/'index-context-native-chains-v64.json')['aux_checkpoint_graph_chains']
    taskops={(r['database_index'],r['TASK_rowid']):r for r in pop if r['TASK_rowid']is not None}
    bygraph=defaultdict(list)
    for r in pop:
        if r['semantic_role']=='ATTENTION_INPUT_SP_ALLGATHER'and r.get('graph_ordinal')is not None:bygraph[(r['database_index'],r['graph_ordinal'],r['block_ordinal'])].append(r)
    pairs=[]
    for a in aux:
        di=a['database_index'];ag=taskops[(di,a['allgather']['TASK_rowid'])];sp=bygraph[(di,a['graph_ordinal'],a['checkpoint_before_block'])];assert len(sp)==1;sp=sp[0]
        assert ag['semantic_role']=='DSPARK_AUX_RESIDUAL_ALLGATHER'and ag['graph_ordinal']==a['graph_ordinal']
        assert ag['count']==sp['count']and ag['rankSize']==sp['rankSize']==8 and ag['dtype']==sp['dtype']=='BFP16'
        assert ag['endNs']<=sp['startNs']and ag['count']%5120==0
        pairs.append(dict(database_index=di,graph_ordinal=a['graph_ordinal'],root_bound=a['root_bound'],
            checkpoint_before_block=a['checkpoint_before_block'],mean=a['mean'],aux_allGather=ag,attention_SP_allGather=sp,
            sender_rows_from_count=ag['count']//5120,same_group_label=ag['group_name']==sp['group_name'],
            actual_tensor_alias_group_members_or_ready_events_bound=False,
            grade='EXACT_NATIVE_KEYS_GRAPHCONTROL_BLOCKORDER_SCHEMA;NOT_EQUAL_VALUE_OR_POINTER_PROOF'))
    assert len(pairs)==240 and sum(p['root_bound']for p in pairs)==192
    eager=[r for r in pop if r['semantic_role']=='DSPARK_AUX_RESIDUAL_ALLGATHER'and r.get('graph_ordinal')is None];assert len(eager)==72
    write(O/'aux-sp-checkpoint-profile-v131.json',dict(pairs=pairs,pair_count=240,root_bound_pairs=192,phase_unbound_pairs=48,
        eager_aux_operators_retained_unpaired=eager,all_aux_operators=312,
        count_geometry=dict(Counter(r['sender_rows_from_count']for r in pairs)),
        equalcount_or_group_label_not_value_alias_or_common_group_proof=True,new_native_promotions=0,new_device_work=False,overall_complete=False))
    rawrefs=read(O/'host-launch-original-API-coverage-v126.json')['databases'];records=[];dbrefs=[]
    native=N/'original-profile-cache-v53/analysis/native-execution-identity-v14'
    for ref in rawrefs:
        p=Path(ref['path']);identity=pin(p);assert identity['sha256']==ref['sha256']
        db=sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row
        roles={};matched=defaultdict(list)
        for r in db.execute("SELECT c.id,c.stackDepth,s.value FROM PYTORCH_CALLCHAINS c JOIN STRING_IDS s ON s.id=c.stack WHERE s.value LIKE '%/worker/model_runner_v1.py(2069): propose_draft_token_ids'OR s.value LIKE '%/worker/model_runner_v1.py(2089): propose_draft_token_ids'"):
            role='EARLY_PREFIX_SOURCE_ARM'if '(2069)'in r['value']else'SAMPLED_INDEX_SOURCE_ARM';roles[r['id']]=role;matched[r['id']].append(dict(stackDepth=r['stackDepth'],frame=r['value']))
        strings={r['id']:r['value']for r in db.execute('SELECT *FROM STRING_IDS')};selected={}
        for r in db.execute('SELECT rowid AS api_rowid,*FROM PYTORCH_API WHERE callchainId IS NOT NULL'):
            if r['callchainId']in roles and strings.get(r['name'])=='aten::cat':
                selected[r['api_rowid']]=dict(r,role=roles[r['callchainId']],name_text='aten::cat',matched_frames=matched[r['callchainId']],native_tasks=[])
        cached=native/f"db-{ref['database_index']:02d}.json";data=read(cached);defs={d['globalTaskId']:d for d in data['COMPUTE_definitions']}
        child_parent={}
        for parent_id,parent in selected.items():
            children=[a for a in data['linked_PYTORCH_APIs']if strings.get(a['name'])=='aclnnCat'and a['globalTid']==parent['globalTid']and int(parent['startNs'])<=int(a['startNs'])and int(a['endNs'])<=int(parent['endNs'])]
            assert len(children)==1,(ref['database_index'],parent_id,len(children))
            child=children[0];parent['exact_nested_native_API']=child
            assert child['api_rowid']not in child_parent;child_parent[child['api_rowid']]=parent_id
        for t in data['TASK_records']:
            for aid in set(t.get('native_linked_API_rowids',[]))&child_parent.keys():
                selected[child_parent[aid]]['native_tasks'].append(dict(TASK=t,definition=defs.get(t['globalTaskId'])))
        for r in selected.values():records.append(dict(database_index=ref['database_index'],**r))
        dbrefs.append(dict(**identity,database_index=ref['database_index'],cached_native=pin(cached),API_count=len(selected)))
        db.close()
    counts=Counter(r['role']for r in records);typed=Counter()
    for r in records:
        assert len(r['native_tasks'])==1 and r['native_tasks'][0]['definition']['opType_text']=='ConcatD'
        for t in r['native_tasks']:typed[t['definition']['opType_text']if t['definition']else'NO_COMPUTE']+=1
    write(O/'aux-sp-selector-profile-v131.json',dict(records=records,databases=dbrefs,API_counts_by_source_arm=dict(counts),native_TASK_types=dict(typed),
        scope='OnlyrecordedTorchcat producers oftarget_hidden_states at2069/2089. Positive sourcearm isnotactualIvalues,earlyconstancy,requestorbankidentity.',
        no_arm_excluded_by_zero_hits=True,actual_index_values_or_normalized_request_map_recovered=False,new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='AUX_SP_CHECKPOINT_SELECTOR_PROFILE_RECONCILED',graph_pairs=240,root_bound=192,unbound=48,eager_aux_unpaired=72,
        selector_API_counts=dict(counts),selector_native_types=dict(typed),new_device_work=False)))


if __name__=='__main__':main()
