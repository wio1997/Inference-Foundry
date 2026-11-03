"""Reconcile HC with draft input/context,proposal metadata and target sampling epochs."""
from pathlib import Path
from collections import Counter
from itertools import product
import hashlib,json,math,struct
from finalize_origin_coverage_v9 import N,O,W,read,write


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def schedule(p,early=False,fork=False,price=False):
    meta=p['metadata'];ctx=[];c=max(3,meta)
    for j in range(3):c+=p['context']+j+(p['extra']if price else 0);ctx.append(c)
    prev=(meta if early else ctx[-1])+p['embedding'];out=[]
    for j in range(3):
        y=prev+1;c=prev+p['coeff']+(p['extra']if price else 0)
        q=(y if fork else max(y,c))+p['query']+(p['query_penalty']if price else 0)
        attn=max(q,ctx[j])+p['attention']
        post=max(attn,c)+2
        prev=post+p['ffn'];out.append(dict(Y=y,coeff=c,Q=q,context=ctx[j],attention=attn,Post=post,FFNPost=prev))
    return dict(layers=out,proposal_ready=prev+3+2+p['head_markov'],context_bank_last_read=ctx[-1])


def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]


def main():
    assert not(O/'REVIEW-CHECKPOINT-v142.json').exists()
    native=read(O/'hc-draft-native-frontiers-v142.json');root=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend';parts=[]
    for rel,a,b in [('models/deepseek_v41/model.py',844,920),('models/deepseek_v41/model.py',1556,1565),('models/deepseek_v41/dspark.py',248,309),('models/deepseek_v41/dspark.py',314,332),('models/deepseek_v41/dspark.py',403,441),('spec_decode/dspark_proposer.py',231,259),('spec_decode/dspark_proposer.py',270,307),('spec_decode/llm_base_proposer.py',1040,1078),('spec_decode/llm_base_proposer.py',1476,1495),('spec_decode/llm_base_proposer.py',1577,1614),('worker/model_runner_v1.py',2020,2040),('worker/model_runner_v1.py',2084,2117)]:
        p=root/rel;parts.append(dict(**pin(p),first_line=a,last_line=b,source=p.read_text().splitlines()[a-1:b]))
    facts=[
      'Draft uses the shared decoder HC implementation but distinct draft embedding/weights/positions/cache and first premix. Target F01 collapsed Y, target normalized tail,three auxiliary means,combined main context,draft repeated embedding and draft raw final collapse are different values;H5120 alone cannot establish CSE.',
      'Target auxiliary values are means of the full residual entering configured layers,not their attention HCPre weighted Y and not final target normalized hidden. Existing24 eager roots/72aux anchors retain entering-layer39 mean before the final target block;no request association to later draft is promoted.',
      'Initial draft repeat produces four equal embedding channels and first premix is FP32 [1,0,0,0]. A finite value-domain collapsed-Y shortcut or broadcast residual representation may compose with early HC publication;coefficients remain data-dependent andfirstPost still needs the initial residual. Infinity*0 and native accumulation/rounding forbid unconditional real-algebra substitution.',
      'Context precompute uses combined target auxiliary context and distinct per-draft-layer Wkv/norm/C/S/slots. Draft query branch uses embedding then current layer HCPre/inputRMS. They may overlap before each SWA reads its required context/query cache ranges,under actual RAW/WAW/epochs/queue gates;one entire context publication barrier is not universally necessary.',
      'Current saved context cycles finish before initial repeat. Moving first draft HC/query preparation earlier is a conditional schedule/ABI change,not replacement of query input by context nor free compute reduction. Delaying context projection for joint context/query contraction is another version that extends context-bank readers;cannot combine its lifetime credit with early precompute retirement.',
      'Same-call contextWkv and queryWkv can use a common per-layer operator onlyafter their distinct inputs are ready and exact parameters/rows/rounds agree. No equality across three draft layers or target vsdraft weights is attested by source orprofile. Existing v130 recipient/hybrid alternatives are reused,not new primitive gains.',
      'Final draft HC collapse is BF16 weighted sum,not another HcPre or confidence HChead. Its unnormalized BF16 output remains confidence input in the optional branch;head RMS has an independent rounded value. Early logits or target raw-HC snapshot deletion does not prove raw draft collapsed output dead.',
      'Markov seed for current draft is copied from sampler-derived next_token_ids before input generation;five Markov steps later consume dependent IDs. Seed publication into output matrix after decoder is not creation of the seed value. Optional confidence after decoder/Markov cannot prune current noncausal decoder/metadata;it controls a future eligible verification prefix only.',
      'A following target verification produces a new accepted/rejected matrix;its literal next-ID/count depends on that outcome,not prior proposal keep. Current scheduled metadata/fullslot/CP/DP/mask/requestgeneration andnextacceptedstate mustnot share banks solelybyshape/graphbucket.',
      'F35 target head row demand can restrict selected final-layer queries onlybehind all-reader/state gates;earlierlayer HC/KV/state rows needed by those queriesremain. Current layer source query cut isnot an instruction to skip fullneededcachewrites/noQcollective participants.',
      'Raw source string tables have unresolved stack IDs in13DBs. Required selected positive frames/CIDs are retained and missing unrelated or selected frames are never fabricated. No absence-to-disabled/confidence/all-reader conclusion or global phase/functional promotion.',
      'Reference DAGs use three selected full-query SWAfrontiers and unchanged serial context services plus optional early embedding/HCfork. They are sufficient chosen schedules,not physical bounds/minimal wholeoperator barriers/actual native gain. Dummy None-metadata zero-output branch stays separate.'
    ]
    write(O/'hc-draft-source-contract-v142.json',dict(parts=parts,facts=facts,all_runtime_or_new_native_gates_closed=False,new_device_work=False,overall_complete=False))
    observed=[];geometry=Counter();unknownselected=0
    for r in native['records']:
        c=r['context_parent'];m=int(r['layers'][0]['stages']['HC']['definition']['inputShapes_text'].strip('"').split(',')[0]);geometry[c['context_rows'],c['query_KV_rows'],m,r['source_ancestry']]+=1
        selected=[r['repeat'],r['seed']]+list(r['collapse'].values())+[s for l in r['layers']for s in l['stages'].values()];unknownselected+=sum(f['value']is None for s in selected for f in s['selected_frames'])
        observed.append(dict(key=[r['database_index'],r['source_order_ordinal']],context_rows=c['context_rows'],query_rows=c['query_KV_rows'],local_M=m,ancestry=r['source_ancestry'],context_before_repeat=all(l['context_cycle']['endNs']<=r['repeat']['TASK']['startNs']for l in c['layers']),HC_before_SP=[l['stages']['HC']['TASK']['endNs']<=l['context_SP_parent']['draft_SP_allGather']['startNs']for l in r['layers']],collapse_after_last_FFN=r['layers'][-1]['stages']['FFN_POST']['TASK']['endNs']<=r['collapse']['CAST_IN']['TASK']['startNs'],actual_same_client_step_or_tensor_bank_bound=False))
    target=read(O/'hc-sp-eager-native-frontiers-v134.json');aux=[]
    for r in target['records']:
        if not r['aux_mean']:continue
        aux.append(dict(database_index=r['database_index'],forward_API_rowid=r['forward_API_rowid'],entering_layer=r['source_block_ordinal'],HC=r['stages']['HC'],mean=r['aux_mean'],allGather=r['aux_allGather'],input_residual_preceding_FFN=r['previous_FFN_Post'],actual_later_draft_request_bound=False))
    write(O/'hc-draft-observed-frontiers-v142.json',dict(records=observed,geometry=[dict(context_rows=c,query_rows=q,local_M=m,ancestry=a,n=n)for(c,q,m,a),n in sorted(geometry.items())],target_eager_aux_frontiers=aux,source_bound_target_aux=72,unresolved_selected_frame_references=unknownselected,raw_DBs_with_unresolved_stack_ids=sum(bool(x['unresolved_raw_stack_ids'])for x in native['databases']),not_all_source_stack_or_runtime_path_closed=True,new_device_work=False,overall_complete=False))
    pairs=[['F01',x]for x in ['F28','F29','F30','F31','F35']];contracts=[
      'Draft repeated embedding/onehot firstpremix/broadcast residual and early HCpublication compose onlyin actualfinite/typed/alias domains;targetY/context/embedding not interchangeable,coeffs/firstPost/fullstate remain.',
      'Draft HC/query can overlap already-ready targetcontext precompute with per-layer context/cache row deadlines. Context-query joint KV uses distinct inputs and explicit readiness/parameter/position/slot contracts;retirement vsdelayedprojection is version-dependent.',
      'Draft HcPost/collapse/head/Markov/confidence are separate values and epochs. Raw collapsedBF16 confidence input remains;later cutoff cannot prune its producing decoder. Markov recurrence/source seed and native rounds remain.',
      'HC input/premix/ID/mask/slot metadata belongs to current requestgeneration;sampling summary,newproposal keep andfollowingacceptedstate are distinct. Compatible producer publication may share work after full row/bank/queue contracts,not equal-shape reuse.',
      'HC query-demand/coeff fork andtarget sampling demand compose behind full KV/state and allreader cuts. Selected final-layer output rows do not erase earlier layer rows;exact ordered/duplicate head rows,raw scores,RNG,bonus andnextacceptedstate remain.'
    ]
    ex=[['HG93-HC-ATTENTION-FORK-DEADLINE',x]for x in ['F28','F29','F30','F31','F35']]+[['HG85-VERIFY-CONTEXT-DRAFT','F01'],['HG73-TAIL','F01'],['HG86-TERMINAL-QUERY-STATE-CUT','F31'],['HG86-TERMINAL-QUERY-STATE-CUT','F35']]
    write(O/'hc-draft-composition-v142.json',dict(clauses=[dict(id='REL142-0'+str(i+1),a=a,b=b,contract=c)for i,((a,b),c)in enumerate(zip(pairs,contracts))],new_pairs=pairs,new_pair_clauses=5,group_extensions=ex,new_groups=0,F01_star_conditional_pairs_after=34,whole_F01_or_pair_or_all_subsets_closed=False,all_runtime_gates_proved=False,speedup_additive=False,overall_complete=False))
    cases=[];classes=Counter();hidden=Counter()
    for vals in product([2,15],[1,24],[0,12],[1,5],[2,11],[2,9],[3,18],[0,14],[1,8],[0,5,30]):
        p=dict(zip(['coeff','context','metadata','embedding','query','attention','ffn','query_penalty','head_markov','extra'],vals));ans={k:schedule(p,e,f,pr)for k,e,f,pr in [('baseline',False,False,False),('HC_fork_only',False,True,False),('early_draft_only',True,False,False),('joint_ideal',True,True,False),('joint_repriced',True,True,True)]};b=ans['baseline']['proposal_ready'];v=ans['joint_repriced']['proposal_ready'];classes['FASTER'if v<b else'SLOWER'if v>b else'TIED']+=1
        single=b-ans['HC_fork_only']['proposal_ready']+b-ans['early_draft_only']['proposal_ready'];joint=b-ans['joint_ideal']['proposal_ready'];hidden['OVERLAPPING_CREDITS'if joint<single else'SYNERGY'if joint>single else'EQUAL']+=1
        cases.append(dict(parameters=p,**ans))
    write(O/'hc-draft-readiness-DAGs-v142.json',dict(cases=cases,count=len(cases),classes=dict(classes),ideal_interactions=dict(hidden),endpoint='Current draft proposal IDs;following target execution,sampling and metadata clocks not included',assumptions='Selected three-layer positive-query branch only,arbitrary services;metadata includes current seed/input/slot/mask readiness. Context services serial as current,early embedding independent onlyunderlegalnew schedule. Query/cache/SWA reference waits context demanded by its chosen fullfrontier;perrange pipeline may be finer. Original shared coefficients/Post/FFN/pre4 androunds retained. Loaded/native gain/physical optimum not proved.',new_device_work=False,overall_complete=False))
    # Small explicit witnesses: algebra,sourcevalue identity,epoch anddemand distinctions.
    def dot_seq(w,e):
        s=0.
        for a,b in zip(w,e):s=f32(s+f32(a*b))
        return s
    weights=[[2**24,1],[-2**24,0],[0,0],[0,0]];fold=[sum(w[k]for w in weights)for k in range(2)];original=dot_seq([a for w in weights for a in w],[1.,1.]*4);prefold=dot_seq(fold,[1.,1.]);assert original==0 and prefold==1
    nodes=['SAMPLED0','NEXT0','META0','DRAFT_HC','DRAFT_ATTN','DRAFT_COLLAPSE','MARKOV','CONF','KEEP','META1','VERIFY1','SAMPLED1','NEXT1'];edges=[['SAMPLED0','NEXT0'],['NEXT0','META0'],['META0','DRAFT_HC'],['DRAFT_HC','DRAFT_ATTN'],['DRAFT_ATTN','DRAFT_COLLAPSE'],['DRAFT_COLLAPSE','MARKOV'],['DRAFT_COLLAPSE','CONF'],['MARKOV','CONF'],['CONF','KEEP'],['KEEP','META1'],['META1','VERIFY1'],['VERIFY1','SAMPLED1'],['SAMPLED1','NEXT1']]
    write(O/'hc-draft-value-epoch-and-demand-witnesses-v142.json',dict(
      raw_confidence_not_normalized=dict(raw=[1.,2.],epsilon=1e-6,BF16_normalized=[1.,1.],constructed_confidence_scores=[1.,2.],frozen_confidence_weights_or_active_branch_bound=False),
      mean_vs_collapsed=dict(X=[1,2,3,4],premix=[1,0,0,0],HC_Y=1,aux_mean=2.5,main_context_equals_either=False),
      finite_first_Y=dict(embedding=[2,-3],repeat=[[2,-3]]*4,premix=[1,0,0,0],Y=[2,-3],coefficients_or_full_first_Post_dead=False),
      nonfinite_first_Y=dict(embedding='+Inf',onehot_products=['+Inf','NaN','NaN','NaN'],literal_collapse='NaN',shortcut='+Inf',unconditional_equality=False),
      coefficient_prefold_round=dict(weights=weights,E=[1.,1.],original_sequential_FP32=original,real_prefold_weights=fold,prefold_FP32=prefold,loaded_HF32_or_native_accumulation_attested=False),
      selected_query_needs_other_KV=dict(original_causal_query_position=1,query=0,keys=[1,2],values=[2,10],selected_query_output=6,wrong_remove_unselected_KV_output=10,earlier_HC_rows_deletable=False),
      same_shape_wrong_epoch=dict(old=dict(request='A',generation=1,positions=[10,11],slots=[0,1],mask=[True,False]),new=dict(request='B',generation=2,positions=[20,21],slots=[0,1],mask=[False,True]),shape=[2],metadata_reuse_legal=False),
      bank_retirement=dict(old_value=1,new_value=3,old_reader_time=8,new_write_time=5,old_reader_observes_after_unsafe_reuse=3,expected_old_reader_value=1,last_reader_and_queued_writer_contract_required=True),
      proposal_not_accepted=dict(seed=7,proposal=[10,11],all_accepted_sampled=[7,10,11],early_reject_sampled=[7,99,-1],all_accepted_next=11,reject_next=99,keep_count=2,next_ID_determined_by_keep=False),
      generation_graph=dict(nodes=nodes,edges=edges,illegal_future_cutoff_edge=['KEEP','META0'],cycle=['META0','DRAFT_HC','DRAFT_ATTN','DRAFT_COLLAPSE','CONF','KEEP','META0'],optional_confidence_not_runtime_attested=True),
      witness_scope='Constructed finite/reference FP32,source literal or abstract epoch counterexamples;not frozen actual tensor values/new native parity.',new_device_work=False,overall_complete=False))
    work=[]
    for(c,q,m,a),n in sorted(geometry.items()):
        work.append(dict(context_rows=c,query_rows=q,local_M=m,ancestry=a,n=n,H=5120,global_repeat_value_content=8*q*5120,local_repeat_value_content=8*m*5120,local_embedding_value_content=2*m*5120,full_first_residual_obligation=True,
            HC_coeff_classical_matrix_reference_per_layer=2*m*24*(4*5120),unchanged_current_context_three_KV_matrix_reference=3*2*c*5120*512,early_HC_schedule_matrix_work_credit=0,
            raw_draft_collapsed_global_value_content=2*q*5120,raw_draft_confidence_observer_closed=False,context_bank_content=2*c*5120,context_bank_retirement='After actual last context reader;delaying perlayer projection extends it. EarlyHC alone withunchangedprecompute doesnotmovecontext reader end earlier.',
            physical_HBM_or_allocation_or_time=None))
    write(O/'hc-draft-work-and-lifetimes-v142.json',dict(geometry=work,scope='Value content/classical matrix references only within declared dense/direct implementation. No physical necessary HBM/actual allocation or time gain. Early scheduling deletes0 matrixwork;broadcast/onehot specialization needs separate typed nativeABI.',bound_contract='Only proven required work divided by matching valid hardware capacity upper establishes L;combine nonduplicate resource lower bounds by max with actual dependency/streaming model. Actual PMU/capacities/full allreader domains unbound.',new_device_work=False,overall_complete=False))
    names=['hc-draft-native-frontiers-v142.json','context-sp-profile-frontiers-v130.json','context-sp-source-contract-v130.json','context-sp-joint-graph-and-conditions-v130.json','draft-cache-join-source-contract-v113.json','draft-cache-order-and-witnesses-v113.json','M19-layout-ownership-and-context-query-reorganization-v35.json','DSpark-tail-reader-closure-v85.json','DSpark-tail-reorganization-variants-v85.json','terminal-layer-demand-cut-v86.json','verify-metadata-source-contract-v129.json','verify-metadata-value-and-epoch-witnesses-v129.json','verify-metadata-joint-contract-v129.json','sampling-operator-guard-contract-v102.json','sampling-visibility-profile-projection-v117.json','heads-sampling-native-chains-v65.json','hc-sp-eager-native-frontiers-v134.json','candidate-relation-clauses-v141.json','hc-moe-postcheck-result-v141p.json']
    write(O/'hc-draft-inputs-v142.json',dict(inputs=[pin(O/n)for n in names],questions=['Which target/draft HC,aux,context,embedding andtail values share a legitimate operator boundary?','Can currentdraft HC/query overlap context production without changing cache and first-residual obligations?','Does later Markov/confidence/acceptance justify pruning currentHC/metadata or reusing same-shape banks?','Which fullKV/aux/rawconfidence/score/RNG/readers cut backward propagation of target sampling demand?'],profile_first=True,new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='MODELED',draft_frontiers=104,draft_HC_layers=312,TASKs=2496,target_aux=72,DAGs=len(cases),classes=classes,interactions=hidden,unresolved_selected_frames=unknownselected,new_pairs=5,overall_complete=False)))


if __name__=='__main__':main()
