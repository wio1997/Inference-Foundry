"""F30-F31 generation, row-demand and publication composition, profile first."""
from pathlib import Path
import ast,hashlib,json
from finalize_origin_coverage_v9 import N,O,W,read,write
from verify_metadata_relation_models_v129 import publish_prefixes,prefix_layout,next_count,topk_counts,enumerate_layouts,enumerate_ties


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def segment(p,a,b):
    return dict(**pin(p),first_line=a,last_line=b,source=p.read_text().splitlines()[a-1:b])


def main():
    assert not (O/'REVIEW-CHECKPOINT-v129.json').exists()
    root=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend'
    prop=root/'spec_decode/llm_base_proposer.py';util=root/'spec_decode/utils.py';worker=root/'worker/model_runner_v1.py'
    expected={prop:'458fb50e6839f3a353ddc7181ace4fdae02e35093d8ccd1c4327dc24ae5fff31',util:'94a569fa2b54fcd206d6992c29e13a852ee5f6596d6d483fd4a33ca182c3f1f7',worker:'568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c'}
    for p,h in expected.items():assert pin(p)['sha256']==h
    scope=read(O/'verify-metadata-profile-scope-v129.json')
    absent=['DYNAMIC_SCHEDULER_SOURCE_SURFACE','KEEP_LENGTH_RETRIEVAL_SOURCE_SURFACE','CONFIDENCE_WRAPPER_SOURCE_SURFACE','CONFIDENCE_HEAD_SOURCE_SURFACE']
    assert all(scope['API_role_counts'].get(k,0)==0 for k in absent)
    assert scope['TASK_role_counts']['SEED_PUBLICATION']==104 and scope['TASK_role_counts']['ID_PUBLICATION']==520
    write(O/'verify-metadata-source-contract-v129.json',dict(parts=[
        segment(prop,1040,1060),segment(prop,1220,1229),segment(prop,1354,1399),segment(prop,1476,1495),segment(prop,1577,1617),segment(prop,1637,1645),segment(prop,2211,2252),
        segment(util,450,505),segment(util,505,570),segment(util,570,635),segment(worker,2028,2040),segment(worker,2077,2117),segment(worker,2205,2226)],
        facts=[
            'Selected padded runner path computes next-ID/count from the CURRENT target sampled matrix before preparing inputs and calling drafter._propose. Its count is not the keep length of newly generated proposals.',
            'Merged draft builds its first-pass inputs and executes its decoder before generating Markov IDs and calling the optional confidence scheduler. No current decoder-input metadata can depend causally on that later result.',
            'Confidence returns per-request keep counts from a GLOBAL TopK request histogram. Retrieval publishes a PREFIX of each token list using clamped counts and zip, not the exact positions chosen by TopK.',
            'In an eligible pure-decode next verification batch with matching row order/generation,actual returned prefix length ell gives query length ell+1;raw k alone needs cap/actual-width/scheduler-row-map conditions.',
            'Target acceptance/rejection later produces a new sampled matrix and a separate literal validity count. The count predicate accepts values<-1 belowV and gathers at count-1;holes/rightmost-valid shortcuts remain invalid.',
            'num_speculative_tokens0 returns before scheduler update;disabled/None scheduler,changed widths,B/request order and zip mismatch are separate applicability branches. Do not reuse a stale keep bank by equal shape.',
            'Metadata callbacks/banks/frontiers already have asynchronous input/reuse/readiness protocols. Composition may share compatible value production;it cannot erase zero-query effects or publish partly overwritten banks.'],
        opaque_provider_contract='M21-count-metadata-and-output-publication-contract-v37.json',
        actual_dynamic_spec_or_full_input_signature_attested=False,whole_module_closed=False,new_device_work=False,overall_complete=False))
    layouts=enumerate_layouts();ties=enumerate_ties()
    witnesses=[]
    for k in range(6):
        outcomes=[]
        for accepted in range(k+1):
            row=list(range(accepted+1))+[-1]*(k-accepted)
            nxt,c=next_count(row,99,100)
            assert c==accepted+1
            outcomes.append(dict(accepted_drafts=accepted,sampled=row,next_ID=nxt,valid_count=c))
        _,discard=next_count([0]+[-1]*k,99,100,True);assert discard==0
        witnesses.append(dict(proposal_keep_length=k,target_query_length=k+1,possible_target_valid_counts=list(range(k+2)),outcomes=outcomes,discard_valid_count=discard))
    selected=topk_counts([[4,4,4,4]],1,1,[2]);assert selected==[2]
    assert publish_prefixes([[10,11,12,13]],selected,4)==[[10,11]]
    prefix_not_indices=dict(survival=[[1,1,1,1]],min_k=1,chosen_flat_candidate_indices=[2],selected_draft_positions=[0,3],published_keep=[2],published_draft_prefix=[10,11],not_native_observed_values=True)
    k=topk_counts([[4,3,2],[4,1,1]],1,2,[0,1]);assert k==[3,1]
    qsl,owner=prefix_layout(k);uniform_qsl,uniform_owner=prefix_layout([2,2]);assert qsl!=uniform_qsl and len(owner)==len(uniform_owner)==6
    reordered=prefix_layout(list(reversed(k)))[0];assert reordered!=qsl
    cap=publish_prefixes([[1,2],[3,4,5]],[5,0],5);assert list(map(len,cap))==[2,0]
    mismatch=publish_prefixes([[1,2],[3,4]],[1],5);assert mismatch==[[1]]
    hole=next_count([7,-1,9],99,100);negative=next_count([-2,5,-1],99,100)
    assert hole==(-1,2)and negative==(5,2)
    write(O/'verify-metadata-value-and-epoch-witnesses-v129.json',dict(
        proposal_keep_vs_target_acceptance=witnesses,TopK_selection_vs_prefix=prefix_not_indices,
        same_total_different_row_layout=dict(keep=k,qsl=qsl,owner=owner,uniform_qsl=uniform_qsl,uniform_owner=uniform_owner,reordered_qsl=reordered),
        actual_list_width_limits_keep=dict(tokens=[[1,2],[3,4,5]],keep=[5,0],published=cap),
        zip_length_mismatch=dict(token_rows=2,keep_rows=1,published=mismatch,source_behavior_only_not_current_bug=True),
        literal_next_ID_witnesses=dict(hole_row=[7,-1,9],hole_result=list(hole),negative_row=[-2,5,-1],negative_result=list(negative)),
        saved_same_shape_or_total_does_not_bind_request_parameter_or_bank_epoch=True,
        native_flags_or_tie_precision_or_runtime_values_attested=False,new_device_work=False,overall_complete=False))
    write(O/'verify-metadata-layout-models-v129.json',dict(cases=layouts,case_count=len(layouts),
        new_ABI='INT32 observer-visible k[B],INT32 query_start_loc[B+1],optional INT32 token-owner[Q];puredecode materialized prefixes only.',
        scope='0<=B<=4,0<=D<=5,allprefix vectors. Required logical content under this chosen ABI,not measured allocation/HBM or a current device scan baseline.',
        byte_formula='4B+4(B+1)+4Q, Q=sum(ell_i+1). k remains observable;only a hypothetical separate device reread4B can be avoided by a compatible final producer.',
        input_order_rule='Map by full request+generation+scheduled row identity,then apply actual token-list width and consumer cap. Matching totalQ or graphbucket is insufficient.',
        actual_current_device_k_to_qsl_kernel_or_read_cost_established=False,critical_path_gain_us=None,new_device_work=False,overall_complete=False))
    write(O/'verify-metadata-tie-reference-validation-v129.json',dict(**ties,
        interpretation='An allowed equal-score TopK position set can differ from the published prefix. Metadata must use the completed request counts and consumer trim,not raw selected positions.',
        all_native_TopK_choices_proved=False,new_device_work=False,overall_complete=False))
    nodes=[('Y0','Current target sampled IDs'),('S0','Literal next-ID andvalid-count summary'),('P0','Input/rejection/position preparation'),('M0','Current draft input/attention metadata'),('D0','Full selected noncausal draft decoder'),('L0','Basehead andMarkov recurrence'),('C0','Optional confidence rawhidden/orderedembedding'),('B0','Periodic budget state update'),('K0','Global selection/request histogram keepcounts'),('T0','Host retrieval cappedprefix/request IDs'),('A1','Next scheduled rowidentity andprefixmetadata'),('V1','Next target verification/sample result'),('S1','Next accepted-state next-ID/count summary')]
    edges=[('Y0','S0','RAW sampled values'),('S0','P0','next-ID/count,rejection correction'),('P0','M0','positions/lengths/slots'),('M0','D0','input ready/frontiers'),('D0','L0','hidden/query rows'),('D0','C0','rawhidden'),('L0','C0','ordered Markov inputs'),('C0','B0','survival andperiodic state'),('B0','K0','final current budget'),('C0','K0','global candidate scores'),('K0','T0','samegeneration keepbank'),('L0','T0','draft ID lists'),('T0','A1','returnedprefix+scheduledrowmap'),('A1','V1','metadata ready/targetinput'),('V1','S1','new sampled/rejection outcome')]
    graph_edges=[dict(producer=a,consumer=b,dependency=c)for a,b,c in edges]
    next(e for e in graph_edges if (e['producer'],e['consumer'])==('C0','B0')).update(
        value_read_condition='Periodic budget recompute consumes survival;otherwise the known budget/counter path does not require a new survival-value wait.')
    write(O/'verify-metadata-generation-graph-v129.json',dict(nodes=[dict(id=i,label=t)for i,t in nodes],edges=graph_edges,
        domain='Selected padded DSpark merged first-pass/early-return domain plus optionaldynamic branch and eligible puredecode next scheduled prefix. Nonparallel autoregressive continuation remains a separate branch. Crossiteration nodeindices are symbolic,not captured request binding.',
        forbidden_rewrites=['C0/K0cannotremoveM0orD0 by a later verification cutoff.','K0isnotS0valid_count anddoesnotpredictS1accepted_count.','RawTopKposition set isnotT0publishedprefix.','A1cannotbindbyshape/totalalone orreusebankbeforeoldreadersretire.'],
        readiness='Only value/range dependencies shown. No new whole-operator/global barrier is inferred;finalize each range when its prerequisite counts/cutoffs and reader retirement are proven.',
        native_graph_nodes_added=0,new_central_pair_reconciliation=True,overall_complete=False))
    write(O/'verify-metadata-joint-contract-v129.json',dict(pair=['F30','F31'],
        relation='CROSS_ITERATION_PREFIX_DEMAND_AND_ACCEPTED_STATE_ARE_DISTINCT;CONDITIONAL_KEEP_TO_METADATA_COMPOSITION',
        applicable_domains=[
            'Same proposal:retain current draft metadata/noncausal input rows;confidence can reuse already-known embeddings under existingF30conditions,but cannot retroactively trimdecoder.',
            'Next target puredecode batch:compose finalrequestkeepcounts/actualprefixwidths with compatible scan/owner/metadata producers onlyafter fullrowidentity/attrs/capture/bank contract isbound.',
            'Following acceptedstate:keep sampler-derived literal next-ID/count/rejection snapshot distinct;existingF31summaryfusion remains its own opportunity.',
            'Disabled/None/draftwidth0/variablewidth/periodicbudget/reorderedrequests/ties/NaNs/RNG/fullgraph/zeroquery branches stay explicitly gated.'],
        reuse_conditions=['Actualdynamic flag andsame B/D/materializedwidth/consumer cap domain.',
            'PreserveFP32confidence/sigmoid/cumprod andperiodiccounter/budget timing,globalTopKties/requesthistogram andRNG/scoreobservers.',
            'RequestID+generation+scheduledrowmap,CP/DP/group/seq/page/slot/layout/mask input identities;opaque SMLA/QLI planpurity andout ABI remainunproven.',
            'Keep k andacceptedcount dtypes/values/observers separate;host retrieval/int(k)/zip/error effects remain untilreplacedbyanexplicitcompatible interface.',
            'Respectpersistentbank oldreaders,queuedwriters,inputready/frontier/capture/update epochs andzeroquery writes. Do not invent a globalbarrier orindependent savings.'],
        costs='Only newABI logical content/elementwise dependency cuts. No present device scan baseline,HBMbytes,latency saving,wholedecoderpruning,physicalcapacity orMFU proved.',
        existing_groups_refined=['HG73-TAIL','HG85-VERIFY-CONTEXT-DRAFT'],new_main_candidates=0,
        new_pair_clauses=1,new_higher_order_groups=0,original_pending_promotions=0,new_device_work=False,overall_complete=False))
    inputs=[O/'native-store-postcheck-result-v128p.json',O/'candidate-pair-navigation-v128.json',O/'candidate-relation-clauses-v128.json',
        O/'verify-metadata-profile-scope-v129.json',O/'M20-Markov-confidence-and-verification-contract-v36.json',O/'F30-dynamic-verify-budget-and-prefix-bounds-v36.json',
        O/'M21-count-metadata-and-output-publication-contract-v37.json',N/'M20-step-audit166-v36.json',N/'M20-exact-profile166-v36.json',
        N/'verify-metadata-profile-run-v129.json',N/'query_verify_metadata_profile_v129.py']+list(expected)
    write(O/'verify-metadata-inputs-v129.json',dict(inputs=[pin(p)for p in inputs],questions=[
        'Can the later proposal confidence cutoff remove current draft metadata/decoder rows?',
        'Are confidencekeepcounts,TopKpositions andsampleraccepted-count/next-ID interchangeable?',
        'Underwhich request/width/epoch/globalbudget/frontier conditions could finalkeep publication compose with next verificationmetadata?',
        'What logical work/content follows under a declarednewABI without inventing currentbaseline savings?'],
        profile_first=True,no_framework_wide_survey_or_new_source_download=True,new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='F30_F31_CONDITIONAL_RELATION_RECONCILED',layout_cases=len(layouts),tie_candidate_subsets=ties['candidate_subsets'],valid_tie_subsets=ties['valid_TopK_subsets'],nonprefix_position_cases=ties['nonprefix_selected_position_cases'],new_pairs=1,new_device_work=False)))


if __name__=='__main__':main()
