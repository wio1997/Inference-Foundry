"""DP graph-mode/capture alignment analysis using saved cases and bounded dispatch source."""
from copy import deepcopy
from datetime import datetime,timezone
from itertools import product
from pathlib import Path
from math import lcm,ceil
import hashlib,json

from finalize_origin_coverage_v9 import O,W,read,write


def main():
    assert not(O/'REVIEW-CHECKPOINT-v46.json').exists()
    stamp=datetime.now(timezone.utc).isoformat();src=read(O/'M22-mode-padding-point-source-v46.json')
    assert hashlib.sha256(Path(src['path']).read_bytes()).hexdigest()==src['sha256']
    base=read(O/'padding-domain-review-v16.json');assert base['minimum_positive_uniform_TP_extent']==24
    config=read(O/'M21-captured-config-and-selected-dispatch-contract-v42.json')
    refs={k:{x:v for x,v in r.items()if x!='payload'}for k,r in src['historical_receipts'].items()}
    facts=[
        dict(grade='MEASURED_SAVED_PROFILE',statement='64exact phase-associated decode graphs:16roots scheduled2requests/12tokens,48roots4requests/24tokens; recorded targetM3×TP8 gives24physical rows. Other CPU eligibility counts and16unbound graphs are not silently assigned to this population.'),
        dict(grade='DOCUMENTED',statement='Uniform decode default requires initial state,per-request max scheduled width equal to uniform_decode_query_len,and num_tokens=max_width*num_reqs; DCP/hybrid/recompute adds done-prefilling guards. Explicit force_uniform_decode overrides this decision. Equal total tokens alone is insufficient.'),
        dict(grade='DOCUMENTED',statement='Token rows first TP-pad when DSA-CP or SP is enabled; FULL uniform descriptor must also be divisible by query length. For width6/TP8 this current interface requires positive multiples24. An unconstrained16-row replacement is not valid.'),
        dict(grade='DOCUMENTED_PLUS_DERIVED',statement='Worker _check_and_update_cudagraph_mode may pre-align capture sizes to LCM(query_len,TP) before backend-driven mode resolution,then passes resolver TP1. With guard true,width6/TP8,max24,the saved Engine[8,16,24]rounds to[24]. The actual registry is initialized from resolved capture descriptors,not directly from that Engine startup list.'),
        dict(grade='DOCUMENTED',statement='Backend/cache-group minimum graph support is aggregated over all builder classes/specs. Resolver may downgrade mode; source notes that prealignment can shrink/fail before downgrade. Actual final backend/mode/descriptor set remains unrecorded here; one cannot promote Engine startup list to live registry snapshot.'),
        dict(grade='DOCUMENTED',statement='Dispatcher normalizes uniform using separate_routine,prioritizes matching FULL key,then relaxed PIECEWISE key(uniformFalse,num_reqsNone),then NONE if allowed. Cascade/encoder flags exclude FULL; force_eager returns NONE but does not bypass subsequent DP coordination.'),
        dict(grade='DOCUMENTED',statement='Non-skipped DP synchronization uses CPU packed2×DP tensor with each rank writing its own column; default all_reduce sums disjoint columns. Token maximum and minimum graph mode are then computed explicitly. Global token count is max,not sum of DP counts; one NONE mode makes synchronized modeNONE.'),
        dict(grade='DOCUMENTED',statement='After coordination,graph replay,explicit draft,ALLGATHER/MC2,applicableFUSED_MC2-MegaMoe or finegrained TP forces uniform maximum runner rows. Otherwise eager independent rows may be kept. Redispatch is restricted to synchronized mode and asserts exact agreed padded token count.'),
        dict(grade='DOCUMENTED',statement='DP skip policy is recomputed per call: hierarchy or MegaMoe-supported first forces no-skip; dense context skips; ordinary MoE non-KV-consumer does not skip; KV consumer may skip only under remaining communication/recompute/mode conditions and no CANN MegaMoe. Do not memoize merely by config object id or infer skip from MC2 name.'),
        dict(grade='DOCUMENTED',statement='GraphParams main/draft are initialized from capture descriptors; a speculative config can initialize draft registry even if draft outer graph is eager. Registry existence is not evidence of actual draft graph execution.'),
        dict(grade='HISTORICALLY_STAGE_ATTESTED_SOURCE',statement='v16 padding-config immutable receipt contains utils.py SHA5f88fd... matching current frozen source,so the v44/v45 imported predicate source has historical166stage hash support. This is not a live backend,ENPU,registry or native-epoch snapshot.'),
    ]
    write(O/'M22-mode-padding-and-DP-contract-v46.json',dict(generated_utc=stamp,facts=facts,historical_receipt_provenance=refs,
        legacy_subset_hash_not_full_result_hash=True,source=['M22-mode-padding-point-source-v46.json','padding-domain-review-v16.json','M21-captured-config-and-selected-dispatch-contract-v42.json'],
        fixed_inputs=dict(TP=8,DP=2,speculative_tokens=5,uniform_query_width_domain=6,Engine_startup_capture_sizes=[8,16,24],max_capture=24),
        actual_final_worker_capture_descs=None,actual_DP_skip_branch=None,actual_rankwise_mode_values=None,
        all_live_optional_fields_bound=False,implementation=False,native_validation=False,critical_path_gain_us=None,whole_module_closed=False,overall_complete=False))

    # Pure integer model of the already source-defined constraints, not a real mode dispatch.
    aligned=sorted({ceil(n/24)*24 for n in [8,16,24]if ceil(n/24)*24<=24});assert aligned==[24]
    cases=[]
    for b in range(1,5):
        tokens=6*b;tp_padded=ceil(tokens/8)*8;full_uniform=ceil(tokens/lcm(6,8))*lcm(6,8)
        assert full_uniform%6==full_uniform%8==0
        cases.append(dict(requests=b,scheduled_tokens=tokens,TP_alone=tp_padded,FULL_uniform_TP=full_uniform))
    assert cases[1]==dict(requests=2,scheduled_tokens=12,TP_alone=16,FULL_uniform_TP=24)
    write(O/'H06-mode-aware-alignment-domain-proof-v46.json',dict(generated_utc=stamp,grade='DERIVED_SOURCE_DOMAIN',
        width=6,TP=8,lcm=24,initial_Engine_sizes=[8,16,24],max_capture=24,aligned_sizes_if_guard_true=aligned,request_cases=cases,
        FULL_uniform_16_invalid=True,TP_only_16_valid=True,
        implication='Query/padding domain reduction needs an actually supported nonuniform/PIECEWISE/ragged interface or separate carrier,not a16row replacement under unchanged FULLuniform6+TP8.',
        chronology='Engine startup list precedes worker prealignment/resolver/descriptor initialization; no final live descriptor set is claimed.',
        no_measured_padding_savings=True,implementation=False,whole_module_closed=False,overall_complete=False))

    dp_cases=[]
    for n0,n1,m0,m1 in product([8,16,24],[8,16,24],[0,1,2],[0,1,2]):
        packed=[[n0,n1],[m0,m1]];mode=min(packed[1]);maximum=max(packed[0]);after=[maximum]*2 if mode!=0 else list(packed[0])
        assert mode==0 if (m0==0 or m1==0)else mode>0
        if mode:assert after[0]==after[1]
        dp_cases.append(dict(tokens=[n0,n1],modes=[m0,m1],synced_mode=mode,example_no_extra_uniform_predicates=after))
    skip_cases=[]
    for hierarchy,mega_supported,moe,consumer,cann_mega,decode_mc2,prefill_mc2,separate,recompute in product([False,True],repeat=9):
        if hierarchy or mega_supported:skip=False
        elif not moe:skip=True
        elif not consumer:skip=False
        elif cann_mega:skip=False
        else:skip=decode_mc2 and ((prefill_mc2 and not separate)or recompute)
        if hierarchy or mega_supported or(moe and not consumer):assert not skip
        skip_cases.append(dict(hierarchy=hierarchy,mega_supported=mega_supported,moe=moe,consumer=consumer,cann_mega=cann_mega,decode_mc2=decode_mc2,prefill_mc2=prefill_mc2,separate=separate,recompute=recompute,skip=skip))
    assert len(dp_cases)==81 and len(skip_cases)==512
    write(O/'M22-DP-mode-padding-and-skip-domain-proof-v46.json',dict(generated_utc=stamp,grade='DERIVED_SOURCE_CASE_ALGEBRA',
        DP_example_valuations=dp_cases,DP_valuations=81,skip_boolean_valuations=skip_cases,skip_valuations=512,
        raw_column_sum_is_not_global_token_sum=True,forced_uniform_predicates_remain=['synced_mode!=NONE','is_draft','ALLGATHER/MC2','applicableFUSED_MC2+MegaMoe','finegrainedTP'],
        actual_global_case_pairing_not_reconstructed=True,mode_integer_examples_not_actual_native_epoch=True,
        no_latency_or_window_frequency_gain=True,implementation=False,whole_module_closed=False,overall_complete=False))

    # Exact original37 sites plus4 alignment helper sites, retaining every original site ID.
    sem=deepcopy(read(O/'M22-semantic-branch-inventory-v45.json'));new=[]
    extra={'NPUModelRunner._should_align_cudagraph_capture_sizes','NPUModelRunner._pad_for_sequence_parallelism'}
    notes={
        837:'DP1 exits; current savedDP2 does not by itself select skip branch.',840:'Skip decision returns replicated local count without DP collective; actual policy must be bound,not inferred from graph shape.',857:'FUSED_MC2 alone insufficient; CANN MegaMoe predicate controls uniformity.',861:'Any finegrained TP dimension>1 adds uniformity requirement.',872:'Uniform max when any graph/draft/comm/finegrained predicate holds; eager alone does not waive it.',873:'Preserve all OR terms including draft/MC2 even if graph modeNONE.',
        3343:'DCP/hybrid adds prefill-state checks.',3350:'PD recompute exception only for hybrid,non-DCP,enabled scheduler.',3351:'Keep all three recompute predicates.',3356:'Initial state and per-request done-prefilling both required.',3358:'Explicit force_uniform_decode overrides computed criterion.',3359:'Default uniform requires initial state,width equality,total=width*requests.',3368:'Encoder output can invalidate FULL.',3370:'Forced active-LoRA count differs from runtime request count.',3374:'Forced has_lora overrides derived boolean.',3378:'Force eager returns NONE but later DP synchronization still executes.',3386:'disable_full creates FULL exclusion; not a general no graph flag.',3390:'Cascade or encoder output disables initialFULL dispatch.',3392:'SP token alignment asserted after dispatcher.',3393:'Selected physical rows divisible by TP under SP.',3399:'DP>1 enters metadata coordination.',3406:'Non-None coordinated vector drives rank-specific padded count and redispatch.',3416:'Redispatch must exactly preserve agreed DP count.',3418:'Optional graph metrics observes unpadded/padded rows; no overhead/gain inferred from source presence.',
        6370:'Visit all backend sets paired with KV specs.',6373:'Each backend builder contributes capability.',6378:'Use minimum capability across groups,not first/most capable backend.',6385:'LCM prealignment conditional; happens before backend mode resolution.',6402:'Only eligible speculative/drafter algorithms get their own key initialization.',6403:'Speculative config+drafter+algorithm predicates all retained.',6406:'EAGLE or extract-hidden eligibility distinct from method namealone.',6410:'Required drafter interface types must match eligible branch.',6419:'Capture sizes derived from every resolved mode descriptor list.',6420:'Each descriptor contributes physical num_tokens; set deduplicates.',6425:'Only use_aclgraph initializes parameter maps.',6427:'Speculative config may initialize draft registry regardless of actual outer replay.',
        6358:'Alignment requires FULL decode mode plus CP/SP/pass-SP disjunction.',6359:'DSA-CP,SP and compiler pass-SP are distinct enabling predicates.',3278:'TP round-up if DSA-CP or SP; else preserve input count.'}
    counts={b['id']:0 for b in sem['branches']}
    ev=['M22-mode-padding-point-source-v46.json','M22-mode-padding-and-DP-contract-v46.json','H06-mode-aware-alignment-domain-proof-v46.json','M22-DP-mode-padding-and-skip-domain-proof-v46.json']
    for s in sem['static_branch_sites']:
        if s['owner']in extra:s['semantic_family']='M22-S04'
        counts[s['semantic_family']]+=1
        if s['semantic_family']=='M22-S04':
            new.append(dict(id=s['id'],owner=s['owner'],line=s['line'],condition=s['condition'],semantics=notes[s['line']],evidence=ev))
            s.update(review_status='EXACT_GUARD_LOCAL_DATAFLOW_RECONCILED',semantic_evidence=ev,runtime_binding='FIXED_PROFILE_DOMAINS_PLUS_EXPLICIT_UNBOUND_FLAGS')
    assert len(new)==41
    for b in sem['branches']:
        b['existing_static_site_count']=counts[b['id']]
        if b['id']=='M22-S04':b.update(state='41_MODE_DP_ALIGNMENT_GUARD_DATAFLOWS_LOCALLY_ANALYZED',
            scope='37original mode/DP/descriptor sites plus4 alignment-helper sites reclassified from adjacent source families.',
            gap='Actual final capture descriptors/DPskip/rankwise mode+count values and runtime optional overrides remain where decisive; all original IDs retained.',evidence=ev)
        if b['id']=='M22-S16':b['evidence']=list(dict.fromkeys(b['evidence']+ev[:2]));b['gap']='utils imported predicate has historical166stage SHA support; actual backend/ENPU/native update epochs and other external conditions remain.'
    sem.update(generated_utc=stamp,parent_inventory='M22-semantic-branch-inventory-v45.json');assert sum(counts.values())==1101
    write(O/'M22-semantic-branch-inventory-v46.json',sem)
    prior=deepcopy(read(O/'M22-original-scope-and-guard-reconciliation-v45.json'));oldids={r['id']for r in prior['guard_dataflow_reconciled']};assert not oldids&{r['id']for r in new}
    prior.update(generated_utc=stamp,parent_scope='M22-original-scope-and-guard-reconciliation-v45.json',guard_dataflow_reconciled=prior['guard_dataflow_reconciled']+new,local_reconciled_sites=110,other_static_sites_not_automatically_proved=991)
    write(O/'M22-original-scope-and-guard-reconciliation-v46.json',prior)
    write(O/'M22-mode-DP-guard-crosswalk-v46.json',dict(generated_utc=stamp,guard_dataflow_reconciled=new,original_S04_sites=37,alignment_helper_sites_reclassified=4,new_exact_local_guards=41,total_local_core_guards=110,whole_module_closed=False,overall_complete=False))

    variant=dict(id='H06-v46-MODE-AWARE-CAPTURE-ALIGNMENT',parent='H06',transformation='Evaluate backend-resolved graph mode before imposing FULL-uniform LCM constraints; nonuniform/PIECEWISE branches may retain TP-only capture domains when all native/metadata/key constraints support them.',
        eligibility='Conditional alternative; current fixed FULLuniform6/TP8 remains24. No setting-only16row replacement or measured fixed-case saving claimed.',implemented=False,native_validation=False,critical_path_gain_us=None)
    conditions=[
        'Keep FULLuniform query-width and TP divisibility; only genuinely supported nonuniform/PIECEWISE interface may relax their joint domain.',
        'Resolve minimum backend/spec support,mode and actual descriptor keys; startup Engine list is not final registry state.',
        'Preserve cascade/encoder/LoRA/force overrides and exact DP synchronized-mode redispatch.',
        'Retain DP skip eligibility and uniform MC2/MegaMoe/finegrained/draft carrier requirements; kernel name/config id is insufficient.',
        'Any separate local-query/global-carrier layout preserves row/slot/position/QSL/logits maps,rank participation and all capture bank/native epochs.',
        'No CPU-only eligibility or16/64window frequency becomes full-case frequency or linear gain; no fresh benchmark run.',
        'Preserve one-time full registry initialization from final capture descriptors and independent main/draft namespaces.',
        'Keep v44-v45 all-scalar,CPU host completion,workspace alias,ready/last-reader/feedback/exception contracts.'
    ]
    write(O/'H06-mode-padding-reorganization-conditions-v46.json',dict(generated_utc=stamp,new_conditional_variant=variant,conditions=conditions,
        refines_existing_H06_padding_domains=True,no_fixed_case_gain_proved=True,implementation=False,whole_module_closed=False,overall_complete=False))
    audit=deepcopy(read(O/'M22-original-requirement-completion-audit-v45.json'));audit.update(generated_utc=stamp,parent_audit='M22-original-requirement-completion-audit-v45.json',locally_reconciled_guard_sites=110,result='NOT_COMPLETE')
    for r in audit['requirements']:
        r['evidence']=list(dict.fromkeys(r['evidence']+ev+['M22-mode-DP-guard-crosswalk-v46.json']));r['completion_proved']=False
        r['note']='Local mode/DP/alignment source cases now answered; final worker keys and actual rankwise fields/native bank epochs/16phases plus relevant adjacent worker/resource/process scope remain.'
    write(O/'M22-original-requirement-completion-audit-v46.json',audit)
    print(json.dumps(dict(new_guard_dataflows=41,total_local_core_guards=110,original_sites=1101,semantic_families=16,DP_valuations=81,skip_valuations=512,conditional_new_H06_variant=1,variant_constraints=8,original8gates_open=True,whole_module_closed=False,overall_complete=False)))


if __name__=='__main__':main()
