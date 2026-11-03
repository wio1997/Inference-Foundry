"""Reconcile staged operator evidence with prior v11, without replaying target work."""
from copy import deepcopy
from pathlib import Path
import hashlib,json
from finalize_origin_coverage_v9 import N,O,read,write

GUARDS=[
    'Direct Ascend v1 DSpark drafter-content question was answered at v11; it does not close all snapshot observers.',
    'No additional registered/external/native content observer may require the removed snapshot or final Post output; this is still an open applicability guard.',
    'Preserve exact active implementation/method, parameter epochs, output aliases, graph buffer addresses and last-reader/event/lifetime contracts; loaded historical bytes are not attested.',
    'For no-buffer/localization, also preserve all normalized-output readers, TP group/order/pad/trim/readiness and native changed-shape reduction/rounding policy.'
]

def receipt(path):
    b=path.read_bytes()
    return dict(path=str(path),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def main():
    out=O/'DSpark-tail-prior-evidence-reconciliation-v85.json'
    assert not out.exists() and not(O/'REVIEW-CHECKPOINT-v85.json').exists()
    oldpath=O/'F26-consumer-contract-v11.json';installed=N/'installed-fullhc-consumers166-v11.json'
    old=read(oldpath);result=json.loads(read(installed)['stdout'])['result']
    assert len(result['matches'])==90 and len(result['matched_files'])==26
    assert old['search']['matches']==90 and old['search']['matched_files']==26
    src=read(O/'DSpark-tail-source-index-v85.json')
    pins={r['archive_member']:r['sha256']for r in src['source_files']}
    assert all(pins[k]==v for k,v in old['source_sha256'].items())
    ledger=read(O/'candidate-profile-ledger-v84.json')
    prior=next(c for c in ledger['candidates']if c['id']=='F26')['scoped_detail']['raw_snapshot_consumer_refinement_v11']
    assert prior['scoped_answer']==old['scoped_answer'] and prior['not_global_dead_state']
    files=['DSpark-tail-source-index-v85.json','DSpark-tail-reader-closure-v85.json',
        'DSpark-tail-reorganization-variants-v85.json','DSpark-tail-row-local-proof-v85.json',
        'DSpark-tail-work-communication-models-v85.json','DSpark-tail-transfer-credit-v85.json',
        'DSpark-tail-composition-clauses-v85.json','tail-shared-memory-budget-v85.json',
        'tail-shared-memory-relations-v85.json']
    before={name:read(O/name)for name in files}
    write(O/'DSpark-tail-pre-reconciliation-v85.json',dict(
        status='SUPERSEDED_STAGED_DRAFTS_NOT_CURRENT_CONCLUSIONS',
        reason='Draft scope/novelty overstated the direct-reader fact. Immutable v11 already settled it; all-observer guards remain open.',
        artifacts=before,original_receipts=[receipt(O/n)for n in files],overall_complete=False))
    src['prior_installed_symbol_inventory']=dict(input_receipts=[receipt(oldpath),receipt(installed)],
        matches=90,matched_files=26,roots=old['search']['roots'],partial_frozen_inventory_is_not_a_replacement=True,
        alternative_paths_to_preserve=old['other_paths_to_preserve'],limits=result['limits'])
    src['frozen_reference_inventory_scope']='4 files /10 references within partial frozen snapshot; not full installed-source coverage or a new all-reader proof.'
    write(O/files[0],src)
    d=before[files[1]]
    d.update(decision='PRIOR_V11_SCOPED_DIRECT_DRAFTER_CONTENT_ANSWER_REVALIDATED',
        prior_direct_drafter_question_established_v11=True,all_snapshot_observers_closed=False,
        new_full_dead_buffer_proof=False,remaining_observer_guards=GUARDS,
        raw_buffer_source_value_read_in_this_pipeline_scope='Only the inspected Ascend v1 DSpark direct drafter argument; not registered/native/external observers.',
        other_paths_to_preserve=old['other_paths_to_preserve'])
    d['proof'][1]='Model copies pre-collapse HC. Current partial frozen snapshot has4 exact symbol-reference files; prior installed search already found90 matches/26files. The inspected Ascend v1 getter result is loaded only in method==mtp guarded rebind.'
    d['proof'][-1]='Normalized target output,3checkpoint means and draft raw-confidence input remain live. Final Post output is private ONLY IF all remaining snapshot/output observers and alias/event/lifetime guards are discharged.'
    write(O/files[1],d)
    d=before[files[2]];d.update(source_observer_question_closed=False,
        direct_drafter_content_question_previously_answered_v11=True,all_snapshot_observers_closed=False,
        remaining_observer_guards=GUARDS)
    origins=['v11 F26a: copy-only conditional variant','v11 F26b: existing bufferNone branch',
        'Prior local-normalizer commutation review (v65); refined typed costs here',
        'v85 conditional final Post/private-collapse boundary',
        'v85 conditional last-merge/Post/collapse/norm transfer union']
    assert len(d['variants'])==5
    for row,origin in zip(d['variants'],origins):
        row['origin_and_novelty']=origin
        row['conditions']=GUARDS+row['conditions']
        row['all_snapshot_observers_closed']=False
    d['variants'][0]['state']='PRIOR_V11_DIRECT_CONTENT_ANSWER_REVALIDATED_COPY_ONLY_GUARDS_OPEN'
    d['common_conditions']=GUARDS+d['common_conditions']
    write(O/files[2],d)
    for name in files[3:]:
        d=before[name];d['remaining_raw_observer_and_lifetime_guards']=GUARDS
        d['all_snapshot_observers_closed']=False
        d['prior_direct_drafter_content_answer_v11']=True
        if name=='DSpark-tail-work-communication-models-v85.json':
            d['comparison_graphs'][1]='G0S omit snapshot copy only IF remaining observer/lifetime guards hold'
        if 'clauses'in d:
            for c in d['clauses']:
                c['remaining_observer_guards']=GUARDS
                c['all_snapshot_observers_closed']=False
                c['prior_evidence']='F26-consumer-contract-v11.json; direct-content answer preexisting, not all-observer closure'
            for g in d.get('higher_order_groups',[]):g['remaining_observer_guards']=GUARDS
            if name=='DSpark-tail-composition-clauses-v85.json':
                d['clauses'][0]['relation']='FINAL_POST_PRIVATE_ONLY_IF_REMAINING_OBSERVER_GUARDS_HOLD'
                d['refine_existing_relations']=[s.replace('rawHCsourceobserver closed only in fixedDSpark',
                    'direct drafter-content answer preexistingv11; all-observer guards remain open')for s in d['refine_existing_relations']]
        write(O/name,d)
    write(out,dict(status='PRIOR_EVIDENCE_RECONCILED_NO_NEW_ALL_OBSERVER_CLOSURE',
        immutable_prior_input_receipts=[receipt(oldpath),receipt(installed)],
        prior_scope=old['scope'],prior_scoped_answer=old['scoped_answer'],prior_answer_first_established_version=11,
        current_four_historical_source_hashes_match=True,prior_installed_symbol_search=[90,26],
        current_partial_frozen_reference_inventory_files=4,
        old_conditional_observer_guards_were_not_disproved=True,other_paths_to_preserve=old['other_paths_to_preserve'],
        remaining=old['remaining'],remaining_observer_guards=GUARDS,
        genuinely_new_analysis=['80last-layer graph frontiers rejoined to baseline104tails/208AGs',
          'Conditional final Post/private-collapse transfer union and typed work/communication domains',
          'One common memory-budget remainder for raw-HC capacity credit, WKV memo and private C2 state'],
        no_new_full_dead_buffer_proof=True,source_read_is_not_new_native_evidence=True,overall_complete=False))
    # Unavoidable service payload only in the explicitly fixed boundary/representation domain.
    # No physical HBM, topology speedup or hardware rate is inferred from logical values.
    work=read(O/'DSpark-tail-work-communication-models-v85.json');models=[]
    for row in work['models']:
        m,h,H,p=row['M'],row['h'],row['H'],row['P']
        q=2*m*(h+1)*H+4*m*h+2*H
        models.append(dict(M=m,h=h,H=H,P=p,
            fused_local_collapse_norm_boundary_value_payload_bytes=q,
            optional_legacy_rstd_output_bytes=4*m,
            explicit_rstd_reference_total_bytes=q+4*m,
            baseline_ring_send_reference_bytes=row['ring_reference_raw_HC_send_bytes']+row['ring_reference_premix_send_bytes'],
            localized_ring_send_reference_bytes=row['ring_reference_collapsed_or_normalized_send_bytes'],
            actual_numeric_time_bound_us=None))
        assert q+4*m==row['fused_localCollapseNorm_noncollective_value_reference_bytes']
    write(O/'DSpark-tail-ceiling-domains-v85.json',dict(models=models,
        payload_domain='Generic dense Rbf16[M,h,H],premixFP32[M,h],gammaBF16[H] available only via declared GM boundary; every normalized BF16[M,H] output materialized for the retained collective. Count each unique boundary value once; gamma once, no mandatory rstd output.',
        payload_formula='Qboundary=2M(h+1)H+4Mh+2H. Fixed representation/evaluation boundary; not a universal algorithm or physical HBM lower bound. Post+collapse fusion changes this boundary and must be modeled separately.',
        capacity_bound='If GM service must carry these boundary values and B_GM is a valid aggregate service capacity upper, T>=Qboundary/B_GM. Separate read/write capacity constraints and Vector/cast/reduction/rsqrt envelopes can strengthen it through max, not unconstrained summation.',
        attainability='A lower bound on time is an upper bound on attainable throughput only in its stated domain; it does not establish that the bound is reachable. Current mixed-resource capacities are unbound.',
        baseline_reference_is_not_a_ceiling='Source-like value traffic and conventional FP32 operation counts are comparison references. Eliminable transfer unions are conditional interfaces, not measured HBM savings or runtime speedups.',
        ring_reference_is_not_a_latency_speedup='P8 payload ratio=(2(P-1)MhH+4(P-1)Mh)/(2(P-1)MH)=h+2h/H=4.0015625. Only an equal-bandwidth/no-other-cost ring send-service reference; compute-before-collective, ownership and startup change.',
        observer_guards=GUARDS,actual_physical_capacity_or_speedup_proved=False,overall_complete=False))
    print(dict(status='RECONCILED',prior_answer_version=11,prior_inventory=[90,26],all_observers_closed=False,new_ceiling_domains=4))

if __name__=='__main__':main()
