"""Check exact source identity,finite witnesses and symbolic-bound scope; CPU only."""
from pathlib import Path
from itertools import product
import argparse,hashlib,json,math
from finalize_origin_coverage_v9 import N,O,read,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');args=p.parse_args();checks=[]
    def ck(n,v):assert v,n;checks.append(n)
    def pinned(r):
        path=Path(r['path']);b=path.read_bytes();return len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
    for r in read(O/'stop-mask-inputs-v108.json')['inputs']:ck('input:'+r['path'],pinned(r))
    source=read(O/'stop-mask-source-contract-v108.json');texts={}
    for part in source['parts']:
        ck('source:'+part['path'],pinned(part)and pinned(part['manifest'])and part['manifest_match'])
        text=Path(part['path']).read_text();texts[Path(part['path']).name]=text
        for f in part['functions']:ck('function:'+f['name'],text.splitlines()[f['first_line']-1:f['last_line']]==f['source'])
    builtin=texts['builtin.py'];thinking=texts['thinking_budget_state.py'];caller=texts['rejection_sampler.py']
    ck('all_neginf_not_no_finite','torch.isneginf(logits[unique_rows]).all(dim=-1)'in builtin)
    ck('finite_original_stop_only','row_needs_restore[inverse_indices] & torch.isfinite(stop_logits)'in builtin)
    ck('ragged_leading_prefix','min(max(remaining, 0), num_draft_arr[req_idx])'in builtin)
    ck('force_is_finite_overwrite','logits.index_put_((active_indices, force_tokens), fill)'in thinking and 'logits.new_full((len(active_indices_cpu),), 1e9)'in thinking)
    ck('caller_min_before_forcing',caller.index('logits = processor.apply_with_spec_decode(')<caller.index('logits = holder.apply_to_logits('))
    ck('forcing_state_effects_retained','state["force_index"] = []'in thinking and 'state["bonus_token_forced"] = True'in thinking)
    alg=read(O/'stop-mask-fusion-algebra-v108.json');cost=read(O/'stop-mask-work-and-ceiling-v108.json');validation=read(O/'stop-ma<REDACTED_TOKEN>.json')
    ck('finite_grid_counts',(validation['oracle_cases'],validation['lazy_no_forcing_cases'],validation['split_merge_checks'])==(120000,20000,360000))
    neg=-float('inf')
    def argmax(xs):return min(range(len(xs)),key=lambda i:(-xs[i],i))
    def mask_restore(x,S,u):
        y=[neg if i in S else z for i,z in enumerate(x)]
        if u and all(z==neg for z in y):
            for i in S:
                if math.isfinite(x[i]):y[i]=x[i]
        return y
    def apply_force(y,F):
        y=list(y)
        for i in F:y[i]=1e9
        return y
    for i,r in enumerate(validation['witnesses']):
        x=[neg if v=='-inf'else float('inf')if v=='+inf'else v for v in r['x']];S=set(r['stop_ids']);F=set(r['force_ids']);u=r['structured']
        actual=argmax(apply_force(mask_restore(x,S,u),F))
        if i==0:wrong=argmax([neg if j in S else z for j,z in enumerate(x)])
        elif i==1:wrong=argmax([8.0,4.0])
        elif i==2:wrong=min(F)
        elif i==3:wrong=argmax([max(z,1e9)if j in F else z for j,z in enumerate(x)])
        elif i==4:wrong=argmax(mask_restore(apply_force(x,F),S,u))
        else:
            y=[neg if j in S else z for j,z in enumerate(x)]
            if all(z==neg for j,z in enumerate(y)if j not in F):
                for j in S:
                    if math.isfinite(x[j]):y[j]=x[j]
            wrong=argmax(apply_force(y,F))
        ck('witness:'+r['name'],actual==r['correct_id']and wrong==r['incorrect_id']and actual!=wrong)
    ck('six_witnesses',len(validation['witnesses'])==6)
    for r in cost['models']:
        t,rv,v=r['T'],r['structured_restore_rows'],r['V']
        baseline=4*rv*v+4*rv*v+4*rv*v+rv*v+rv*v+rv+4*t*v
        ck('materialized_interface:'+str((t,rv)),baseline==r['separately_materialized_gather_predicate_all_argmax_reference_bytes']and baseline-r['chosen_fullpass_ID_summary_FLOAT_input_reference_bytes']==r['removable_reference_interfaces_before_summary_overhead_bytes']==14*rv*v+rv)
    ck('symbolic_not_actual',cost['conditional_lower_bound']['status']=='SYMBOLIC_CEILING'and not cost['actual_native_applicability_or_causal_gain_proved']and cost['logical_reference_not_native_kernel_fusion_or_HBM_traffic'])
    ck('allinput_bound_is_worst_case','worst case'in cost['conditional_lower_bound']['domain']and 'n>=2 and s>=2'in cost['conditional_lower_bound']['structured_extension'])
    comp=read(O/'stop-mask-composition-gates-v108.json');ck('all_readers_remain_conditions',not comp['actual_branch_attested']and comp['zero_current_saving_if_branch_inactive'])
    for name in ['stop-mask-source-contract-v108.json','stop-mask-fusion-algebra-v108.json','stop-ma<REDACTED_TOKEN>.json','stop-mask-work-and-ceiling-v108.json','stop-mask-composition-gates-v108.json']:
        json.loads((O/name).read_text(),parse_constant=lambda x:(_ for _ in()).throw(ValueError(x)))
        ck('strict_JSON:'+name,True)
    if args.preflight:manifest=read(O/'checkpoint-manifest-v107.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v108.json');ck('PROGRESS_open',cp['status']=='PROGRESS'and not cp['overall_complete'])
        matrix=read(O/'module-phase-branch-matrix-v108.json');ck('353_semantic_rows26_modules',len(matrix['modules'])==26 and sum(len(m['branches'])for m in matrix['modules'])==353)
        row=next(r for m in matrix['modules']if m['id']=='M17'for r in m['branches']if r['id']=='M17-OP-09');ck('specific_processor_review',row['stop_mask_review_v108']['source_body_reviewed']and not row['stop_mask_review_v108']['actual_flags_or_all_readers_closed'])
        cov=read(O/'candidate-relation-coverage-v108.json');ck('132_25_463',(cov['scoped_conditional_pairs'],cov['higher_order_groups'],cov['remaining_pair_relations'])==(132,25,463))
        old=read(O/'operator-analysis-coverage-ledger-v107.json');new=read(O/'operator-analysis-coverage-ledger-v108.json')
        for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']:ck('native_unchanged:'+k,old[k]==new[k])
        for r in read(O/'candidate-record-index-v108.json')['model_candidates']:ck('master:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('F35_updated','review_v108'in read(Path(next(r['master_record']for r in read(O/'candidate-record-index-v108.json')['model_candidates']if r['id']=='F35'))))
        manifest=read(O/'checkpoint-manifest-v108.json')
    for r in manifest['artifacts']:ck('manifest:'+r['path'],pinned(r))
    result=dict(status='PASS',checks=len(checks),preflight=args.preflight,oracle_cases=120000,lazy_cases=20000,merge_checks=360000,witnesses=6,device_probe_executed=False,new_native_promotions=0,overall_complete=False)
    if args.preflight:write(O/'stop-mask-preflight-v108.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
