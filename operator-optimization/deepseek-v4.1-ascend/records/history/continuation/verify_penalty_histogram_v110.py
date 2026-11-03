"""Independent source,ownership,integer-state and conditional-bound checks;CPU only."""
from pathlib import Path
from collections import Counter
from bisect import bisect_right
import argparse,hashlib,json,math,struct
from finalize_origin_coverage_v9 import N,O,read,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');a=p.parse_args();checks=[]
    def ck(n,v):assert v,n;checks.append(n)
    def pinned(r):
        b=Path(r['path']).read_bytes();return len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
    for r in read(O/'penalty-histogram-inputs-v110.json')['inputs']:ck('input:'+r['path'],pinned(r))
    for r in read(O/'penalty-fallback-source-index-v110.json')['files']:
        b=Path(r['local_path']).read_bytes();ck('retrieval:'+r['path'],r['status']=='READ'and len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256'])
        if r['caller_control']:
            rel=r['path'].split('/vllm-workspace/vllm/vllm/',1)[1]
            from finalize_origin_coverage_v9 import W
            ck('exact_caller_control:'+rel,b==(W/'frozen-source-snapshot/vllm/vllm'/rel).read_bytes())
    src=read(O/'penalty-source-contract-v110.json');functions={}
    for part in src['parts']:
        ck('source:'+part['path'],pinned(part));t=Path(part['path']).read_text()
        if 'manifest'in part:ck('manifest:'+part['path'],pinned(part['manifest'])and part['manifest_match'])
        else:ck('retrieved_caller_controls',pinned(part['retrieval_index'])and part['caller_controls_matched']==2)
        for f in part['functions']:
            ck('segment:'+part['path']+f['name'],t.splitlines()[f['first_line']-1:f['last_line']]==f['source'])
            functions[(Path(part['path']).name,f['name'])]='\n'.join(f['source'])
    binning=functions[('bincount.py','get_token_bin_counts_and_mask_triton')];kernel=functions[('penalty.py','apply_all_penalties_kernel')]
    ck('I32_empty_and_mask','dtype=torch.int32'in binning and 'n_rows == 0 or n_cols == 0'in binning and 'bin_counts > 0'in binning)
    ck('count_cast_and_roles','need_repetition_penalty = (prompt_mask_val | output_mask_val)'in kernel and 'frequency_penalty * output_bin_counts'in kernel and 'presence_penalty * output_mask_val'in kernel and '.to(tl.float32)'in kernel)
    ck('prompt_counts_not_consumed','_, prompt_mask = get_token_bin_counts_and_mask_triton'in functions[('penalty.py','apply_penalties_triton')])
    ck('fallback_I64_Vplus1','vocab_size + 1'in functions[('utils.py','get_token_bin_counts_and_mask')]and 'dtype=torch.long'in functions[('utils.py','get_token_bin_counts_and_mask')])
    ck('native_repetition_is_separate','from vllm._custom_ops import apply_repetition_penalties'in functions[('utils.py','apply_penalties')])
    route=read(O/'penalty-backend-row-and-dtype-contract-v110.json')
    ck('fallback_row_identity_not_proved',not route['actual_fallback_reachability_or_preexpanded_arrays_attested'])
    case=next(r for r in route['row_cases']if r['drafts']==[0,2]);ck('T_equals_B_not_same_owner',case['B']==case['T']==2 and case['expected_request_owners']==[1,1]and case['unexpanded_request_rows']==[0,1])
    ck('no_current_flag_proof',not read(O/'penalty-profile-applicability-v110.json')['actual_no_penalties_or_counts_or_provider_attested'])
    val=read(O/'penalty-integer-reference-validation-v110.json')
    ck('exhaustive_reference_counts',(val['request_contexts'],val['exact_prefix_histogram_row_checks'])==(81120,206856))
    ck('64_ragged_cases',len(val['ragged_cases'])==64)
    for r in val['ragged_cases']:
        lengths=r['lengths'];ends=[];total=0
        for n in lengths:total+=n;ends.append(total)
        expected=[]
        for k in range(total):
            j=bisect_right(ends,k);begin=0 if j==0 else ends[j-1];expected.append([j,k-begin])
        ck('right_biased_owner:'+str(lengths),r['owners']==expected and r['rows']==total)
    for r in val['cross_iteration_transitions']:
        c=Counter(r['committed']);ck('state:'+r['action'],r['counts']==[c[v]for v in range(val['vocab'])])
    demand=read(O/'penalty-parameter-demand-v110.json');ck('8_parameter_patterns',len(demand['cases'])==8 and demand['FLOAT_reference_grouping_checks']==256)
    for r in demand['cases']:
        fields=[]
        if r['rho']!=1:fields.append('U_prompt_OR_output_presence')
        if r['frequency']!=0:fields.append('C_native_cast_count')
        elif r['presence']!=0:fields.append('O_output_presence')
        ck('needed_stats:'+str((r['rho'],r['frequency'],r['presence'])),r['required_semantic_statistics']==fields)
    precision=read(O/'penalty-precision-and-invalid-rewrites-v110.json');F=lambda x:struct.unpack('f',struct.pack('f',x))[0]
    for i,w in enumerate(precision['integer_vs_BOOL_witnesses']):
        P={v for v in w['prompt']};C=Counter(w['output'])
        def ids(pp,cc):
            zs=[]
            for v,x in enumerate(w['x']):
                rep=w['rho']if v in pp or cc[v]>0 else 1.0
                y=(x/rep if x>0 else x*rep)-w['freq']*cc[v]-w['presence']*(cc[v]>0);zs.append(y)
            return max(range(len(zs)),key=lambda v:(zs[v],-v))
        expected=ids(P,C)
        if i==0:C=Counter({v:int(c>0)for v,c in C.items()})
        elif i==1:C=Counter(w['prompt']+w['output'])
        elif i==2:P=set()
        else:C.update([w['current_draft']])
        ck('invalid_rewrite:'+w['kind'],expected==w['correct']and ids(P,C)==w['incorrect']and expected!=w['incorrect'])
    n=precision['FLOAT_reference_witnesses'];r=n['reciprocal_vs_division'];s=n['combine_subtractions']
    ck('BF16_representable_x',struct.unpack('I',struct.pack('f',r['BF16_x']))[0]&65535==0)
    ck('reciprocal_round_counter',F(r['BF16_x']*F(1/r['FLOAT_rho']))==r['source_style']and F(r['BF16_x']/r['FLOAT_rho'])==r['direct_division']and r['source_style']!=r['direct_division'])
    ck('subtract_order_counter',F(F(2**25-1)-1)==s['source_style']==33554432 and F(2**25-2)==s['combined_term']==33554430)
    costs=read(O/'penalty-work-and-resource-models-v110.json')
    for r in costs['models']:
        B,T,V=r['B'],r['T'],r['V'];q=T*V
        ck('logical_reference:'+str((B,T)),r['separate_dense_penalty_reference_bytes']==(8+10+14)*q and r['separate_dense_penalty_plus_argmax_bytes']==36*q and r['output_mask_inline_penalty_reference_bytes']==26*q and r['output_mask_inline_penalty_plus_argmax_reference_bytes']==30*q and r['output_mask_inline_plus_penalty_argmax_stream_reference_bytes']==22*q)
        ck('typed_footprints:'+str((B,T)),r['original_auxiliary_plane_values_bytes']==10*q and r['fallback_I64_count_mask_allocated_value_reference_bytes']==18*q+16*T and r['per_request_base_counts_plus_packed_prompt_values_bytes']==4*B*V+4*B*((V+31)//32))
    cold=read(O/'penalty-cold-query-symbolic-ceiling-v110.json');ck('symbolic_cold_only',cold['status']=='SYMBOLIC_CEILING'and 'Warm cached'in cold['not_applicable_to'])
    for r in cold['reference_adversaries']:
        x=r['x'];history=r['history'];V,L=r['V'],r['L']
        def decision(xx,hh):
            c=Counter(hh);return max(range(len(xx)),key=lambda v:(F(F(xx[v])-F(c[v])),-v))
        origin=decision(x,history);ck('adversary_origin:'+str((V,L)),origin==r['original_ID']==0)
        for i in range(L):
            h=list(history);h[i]=0;ck('every_history_word:'+str((V,L,i)),decision(x,h)==1)
        for v in range(V):
            alt=list(x);alt[v]=-1 if v==0 else(2**((L+1).bit_length())if v==2 else 1)
            ck('every_BF16_logit:'+str((V,L,v)),decision(alt,history)!=origin and struct.unpack('I',struct.pack('f',alt[v]))[0]&65535==0)
        ck('two_selected_input_interfaces:'+str((V,L)),r['necessary_uncached_materialized_FLOAT_input_bytes']==4*V+8*L and r['necessary_uncached_retained_BF16_input_bytes']==2*V+8*L)
    docs=read(O/'penalty-public-contract-index-v110.json');ck('official_Web_snapshot_pinned',pinned(docs['web_source_snapshot'])and docs['target_Torch_version_or_native_backend_not_attested'])
    ck('read_only_SSH_closed',not(N/'ssh166-penalty-v110.sock').exists())
    if a.preflight:manifest=read(O/'checkpoint-manifest-v109.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v110.json');ck('PROGRESS_open',cp['status']=='PROGRESS'and not cp['overall_complete'])
        matrix=read(O/'module-phase-branch-matrix-v110.json');ck('353rows26modules',sum(len(m['branches'])for m in matrix['modules'])==353 and len(matrix['modules'])==26)
        row=next(r for m in matrix['modules']if m['id']=='M17'for r in m['branches']if r['id']=='M17-OP-09');ck('specific_body_analysis',row['penalty_histogram_review_v110']['source_count_and_penalty_bodies_reviewed']and not row['penalty_histogram_review_v110']['actual_flags_or_all_readers_closed'])
        ck('no_new_device_work',not read(O/'candidate-profile-ledger-v110.json')['new_device_work'])
        old=read(O/'operator-analysis-coverage-ledger-v109.json');new=read(O/'operator-analysis-coverage-ledger-v110.json')
        for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']:ck('native_unchanged:'+k,old[k]==new[k])
        cov=read(O/'candidate-relation-coverage-v110.json');ck('132_25_463',(cov['scoped_conditional_pairs'],cov['higher_order_groups'],cov['remaining_pair_relations'])==(132,25,463))
        for r in read(O/'candidate-record-index-v110.json')['model_candidates']:ck('master:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        manifest=read(O/'checkpoint-manifest-v110.json')
    for r in manifest['artifacts']:ck('manifest:'+r['path'],pinned(r))
    result=dict(status='PASS',checks=len(checks),preflight=a.preflight,request_contexts=81120,exact_prefix_rows=206856,
        ragged_cases=64,parameter_cases=8,invalid_integer_rewrites=4,FLOAT_reference_counters=2,new_device_work=False,new_native_promotions=0,overall_complete=False)
    if a.preflight:write(O/'penalty-histogram-preflight-v110.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
