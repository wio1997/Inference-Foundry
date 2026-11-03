"""Operator histogram/penalty/argmax representations and cold-query bounds; CPU only."""
from pathlib import Path
from itertools import product
from collections import Counter
import ast,hashlib,json,math,re,struct
from finalize_origin_coverage_v9 import N,O,W,read,write

def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def source(p,names):
    p=Path(p);t=p.read_text();tree=ast.parse(t);root=next(r for r in p.parents if r.name=='frozen-source-snapshot')
    manifest=root/'SOURCE-MANIFEST.json';r=next(r for r in read(manifest)['files']if r['path']==str(p.relative_to(root)))
    assert r['sha256']==pin(p)['sha256'];parts=[]
    for name in names:
        f=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name==name)
        parts.append(dict(name=name,first_line=f.lineno,last_line=f.end_lineno,source=t.splitlines()[f.lineno-1:f.end_lineno]))
    return dict(**pin(p),manifest=pin(manifest),manifest_match=True,functions=parts)
def retrieved_source(p,names):
    p=Path(p);t=p.read_text();tree=ast.parse(t);parts=[]
    for name in names:
        f=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name==name)
        parts.append(dict(name=name,first_line=f.lineno,last_line=f.end_lineno,source=t.splitlines()[f.lineno-1:f.end_lineno]))
    return dict(**pin(p),retrieval_index=pin(O/'penalty-fallback-source-index-v110.json'),
        caller_controls_matched=2,loaded_runtime_identity_not_attested=True,functions=parts)
def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
def hist(xs,V):
    result=[0]*V
    for x in xs:
        if x==-1:x=V
        assert 0<=x<=V,'Reference domain excludes unsafe other negative tokens'
        if x<V:result[x]+=1
    return result
def score(x,P,c,rho=1.0,freq=1.0,presence=0.0):
    """One separately-rounded FLOAT reference policy,not native compiler attestation."""
    x,rho,freq,presence=map(f32,[x,rho,freq,presence]);r=rho if P or c>0 else 1.0
    scaling=f32(1/r)if x>0 else r
    return f32(f32(f32(x*scaling)-f32(freq*f32(c)))-f32(presence*float(c>0)))
def argmax(xs):return max(range(len(xs)),key=lambda i:(xs[i],-i))

def main():
    assert not(O/'REVIEW-CHECKPOINT-v110.json').exists()
    asc=W/'sampler-source-supplement/frozen-source-snapshot/vllm-ascend/vllm_ascend'
    upstream=W/'frozen-source-snapshot/vllm/vllm/v1/sample/rejection_sampler.py'
    parts=[source(asc/'ops/triton/bincount.py',['token_bin_counts_and_mask_kernel','get_token_bin_counts_and_mask_triton']),
        source(asc/'ops/triton/penalty.py',['apply_all_penalties_kernel','apply_penalties_triton','_apply_all_penalties_triton']),
        source(asc/'sample/penalties.py',['_convert_to_tensors','apply_all_penalties']),
        source(asc/'sample/rejection_sampler.py',['apply_penalties','apply_logits_processors','expand_batch_to_tokens']),
        source(upstream,['_combine_outputs_with_spec_tokens']),
        source(upstream.parent/'sampler.py',['apply_penalties']),
        source(upstream.parent/'ops/penalties.py',['apply_all_penalties','_convert_to_tensors'])]
    utility=next(r['local_path']for r in read(O/'penalty-fallback-source-index-v110.json')['files']if r['path'].endswith('/model_executor/layers/utils.py'))
    parts.append(retrieved_source(utility,['get_token_bin_counts_and_mask','apply_penalties']))
    files=[];matches=[];label_fields=Counter()
    for db in range(32):
        p=N/f'original-profile-cache-v53/analysis/native-execution-identity-v14/db-{db:02d}.json';d=read(p);files.append(p)
        for key in ['COMPUTE_definitions','linked_PYTORCH_APIs','annotation_name_type_census']:
            for row in d[key]:
                labels={k:v for k,v in row.items()if k in ['name','name_text','opType_text']and isinstance(v,str)}
                label_fields[key]+=len(labels)
                if any(re.search('penalt|token_bin_counts|bincount',v,re.I)for v in labels.values()):
                    matches.append(dict(database_index=db,collection=key,row_labels=labels))
    write(O/'penalty-profile-applicability-v110.json',dict(saved_databases=32,label_fields_examined=dict(label_fields),matches=matches,
        search_is_name_navigation_not_flag_attestation=True,actual_no_penalties_or_counts_or_provider_attested=False,
        existing_profile_boundary='v28/v65/v102 F35 selection/cast/clone/greedy chains remain context,not proof of penalty execution.',
        source_flag='no_penalties=True returns before this histogram/penalty subgraph;temperature0/all_greedy alone does not imply no_penalties.',
        pending_BOOL24_exclusion='At V129280,the logical returned histogram mask has n_rows*129280 elements and cannot be a24-element field;this does not exclude opaque native scratch temporaries or identify the pending BOOL24 field.',
        new_native_attributions=0,overall_complete=False))
    write(O/'penalty-source-contract-v110.json',dict(parts=parts,
        facts=[dict(status='DOCUMENTED',fact='Prompt histogram numeric counts are not semantically consumed at this call site;prompt_mask=(prompt_count>0). Output counts and output_mask both feed the penalty kernel.'),
            dict(status='DOCUMENTED',fact='Repetition uses prompt_mask OR output_mask;frequency uses FLOAT(output_count) only;presence uses output_mask only. Prompt and output cannot be pooled for frequency.'),
            dict(status='DOCUMENTED',fact='Counts are zero-initialized INT32 and updated by atomic_add1;mask materialization uses count>0. Empty rows/cols return correctly shaped zero counts and mask;noncontiguous token input is copied.'),
            dict(status='DOCUMENTED',fact='Output wrapper maps -1 placeholders toV;kernel predicate is token<V,without an explicit token>=0 check. Equivalence here requires valid0..V tokens after sanitation and bounded INT32 counts.'),
            dict(status='DOCUMENTED',fact='Each request with n draft tokens emits n histories:committed output followed by prefixes spec[:d],d=0..n-1. Zero-draft requests emit no target-history rows.'),
            dict(status='DOCUMENTED',fact='The source constructs scalar reciprocal before multiply and then subtracts frequency and presence terms in order. Actual reciprocal/FMA/cast/NaN policy is not attested by source alone.')],
        logical_private_auxiliary_planes='prompt INT32 count + prompt BOOL mask + output INT32 count + output BOOL mask=10TV value bytes;not actual allocator peak/reserved memory.',
        new_target_source_files=1,cached_source_files=7,all_helpers_all_consumers_or_native_policy_closed=False,no_framework_wide_research=True,overall_complete=False))
    write(O/'penalty-backend-row-and-dtype-contract-v110.json',dict(
        Triton=dict(count_dtype='INT32',allocated_count_shape='[T,V]',mask_dtype='BOOL',
            request_parameters='Selected source gathers prompt and penalty parameters byrepeat_indices before the provider call.',
            relevant_domain='Valid ragged request mapping,bounded positive counts,noPenaltiesFalse andHAS_TRITONTrue.'),
        fallback=dict(count_dtype='INT64(torch.long)',allocated_count_shape='[T,V+1],returned[:,:V] view',mask_dtype='BOOL',
            repetition='Delegates to vllm._custom_ops.apply_repetition_penalties;actual Ascend native dispatch/numerics unbound.',
            request_parameters='Ascend HAS_TRITONFalse delegates directly to Sampler.apply_penalties without the repeat_indices gathers used by its Triton branch.',
            condition='If metadata arrays remain request-rowB,do not assume they are valid target-rowT inputs. T=B alone is insufficient forragged ownership.',
            precision='I64counts and custom repetition provider are distinct from Triton I32/explicit reciprocal source;do not transfer native precision/resource claims.'),
        row_cases=[dict(drafts=[1,1],B=2,T=2,expected_request_owners=[0,1],request_and_target_rows_match=True),
            dict(drafts=[2,2],B=2,T=4,expected_request_owners=[0,0,1,1],
                coefficient_tensor_shape_problem='[2,1] and[4,V] are not broadcastable under documented tensor semantics;if supplied params are request-rowB.'),
            dict(drafts=[0,2],B=2,T=2,expected_request_owners=[1,1],unexpanded_request_rows=[0,1],
                same_T_B_does_not_prove_owner_match=True,scalar_contract_row0_witness=dict(prompt_rows=[[],[0]],rho_rows=[2,2],logits=[2.0,1.5],correct_owner_ID=1,unexpanded_owner_ID=0)),
            dict(drafts=[2],B=1,T=2,expected_request_owners=[0,0],
                documented_scatter_model='A[1,L] prompt index updates onlyrow0 of an allocated[2,V+1] count tensor;there is no implicit prompt-row broadcast.',
                full_custom_repetition_provider_compatibility='UNKNOWN;parameter broadcast in frequency terms alone is not whole-backend compatibility.')],
        documentation_index='penalty-public-contract-index-v110.json',
        source_analysis_not_actual_runtime_failure=True,actual_fallback_reachability_or_preexpanded_arrays_attested=False,
        no_target_repair_or_execution=True,overall_complete=False))
    write(O/'penalty-histogram-representation-v110.json',dict(
        valid_domain='Valid request/row/parameter epochs;V>0;after -1→V sanitation tokens0..V;sentinelV ignored;per-bin counts<2^31;preserve exact INT32→FLOAT cast/native arithmetic policy.',
        identities=dict(P_jv='OR over prompt positions of token==v',C_jv='committed output occurrence count ofv',
            C_jdv='C_jv + sum(1[spec_j[r]==v] for r<d)',O_jdv='C_jdv>0',
            repetition='P_jv OR O_jdv',frequency='FLOAT(C_jdv)',presence='FLOAT(O_jdv)'),
        choices=[dict(id='P110-1',opportunity='Compute output_mask predicate in penalty consumer from its already loaded INT32 count.',
            preserve='Compare as INT32 before FLOAT conversion;keep source count bounds/aliases/all consumers.',
            new_cost='Predicate/flags stay live in consumer;do not remove count values needed by frequency.'),
            dict(id='P110-2',opportunity='Specialize prompt call-site producer to presence,optionally a packed per-request bitmap.',
            preserve='Only prompt consumer set is established here;do not globally replace helper counts. Need race-free OR/ownership,initialization,packing/tails and native publication.',
            new_cost='Presence construction/zeroing/word access or alternative reduction work;native atomic-bit operation support/capacity is unknown.'),
            dict(id='P110-3',opportunity='One per-request prompt presence and committed-output histogram plus row-specific draft prefix deltas,instead of repeated full histories/histograms.',
            preserve='Actual ragged cu/request ownership,zero-draft omission,repeated-token multiplicity andparameter/slot epochs.',
            new_cost='Base count construction/updates,prefix delta access or small histogram,metadata/map/index/publish and invalidation costs.'),
            dict(id='P110-4',opportunity='Penalty FLOAT result can flow into selected mask/MinTokens/forcing/argmax graph,retaining required rounding without standalone write/read.',
            preserve='v108 A/B/H must see post-penalty,post-earlier-mask values;preserve finite forcing order,state effects and all raw/processed/random/logprob/mutation observers.',
            new_cost='Needed score outputs and reduction/state metadata remain;no full working-plane removal if it has live readers.')],
        integer_count_reassociation_is_not_FLOAT_penalty_reassociation=True,
        no_implementation_or_native_speedup=True,overall_complete=False))
    demand=[];demand_checks=0
    for rho,freq,presence in product([1.0,2.0],[0.0,1.0],[0.0,1.0]):
        required=(['U_prompt_OR_output_presence']if rho!=1 else[])+(['C_native_cast_count']if freq!=0 else[])+(['O_output_presence']if presence!=0 and freq==0 else[])
        observed={}
        for x,P,c in product([-2.0,-0.0,0.0,2.0],[False,True],range(4)):
            U=P or c>0;stat=((U,)if rho!=1 else())+((f32(c),)if freq!=0 else())+((c>0,)if presence!=0 and freq==0 else())
            key=(struct.pack('f',x).hex(),stat);bits=struct.pack('f',score(x,P,c,rho,freq,presence)).hex()
            if key in observed:assert observed[key]==bits
            observed[key]=bits;demand_checks+=1
        demand.append(dict(rho=rho,frequency=freq,presence=presence,required_semantic_statistics=required,
            statistic_is_sufficient_in_reference_domain=True))
    write(O/'penalty-parameter-demand-v110.json',dict(cases=demand,FLOAT_reference_grouping_checks=demand_checks,
        statistics_are_sufficient_not_universal_minimality_or_per_input_necessity=True,
        domain='Finite penalty coefficients,positive repetition rho,nonnegative bounded counts;match native cast and downstream arithmetic policy. Parameter representatives encode eight zero/nonzero/rho1 patterns,not actual flags.',
        distinctions=['Repetition-only needs union presenceU,not separate prompt/output masks or multiplicities.',
            'Presence-only needs output presenceO.',
            'Generic frequency needs the original native-cast count value,not merely BOOL;exact integer state is a sufficient representation,not a universal requirement for every implementation.',
            'When native-cast nonnegative count is available,O can be derived from it. Prompt data is irrelevant if rho1.',
            'No-statistic neutral branch removes history dependence;dropping zero/identity arithmetic itself still needs signed-zero/FTZ/NaN/native-rounding and observer proof.'],
        representation_state_note='Union bitsets alone do not support arbitrary rollback/deletion of duplicates or prompt retention;retain multiplicity/provenance or rebuild.',
        actual_parameter_rows_or_native_pruning_attested=False,overall_complete=False))
    V=3;request_cases=0;row_checks=0
    bases=[xs for k in range(4)for xs in product([0,1,2,V,-1],repeat=k)]
    prompts=[xs for k in range(3)for xs in product(range(V),repeat=k)]
    specs=[xs for k in range(4)for xs in product(range(V),repeat=k)]
    for base,prompt,spec in product(bases,prompts,specs):
        initial=hist(base,V);P=[v>0 for v in hist(prompt,V)];delta=[0]*V;explicit=list(base)
        for d,token in enumerate(spec):
            full=hist(explicit,V);merged=[initial[v]+delta[v]for v in range(V)]
            assert full==merged and P==[bool(prompt.count(v))for v in range(V)]
            assert [c>0 for c in full]==[c>0 for c in merged]
            row_checks+=1;delta[token]+=1;explicit.append(token)
        request_cases+=1
    ragged=[]
    for lengths in product(range(4),repeat=3):
        base=[[j%V,(j+1)%V]for j in range(3)];spec=[[j%V]*n for j,n in enumerate(lengths)]
        explicit=[];planned=[];owners=[]
        for j,(h,sp)in enumerate(zip(base,spec)):
            if not sp:continue
            explicit.append(list(h))
            for d in range(len(sp)-1):explicit.append([*explicit[-1],sp[d]])
            for d in range(len(sp)):planned.append(hist(h+sp[:d],V));owners.append((j,d))
        assert [hist(h,V)for h in explicit]==planned and len(owners)==sum(lengths)
        ragged.append(dict(lengths=list(lengths),rows=len(owners),owners=owners))
    transitions=[];committed=[0];cache=hist(committed,V)
    for label,append,truncate in [('append_verified_emitted_tokens_including_recovery',[1],None),('rollback_committed_tail',[],1),('append_after_rollback',[2,2],None),('slot_reuse_requires_new_base',[],0)]:
        before=list(committed)
        if truncate is not None:
            removed=committed[truncate:];committed=committed[:truncate]
            removed_counts=hist(removed,V);cache=[cache[v]-removed_counts[v]for v in range(V)]
        committed.extend(append);added=hist(append,V);cache=[cache[v]+added[v]for v in range(V)]
        assert cache==hist(committed,V)and all(c>=0 for c in cache)
        transitions.append(dict(action=label,before=before,committed=list(committed),counts=list(cache)))
    write(O/'penalty-integer-reference-validation-v110.json',dict(status='PASS',vocab=V,request_contexts=request_cases,
        exact_prefix_histogram_row_checks=row_checks,ragged_cases=ragged,cross_iteration_transitions=transitions,
        finite_reference_not_native_or_runtime_proof=True,overall_complete=False))
    numeric=dict(reciprocal_vs_division=dict(BF16_x=0.10009765625,FLOAT_rho=f32(1.7),
        source_style=f32(f32(0.10009765625)*f32(1/f32(1.7))),direct_division=f32(f32(0.10009765625)/f32(1.7))),
        combine_subtractions=dict(BF16_x=2**25,FLOAT_frequency=1,FLOAT_presence=1,INT32_count=1,
            source_style=f32(f32(f32(2**25)-1)-1),combined_term=f32(f32(2**25)-f32(1+1))))
    assert numeric['reciprocal_vs_division']['source_style']!=numeric['reciprocal_vs_division']['direct_division']
    assert numeric['combine_subtractions']['source_style']!=numeric['combine_subtractions']['combined_term']
    witnesses=[dict(kind='Replace output counts with BOOL',x=[2.0,1.0],prompt=[],output=[0,0],rho=1,freq=1,presence=0,correct=1,incorrect=0),
        dict(kind='Pool prompt/output counts for frequency',x=[2.0,1.0],prompt=[0,0],output=[],rho=1,freq=1,presence=0,correct=0,incorrect=1),
        dict(kind='Drop prompt presence from repetition',x=[2.0,1.5],prompt=[0],output=[],rho=2,freq=0,presence=0,correct=1,incorrect=0),
        dict(kind='Include current draft instead of preceding prefix',x=[2.0,1.0],prompt=[],output=[],current_draft=0,rho=1,freq=2,presence=0,correct=0,incorrect=1)]
    for w in witnesses:
        P=[c>0 for c in hist(w['prompt'],2)];C=hist(w['output'],2)
        correct=argmax([score(x,P[v],C[v],w['rho'],w['freq'],w['presence'])for v,x in enumerate(w['x'])]);assert correct==w['correct']
        if w['kind'].startswith('Replace'):C=[int(c>0)for c in C]
        elif w['kind'].startswith('Pool'):C=hist(w['prompt']+w['output'],2)
        elif w['kind'].startswith('Drop'):P=[False,False]
        else:C=hist(w['output']+[w['current_draft']],2)
        incorrect=argmax([score(x,P[v],C[v],w['rho'],w['freq'],w['presence'])for v,x in enumerate(w['x'])]);assert incorrect==w['incorrect']and correct!=incorrect
    write(O/'penalty-precision-and-invalid-rewrites-v110.json',dict(integer_vs_BOOL_witnesses=witnesses,FLOAT_reference_witnesses=numeric,
        FLOAT_domain='Separately rounded IEEE FLOAT32 reference,finite bounded inputs;not actual native reciprocal approximation/FMA/FTZ or compiler policy.',
        source_contract='Retaining exact count values permits same native cast and downstream arithmetic;changing reciprocal/division or subtraction grouping needs independent precision proof.',overall_complete=False))
    costs=[];realV=read(O/'heads-sampling-fusion-costs-v65.json')['sampling_models'][0]['V']
    for B,T in [(2,10),(4,10),(4,20)]:
        q=T*realV;bitmap=B*((realV+31)//32)*4
        costs.append(dict(B=B,T=T,V=realV,T_role='Saved F35 row-shape/capacity reference,not attested valid rows or sum(actual drafts)',
            separate_dense_penalty_reference_bytes=32*q,separate_dense_penalty_plus_argmax_bytes=36*q,
            output_mask_inline_penalty_reference_bytes=26*q,output_mask_inline_penalty_plus_argmax_reference_bytes=30*q,
            output_mask_inline_plus_penalty_argmax_stream_reference_bytes=22*q,
            output_mask_materialization_budget_bytes=6*q,penalty_argmax_private_interface_budget_bytes=8*q,
            original_auxiliary_plane_values_bytes=10*q,
            per_request_base_counts_plus_packed_prompt_values_bytes=4*B*realV+bitmap,
            fallback_I64_count_mask_allocated_value_reference_bytes=18*q+16*T,
            fallback_per_request_I64_base_plus_prompt_bitset_values_bytes=8*B*realV+bitmap,
            packed_prompt_values_bytes=bitmap,
            omitted_from_all_traffic_figures='History input reads,contiguous/gather/H2D/padding,atomicRMW/presence creation,parameters,state/prefix metadata,packed-word operations,required score writes and summary costs.',
            shape_reference_not_actual_flags_lengths_peak_or_traffic=True))
    write(O/'penalty-work-and-resource-models-v110.json',dict(models=costs,
        explicit_dense_baseline='Two INT32 zero writes8TV;two count→BOOL passes10TV;penalty(logits read/write8TV+prompt/output masks2TV+output counts4TV)=14TV,total32TV;argmax adds4TV. INT32/BOOL/FLOAT values4/1/4B.',
        output_mask_inline='Remove output count→BOOL read4TV,BOOL writeTV,and penalty BOOL readTV=6TV;consumer predicate retained.',
        penalty_argmax_stream='Remove private FLOAT result write4TV andargmax read4TV=8TV only if all processed-score/mutation readers permit;do not add to existing F35 credits without one union.',
        prompt_presence_and_per_request_representations='Requested-value footprints only;native race-free construction/update/capture lifetime and new work must be charged. No actual allocator peak/reserved or HBM claim.',
        fallback_scope='32/36/30/22TV traffic budgets are Triton I32 chosen references only. Fallback source uses I64V+1 counts and a separate native repetition customop;only requested-value footprints are priced here,not a native traffic/precision equivalence.',
        prefix_options='Direct prefix scan costs sum_j V*n_j*(n_j-1)/2 equality checks if applied across every vocabulary entry. A small prefix histogram/list can change that work but adds construction/probing/ownership cost;none is free.',
        all_baselines_are_logical_references_not_universal_necessary_traffic=True,overall_complete=False))
    adversary=[]
    for V,L in [(3,1),(3,2),(4,3),(8,5)]:
        x=[0.5,0.0]+[-1024.0]*(V-2);h=[2]*L
        def decision(a,t):
            c=hist(t,V);return argmax([f32(f32(z)-f32(c[v]))for v,z in enumerate(a)])
        origin=decision(x,h);assert origin==0
        for i in range(L):
            alt=list(h);alt[i]=0;assert decision(x,alt)==1
        for v in range(V):
            alt=list(x);alt[v]=-1.0 if v==0 else(float(2**math.ceil(math.log2(L+2)))if v==2 else 1.0)
            assert decision(alt,h)!=origin
        adversary.append(dict(V=V,L=L,x=x,history=h,original_ID=origin,every_individual_logit_or_history_word_can_change_ID=True,
            necessary_uncached_materialized_FLOAT_input_bytes=4*V+8*L,
            necessary_uncached_retained_BF16_input_bytes=2*V+8*L))
    write(O/'penalty-cold-query-symbolic-ceiling-v110.json',dict(status='SYMBOLIC_CEILING',
        domain='One row,V>=3,L>=1 andL<=2^20,finite BF16-representable logits andINT64 output-history tokens0..V-1;rho1,freq1,presence0;exact argmax in word-query domain;no prior counts/summaries/residency. Both arrays must cross one declared storage boundary. Choose exactly one stored-input interface:already materialized FLOAT words or retained BF16 words with exact widening.',
        proof='At history all2 and logits[0.5,0,-1024,...],ID0 wins. Changing any unread history position to0 makesID1 win. Any unread logit can also changeID:lowerx0 to-1,raisex1/other to1,orraisex2 to power2>L+1. All witness logit alternatives are BF16-representable. Thus exact algorithms in this query domain must inspect allV+L words;counts and arithmetic are exact in the stated range.',
        reference_adversaries=adversary,necessary_bytes=dict(materialized_FLOAT='4V+8L',retained_BF16='2V+8L'),
        time_lower_bound='T >= (s_logit*V+8L)/U_selected_storage_capacity_upper,where s_logit=4 or2 for the selected input graph',
        effective_input_word_throughput_upper='(V+L)*U_selected_storage_capacity_upper/(s_logit*V+8L)',
        not_applicable_to='Warm cached/request counts,producer-fused implicit logits,compressed inputs,precomputed summaries,different input/query domain or all algorithms. New state representation requires a new bound.',
        actual_storage_capacity_or_native_MFU_or_speedup='UNKNOWN',native_measurement_not_lower_bound=True,overall_complete=False))
    write(O/'penalty-state-and-composition-v110.json',dict(candidate='F35',
        state_contract=['Prompt presence immutable only within the exact request/prompt epoch;slot reuse is not identity.',
            'Commit actual emitted tokens after verification,including recovered/bonus tokens where emitted;do not persist unaccepted draft proposals. On rollback/truncation remove exact multiplicities or rebuild.',
            'Cbase excludes current new draft;delta is spec[:d] with actualcu/request owners,includingzero-draft gaps.',
            'Keep parameter versions,capture graph/pool/addresses,async old readers,new writers and retire/publication ordering.',
            'Persistent count/bitmap banks extend lifetimes;requested per-call array differences are not peak memory credits without concurrent allocator/pool retirement proof.',
            'Count changes/bitset writes require race-free synchronization. I32overflow is outside the reference domain;INT32→FLOAT conversion remains native policy.',
            'Errors/placeholder handling/noncontiguous copies and CPU-to-NPU readiness remain where externally observable.'],
        v108_extension='Penalty may produce the pre-MinTokens row consumed by A/B/H after allowed/bad-word effects;finite forcing remains after restoration. All score/probability/state readers remain gates.',
        gains_if_no_penalties_true=0,actual_no_penalties_flag_attested=False,
        exact_model_parameter_or_native_shapes_not_promoted=True,
        remaining=['Fallback actualrowdomain/reachability and vllm._custom_ops.apply_repetition_penalties Ascend dispatch/numerics remain named questions;do not infer newsourcecodeisloaded.',
            'Other individually named processor/native-bank/readers questions remain;this is one exact call-site subgraph,not the full sampler universe.',
            'Original9nativeoperand,phase,PMU,attribution and allmodulephysicalcapacity gaps remain.'],overall_complete=False))
    previous=[O/'coverage-current-obligations-v109.json',O/'F35-sampling-condition-ledger-v28.json',O/'heads-sampling-fusion-costs-v65.json',
        O/'stop-mask-fusion-algebra-v108.json',O/'recovery-owner-plan-and-state-reuse-v101.json']
    previous += [N/'penalty-fallback-source166-v110.json',O/'penalty-fallback-source-index-v110.json',O/'penalty-public-contract-index-v110.json',N/'penalty-public-contract-web-evidence-v110.json']
    previous += [Path(r['path'])for r in read(O/'penalty-public-contract-index-v110.json')['sources']if r['status']=='READ']
    previous += [Path(r['local_path'])for r in read(O/'penalty-fallback-source-index-v110.json')['files']if r['status']=='READ']
    inputs=list(dict.fromkeys(files+previous+[Path(p['path'])for p in parts]))
    write(O/'penalty-histogram-inputs-v110.json',dict(inputs=[pin(p)for p in inputs],
        new_device_work=False,new_model_executions=0,overall_complete=False))
    print(json.dumps(dict(status='ANALYZED_OFFLINE',request_contexts=request_cases,prefix_row_checks=row_checks,
        ragged_cases=len(ragged),invalid_rewrite_witnesses=4,FLOAT_reference_witnesses=2,profile_label_matches=len(matches),new_native_promotions=0)))

if __name__=='__main__':main()
