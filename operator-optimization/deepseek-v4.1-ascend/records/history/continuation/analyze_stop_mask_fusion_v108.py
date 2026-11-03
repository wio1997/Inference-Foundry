"""Operator-only MinTokens/forced-value/argmax algebra and conditional I/O bounds."""
from pathlib import Path
from itertools import product
import ast,hashlib,json,math
from finalize_origin_coverage_v9 import N,O,W,read,write

NEG=-float('inf')
def best(pairs):return max(pairs,key=lambda p:(p[0],-p[1]),default=(NEG,10**30))
def oracle(xs,stops,structured,forced):
    before=list(xs);ys=[NEG if i in stops else x for i,x in enumerate(xs)]
    if structured and all(x==NEG for x in ys):
        for i in stops:
            if math.isfinite(before[i]):ys[i]=before[i]
    for i in forced:ys[i]=1e9
    return best((x,i)for i,x in enumerate(ys))[1],ys

def summary(xs,stops,forced,start,end):
    # This is mathematical candidate-state notation,not a target kernel implementation.
    a=[];b=[];h=False
    for i in range(start,end):
        y=NEG if i in stops else xs[i]
        h=h or y!=NEG
        if i not in forced:
            a.append((y,i))
            if i in stops and math.isfinite(xs[i]):b.append((xs[i],i))
    return best(a),best(b),h
def merge(x,y):return best([x[0],y[0]]),best([x[1],y[1]]),x[2]or y[2]
def finish(s,structured,forced):
    a,b,h=s
    chosen=best([a,b])if structured and not h else a
    return best([chosen]+[(1e9,i)for i in forced])[1]
def lazy(xs,stops,structured):
    unmasked=[i for i in range(len(xs))if i not in stops]
    a=best((xs[i],i)for i in unmasked);loads=len(unmasked)
    if a[0]!=NEG:return a[1],loads
    if structured:
        b=best((xs[i],i)for i in stops if math.isfinite(xs[i]));loads+=len(stops)
        if b[0]!=NEG:return b[1],loads
    return 0,loads

def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def source(path,names):
    path=Path(path);text=path.read_text();tree=ast.parse(text);segments=[]
    for name in names:
        node=next(n for n in ast.walk(tree)if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))and n.name==name)
        segments.append(dict(name=name,first_line=node.lineno,last_line=node.end_lineno,source=text.splitlines()[node.lineno-1:node.end_lineno]))
    root=next(p for p in path.parents if p.name=='frozen-source-snapshot');manifest=root/'SOURCE-MANIFEST.json'
    entry=next(r for r in read(manifest)['files']if r['path']==str(path.relative_to(root)))
    record=pin(path);assert entry['sha256']==record['sha256']
    return dict(**record,manifest=pin(manifest),manifest_match=True,functions=segments)

def main():
    assert not(O/'REVIEW-CHECKPOINT-v108.json').exists()
    upstream=W/'frozen-source-snapshot/vllm/vllm/v1/sample'
    ascend=W/'sampler-source-supplement/frozen-source-snapshot/vllm-ascend/vllm_ascend/sample/rejection_sampler.py'
    parts=[source(upstream/'logits_processor/builtin.py',['_mask_stop_token_logits','apply_with_spec_decode']),
        source(upstream/'ops/bad_words.py',['_apply_bad_words_single_batch','apply_bad_words_with_drafts']),
        source(upstream/'thinking_budget_state.py',['apply_to_logits','_apply_forcing_to_logits']),
        source(upstream/'rejection_sampler.py',['_combine_outputs_with_spec_tokens']),
        source(ascend,['apply_logits_processors'])]
    write(O/'stop-mask-source-contract-v108.json',dict(parts=parts,
        facts=[dict(fact='MinTokens mask count for request j is min(max(min_tokens-current_length,0),num_draft_tokens[j]);only that leading prefix gets stop masking.',status='DOCUMENTED'),
            dict(fact='Restoration is only for structured-output stop entries whose original PRE-MinTokens logits are finite and whose entire masked row is -inf.',status='DOCUMENTED'),
            dict(fact='Earlier allowed-token/bad-word exclusions are already present in PRE-MinTokens input;restoring raw pre-processor scores would be different.',status='DOCUMENTED'),
            dict(fact='Thinking forcing runs AFTER MinTokens;writes finite 1e9 at selected indices,not an infinite priority or direct token-ID assignment.',status='DOCUMENTED'),
            dict(fact='Thinking forcing also changes force_index/bonus_token_forced and row layout metadata;ID-only projection cannot remove these state effects.',status='DOCUMENTED')],
        runtime_MinTokens_structured_or_forcing_flags_attested=False,all_processors_or_native_precision_reviewed=False,
        new_target_source_files=0,no_framework_wide_research=True,overall_complete=False))
    write(O/'stop-mask-fusion-algebra-v108.json',dict(
        input_domain='A valid nonempty row of pre-MinTokens FLOAT values in finite real values plus +/-inf,noNaN;valid stop/force indices;unique field and row epochs. Lowest global ID wins equal maxima,as an explicit required comparator contract.',
        original_order=['Earlier penalties/allowed/badwords produce x','mask S to -inf',
            'if structured AND every masked-row entry is -inf:restore only finite pre-MinTokens x on S',
            'later forced slots overwrite to FLOAT 1e9','argmax'],
        single_pass_summary=dict(A='argmax over masked y at indices NOT overwritten by forcing,including -inf with its global ID',
            B='argmax over finite PRE-MinTokens x at restorable stop indices NOT overwritten by forcing',
            H='OR over ALL pre-forcing masked y of y != -inf;includes forced positions too',
            merge='Independent comparator merges for A and B;Boolean OR for H;global ID tie order retained.',
            finish='If structured and not H,merge A and B;otherwise use A. Then merge constant(1e9,forcedID) candidates.',
            empty_candidate_identity='(-inf,+infinite_ID);an empty candidate is not the real all--inf row index0.'),
        no_forcing_lazy_alternative=dict(first='Read only unmasked entries and reduce their maximum.',
            second='Only if all unmasked entries are -inf and structured:read stop entries,select among finite values.',
            otherwise_all_inf='If no restored finite value exists,return lowest global index0;not lowest unmasked index.',
            read_words='(V-|S|)+I(structured AND no unmasked non--inf)*|S|',
            limits='Data-dependent reference graph;metadata/control/latency/parallelism and demanded logit observers remain charged. Not evidence of actual values.'),
        numerical_scope='No arithmetic reassociation is introduced by max/index/Boolean summary in this declared domain. Native NaN/signed-zero/tie/index semantics and actual FLOAT forcing value must still be bound.',
        allowed_rewrite='A selected candidate graph can combine row-predicate/reduction and ID production,or split cold restoration after the first reduction. Full mutated-score outputs must still be produced for live observers.',
        candidate='F35',new_native_mapping_or_physical_gain=False,overall_complete=False))
    values=[NEG,-2.0,0.0,2.0,float('inf')];cases=0;lazy_cases=0;merges=0
    for xs in product(values,repeat=4):
        for sm in range(16):
            stops={i for i in range(4)if sm>>i&1}
            for structured in [False,True]:
                expected,_=oracle(xs,stops,structured,set());got,n=lazy(xs,stops,structured)
                assert got==expected and 0<=n<=4;lazy_cases+=1
                for forced in [set(),{0},{1},{2},{3},set(range(4))]:
                    expected,_=oracle(xs,stops,structured,forced)
                    s=summary(xs,stops,forced,0,4)
                    assert finish(s,structured,forced)==expected
                    for cut in [1,2,3]:
                        joined=merge(summary(xs,stops,forced,0,cut),summary(xs,stops,forced,cut,4))
                        assert joined==s and finish(joined,structured,forced)==expected;merges+=1
                    cases+=1
    witnesses=[]
    def witness(name,x,S,u,F,incorrect_id,why):
        i,y=oracle(x,set(S),u,set(F));assert i!=incorrect_id
        encoded=[('-inf'if z<0 else '+inf')if math.isinf(z)else z for z in x]
        witnesses.append(dict(name=name,x=encoded,stop_ids=S,structured=u,force_ids=F,correct_id=i,incorrect_id=incorrect_id,reason=why))
    witness('Drop structured restoration',[NEG,4.0],[1],True,[],0,'Masking both tokens leaves all -inf;original restores finite stop1 and returns1.')
    witness('Restore raw scores before earlier exclusions',[NEG,4.0],[1],True,[],0,'If raw token0 was8 but an earlier exclusion set it to -inf,restoring raw8 would resurrect a forbidden token.')
    witness('Treat finite force as absolute token-ID',[1e10,0.0],[],False,[1],1,'Other score1e10 exceeds finite forced1e9.')
    witness('Max old winner with force instead of overwrite',[2e9,1.5e9],[],False,[0],0,'Force overwrites old2e9 with1e9;old winner must be removed before final reduction.')
    witness('Move force before restoration predicate',[2e9,NEG],[0],True,[1],1,'Original restores stop0 to2e9 before force1=1e9. A pre-applied force would prevent restoration.')
    witness('Drop forced positions from pre-forcing H',[2e9,2.0],[0],True,[1],0,'H=true from y1=2 prevents stop0 restoration. Excluding forced1 from H incorrectly restores2e9 at0.')
    write(O/'stop-ma<REDACTED_TOKEN>.json',dict(status='PASS',oracle_cases=cases,lazy_no_forcing_cases=lazy_cases,
        split_merge_checks=merges,witnesses=witnesses,CPU_only=True,
        finite_grid_is_not_universal_or_native_proof=True,NaN_and_runtime_flags_unattested=True,overall_complete=False))
    costs=[]
    for T,R in [(10,0),(10,1),(10,10),(20,1),(20,20)]:
        V=read(O/'heads-sampling-fusion-costs-v65.json')['sampling_models'][0]['V']
        costs.append(dict(T=T,structured_restore_rows=R,V=V,
            separately_materialized_gather_predicate_all_argmax_reference_bytes=4*T*V+14*R*V+R,
            chosen_fullpass_ID_summary_FLOAT_input_reference_bytes=4*T*V,
            removable_reference_interfaces_before_summary_overhead_bytes=14*R*V+R,
            extra_summary_index_flag_metadata_and_sparse_state_costs_not_zero=True,
            shape_reference_not_actual_R_or_current_gain=True))
    write(O/'stop-mask-work-and-ceiling-v108.json',dict(models=costs,
        baseline_definition='Explicit separate FLOAT gather[RV]:read4RV/write4RV;isneginf:read4RV/writeRV;all:readRV/writeR;argmax:read4TV. Sparse stop saves/restores/scatters and metadata excluded symmetrically and remain separately charged.',
        logical_reference_not_native_kernel_fusion_or_HBM_traffic=True,
        no_forcing_read_tradeoff='One-pass two-candidate summary may read V words. Lazy unmasked-first reads n and only conditionally s;neither schedule latency is a lower bound.',
        conditional_lower_bound=dict(status='SYMBOLIC_CEILING',
            domain='No forcing;no structured restoration;at least n=V-|S|>=2 arbitrary finite unmasked FLOAT values,exact ID,comparison/query domain,no prior input summaries. All n words must cross the declared storage boundary from no initial residency. Bound is worst case over this domain,not each input/all algorithms.',
            necessary_words='n',necessary_comparisons='n-1',
            time_bound='T >= max(4*n/U_selected_storage_upper,(n-1)/U_comparator_upper)',
            throughput_bound='n/T <= n/max(4*n/U_selected_storage_upper,(n-1)/U_comparator_upper)',
            capacities='UNKNOWN at actual native precision/clock/tier;no numeric wallclock,MFU or speedup.',
            structured_extension='When both n>=2 and s>=2,an adversarial all-unmasked--inf/finite-stop domain can require reading n+s words;do not use this as a per-instance V-word lower bound. Other domain/force cases need separate analysis.',
            resource_composition='Take max of independent resource lower bounds. Selected storage boundary/initial residency and shared credits must be declared once for the chosen larger graph.'),
        actual_native_applicability_or_causal_gain_proved=False,overall_complete=False))
    write(O/'stop-mask-composition-gates-v108.json',dict(candidate='F35',related_existing_evidence=['F35-sampling-condition-ledger-v28.json','heads-sampling-fusion-costs-v65.json','sampling-case-fusion-and-ceiling-routing-v102.json'],
        conditions=['Actual MinTokens/structured/force rows and row-owner/stop indices/buffer epochs',
            'Complete raw/processed logits/logprobs,entropy,random/recovery and mutation reader sets',
            'Preserve earlier penalty/allowed/bad-word effects and FLOAT input rounding',
            'Preserve post-MinTokens forcing order,finite overwrite value and CPU force_index/bonus_token_forced state effects',
            'Preserve full scores for demanded observers;ID-only graph does not remove their work',
            'Source CPU errors/index validation and capture/async publication retained',
            'One union against one baseline with existing F35 clone/cast/sampling credits;no additive double count'],
        no_global_row_barrier_required='Mergeable A/B/H summary allows tile consumption;the row-level restoration choice needs global predicate before final ID publication,not necessarily a materialized full-row buffer.',
        zero_current_saving_if_branch_inactive=True,actual_branch_attested=False,
        remaining_named_questions=['Penalty count/mask construction body and any other specific processor reader still require prior-answer check;not covered by this MinTokens subgraph.',
            'Native comparator/NaN/precision and actual graph/storage/parameter identities remain unbound.'],
        no_new_candidate_pair_or_whole_branch_closure=True,overall_complete=False))
    inputs=[O/'coverage-current-obligations-v107.json',O/'F35-sampling-condition-ledger-v28.json',
        O/'heads-sampling-fusion-costs-v65.json',O/'sampling-case-fusion-and-ceiling-routing-v102.json']
    write(O/'stop-mask-inputs-v108.json',dict(inputs=[pin(p)for p in inputs]+[pin(p['path'])for p in parts],
        saved_profile_applicability='Prior F35 native chains reused only as boundary context;no new native MinTokens/force attribution.',new_device_work=False))
    print(json.dumps(dict(status='ANALYZED_OFFLINE',source_files=5,oracle_cases=cases,lazy_cases=lazy_cases,merge_checks=merges,new_native_promotions=0)))

if __name__=='__main__':main()
