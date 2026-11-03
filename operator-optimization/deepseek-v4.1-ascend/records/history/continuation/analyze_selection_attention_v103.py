"""Selection/attention frontier: finalized winners, speculative per-key work and retractable states."""
from pathlib import Path
from collections import Counter, defaultdict
from fractions import Fraction as F
from itertools import permutations, product
import hashlib, json, struct
from finalize_origin_coverage_v9 import N,O,W,read,write

def pin(p):
    p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def f32(x):return struct.unpack('>f',struct.pack('>f',float(x)))[0]
def bf16_rne(x):
    b=struct.unpack('>I',struct.pack('>f',float(x)))[0]
    return struct.unpack('>f',struct.pack('>I',(b+0x7fff+((b>>16)&1))&0xffff0000))[0]
def top(ids,scores,k):return sorted(ids,key=lambda i:(-scores[i],i))[:k]
def partitions(n,p):return [list(range(i*n//p,(i+1)*n//p))for i in range(p)]
def section(p,ranges):
    ls=Path(p).read_text().splitlines();return dict(**pin(p),segments=[dict(first_line=a,last_line=b,source=ls[a-1:b])for a,b in ranges])

def main():
    assert not(O/'REVIEW-CHECKPOINT-v103.json').exists()
    prior=['attention-output-owner-frontiers-v100.json','attention-indexer-native-numerical-contract-v60.json',
        'attention-indexer-fusion-reorganization-variants-v60.json','index-context-operator-contract-v64.json',
        'attention-composition-clauses-v83.json','index_attention-distinct-score-proof-v83.json',
        'attention-projected-state-source-contract-v100.json','whole-scope-requirement-audit-v102.json']
    root=N/'attention-operator-source-v60/vllm-workspace/vllm-ascend'
    qli=root/'csrc/attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_service_vector_arch22.h'
    csa=root/'csrc/attention/sparse_flash_mla/op_kernel/arch22/sparse_flash_mla_csa_block_vector.h'
    prepare=root/'vllm_ascend/ops/triton/prepare_indexer_indices.py'
    src100=read(O/'attention-projected-state-source-contract-v100.json')
    dsa=Path(next(r['path']for r in src100['parts']if Path(r['path']).name=='dsa_v41.py'))
    parts=[section(qli,[(448,453),(843,881)]),section(csa,[(419,448),(454,489),(919,931),(844,857)]),
        section(prepare,[(35,56)]),section(dsa,[(613,686),(704,724)])]
    old=read(O/'attention-indexer-native-numerical-contract-v60.json')
    for p in parts:
        candidates=[r for r in old['parts']if r['path']==p['path']]+[r for r in src100['parts']if r['path']==p['path']]
        if candidates:assert p['sha256']==candidates[0]['sha256']
    facts=[
        'QLI final position scores include native key-scale/membership arithmetic before TopK; candidate-block producer and current candidate sharing are separate persistent outputs, not removed by composing with attention.',
        'Prepare checks nonnegative/visible positions, sorts chronologically and writes valid prefix/-1 suffix. Score-ranked provisional order is not the current SMLA input order.',
        'QLI INT8/head-weight score is distinct from attention BF16 QK. Early attention work still needs its own query/key arithmetic; source v83 counterexample is reused, not a new finding.',
        'SMLA SoftmaxFlashV2 propagates max/sum and rescale state, casts exponential tiles to KV_T before PV, and rescales/adds previous PV state. Final BF16 output uses CAST_RINT. A change in winner set or selected-tile membership is not automatically an append-only update.',
        'Existing v60/v83 final-QLI/prepare/direct handoff and SWA-before-Indexer candidates remain baseline prior proposals. This refinement concerns provisional winner replacement and per-item attention-logit speculation.',
        'Current profile API exposes neither provisional winner events nor per-tile readiness. New per-key cache/certificates/state are analysis interfaces, not existing native ABIs.'
    ]
    write(O/'selection-attention-source-contract-v103.json',dict(parts=parts,facts=facts,
        new_target_source_files=0,new_public_documents=0,new_server_or_device_work=False,
        no_framework_research=True,actual_loaded_native_arithmetic_or_new_ABI_proved=False,overall_complete=False))

    prior_front=read(O/prior[0]);rows=[];unique=set();mode_counts=Counter();families=Counter()
    for f in prior_front['frontiers']:
        s=f['index_selection']
        if s is None:continue
        stages=dict(QLI=s['QLI'],prepare=s['prepare'],SMLA=f['stages']['SMLA'])
        assert stages['QLI']['endNs']<=stages['prepare']['startNs']<=stages['prepare']['endNs']<=stages['SMLA']['startNs']
        assert len({r['streamId']for r in stages.values()})==1
        for r in stages.values():unique.add((f['database_index'],r['task_record_rowid']))
        rows.append(dict(database_index=f['database_index'],graph_ordinal=f['graph_ordinal'],block_ordinal=f['block_ordinal'],
            root_bound=f['root_bound'],family=f['family'],mode=s['mode'],stages=stages,
            observed_QLI_start_to_SMLA_end_ns=stages['SMLA']['endNs']-stages['QLI']['startNs'],
            native_task_order_only_not_fused_latency_bound=True))
        mode_counts[s['mode']]+=1;families[f['family']]+=1
    assert len(rows)==640 and len(unique)==1920
    write(O/'selection-attention-native-frontiers-v103.json',dict(frontiers=rows,count=640,root_bound=512,phase_unbound=128,
        reused_unique_TASKs=1920,modes=dict(mode_counts),families=dict(families),
        source='attention-output-owner-frontiers-v100.json',new_regime_or_assignment_promotions=0,
        limitation='Saved same-graph task-order frontiers only. No selected IDs, per-tile global-selection certificate, QK cache identity or counterfactual readiness/critical-path savings is observed.',overall_complete=False))

    enum=0;updates=0
    for n in range(2,7):
        for values in permutations(range(n)):
            for k in range(1,min(3,n-1)+1):
                final=top(range(n),values,k)
                resident=[];ever=set()
                for i in range(n):
                    after=top(resident+[i],values,k)
                    if i in after:ever.add(i);updates+=1
                    resident=after
                assert resident==final and set(final)<=ever and k<=len(ever)<=n
                for p in range(1,n+1):
                    local=[top(ids,values,k)for ids in partitions(n,p)]
                    union=[i for ids in local for i in ids]
                    assert top(union,values,k)==final
                    assert len(union)==sum(min(k,len(ids))for ids in partitions(n,p))
                    enum+=1
    # Same processed prefix permits incompatible winners without a bound on the unseen score.
    prefix=[4,3];low=[4,3,-1];high=[4,3,5]
    write(O/'selection-winner-readiness-and-cover-v103.json',dict(
        comparison_domain='Finite independent complete position scores with a fixed deterministic total comparator; examples use descending score then ascending ID, not an attestation of native SortAll ties.',
        unseen_score_witness=dict(processed_prefix=prefix,K=1,unseen_low_completion=low,unseen_high_completion=high,
            low_final=top(range(3),low,1),high_final=top(range(3),high,1),
            conclusion='A provisional winner is not certified solely by finishing its source tile; a possible unseen greater score can evict it.'),
        local_union_theorem='With the same complete-score total comparator, every global TopK item is in its partition local TopK. Otherwise K same-partition predecessors would also globally precede it. Merging these lists preserves the selected set.',
        finite_partition_and_rank_cases=enum,tentative_insertion_events_checked=updates,
        speculative_heap_contract='Keep attention logits only for current provisional TopK residents, with entry generation/readiness and outstanding-reader guards. Every final winner was admitted; up to N entries can be computed over time, though at most K resident score rows are needed in this chosen sequential-resident representation.',
        finalized_tile_contract='Early irreversible attention updates need certified membership and an original-order complete selected tile, or a separately proved numerical contract. Bounds/certificates, N<=K selected-set domains, or additional stored/recomputed data can allow progress before a whole operator finishes.',
        no_universal_whole_operator_barrier=True,
        lower_bound_domain='For 1<=K<N independent arbitrary scores, exact selected-set output without prior bounds/certificates has worst-case N score observations and at least N-K comparisons in a comparison model. This does not prove N native INT8 GEMMs or a full-kernel time barrier.',
        all_native_ties_values_or_reachability_proved=False,overall_complete=False))

    # Real arithmetic supports subtraction; rounded summary need not retain that information.
    real_cases=0
    for masses in product([F(1,8),F(1,2),F(1)],repeat=3):
        for vals in product([F(-2),F(0),F(1)],repeat=3):
            l=sum(masses);a=sum(x*v for x,v in zip(masses,vals))
            for i in range(3):
                other=[j for j in range(3)if j!=i]
                assert (a-masses[i]*vals[i])/(l-masses[i])==sum(masses[j]*vals[j]for j in other)/sum(masses[j]for j in other)
                real_cases+=1
    eps=2**-25
    before=[dict(remaining_value=v,m=0,l=f32(1+eps),a=f32(1+eps*v))for v in [0,1]]
    assert before[0]['l']==before[1]['l']==before[0]['a']==before[1]['a']==1
    a=.5019;c=.75;denom=f32(1+a*c)
    native_order_reference=bf16_rne(f32(bf16_rne(a*c)/denom))
    speculative_order_reference=bf16_rne(f32(bf16_rne(a)*c/denom))
    assert native_order_reference!=speculative_order_reference
    write(O/'selection-attention-retraction-and-rounding-v103.json',dict(
        exact_real_reference=dict(anchor='fixed common m',l='sum exp(z_i-m)',a='sum exp(z_i-m)*v_i',
            remove='l_new=l-exp(z_j-m); a_new=a-exp(z_j-m)*v_j',positive_mass_finite_cases=real_cases,
            caveat='Exact arithmetic identity; need removed item contribution, nonempty remaining mass and a usable anchor. Not a FLOAT32 native subtraction equivalence.'),
        rounded_state_collision=dict(masses=[1,eps],removed_item=dict(mass=1,value=1),histories=before,
            after_correct_remaining_outputs=[0,1],rounded_subtraction_l=0,rounded_subtraction_a=0,
            conclusion='The same rounded (m,l,a), even with the same removed item, can correspond to different remaining outputs. This summary alone cannot support exact eviction; retained per-item data or a reread/recompute/certified representation is needed.',
            scope='Declared FLOAT32 round-to-nearest scalar summary model, not actual C8 values or a claim against every possible native/augmented state.'),
        append_only_eviction_witness=dict(initial_selected_ID=0,initial_value=0,replacement_selected_ID=1,replacement_value=1,
            equal_attention_masses=True,append_only_output=.5,correct_K1_replacement_output=1),
        provisional_BF16_rebase=dict(old_relative_exp=a,new_anchor_factor=c,
            rounded_old_exp=bf16_rne(a),correct_new_tile_exp=bf16_rne(a*c),rescaled_old_exp=bf16_rne(a)*c,
            denominator_reference=denom,fresh_final_tile_output=native_order_reference,speculative_rebased_output=speculative_order_reference,
            scope='Finite round-to-nearest BF16/FLOAT32 reference away from BF16 ties, not a measured CANN Exp/CAST_ROUND or SoftmaxFlashV2 result. Changing final selected tile/anchor can cross a quantization boundary.'),
        conclusion='Speculate per-item QK/logits rather than blindly accumulating provisional PV. Replay the final chronological tile/softmax/BF16/PV contract, or declare and prove a changed precision domain.',
        all_native_algorithms_excluded=False,overall_complete=False))

    models=[];H,D,K,S=64,512,512,128
    for length in [256,512,513,2048,16384,34816]:
        k=min(K,length)
        for p in [1,2,4,8,16,64]:
            A=sum(min(k,len(ids))for ids in partitions(length,p))
            baseline=4*H*D*k;new=2*H*D*(A+k)
            models.append(dict(N=length,K_capacity=K,K_valid_reference=k,partitions=p,H=H,D=D,SWA_keys_reference=S,
                local_topk_union_attention_QK_rows=A,selected_compressed_PV_rows=k,
                baseline_compressed_classical_ops=baseline,speculative_compressed_classical_ops=new,
                compressed_compute_work_ratio=F(new,baseline).__str__(),
                all_with_SWA_work_ratio=F(new+4*H*D*S,baseline+4*H*D*S).__str__(),
                partition_union_FLOAT_attention_logit_values_bytes=4*H*A,
                resident_heap_FLOAT_attention_logit_values_bytes=4*H*k,
                worst_case_heap_QK_rows=length,
                baseline_new_private_ID_or_logit_banks_are_not_HBM_or_peak_bytes=True))
    write(O/'selection-attention-speculation-cost-models-v103.json',dict(models=models,model_count=len(models),
        units='Per query; FLOPs count two OPS/MAC in the declared ordinary dense BF16 attention QK/PV algorithm. No indexer work/collective, Vector softmax, packing, cache lifetime or launch cost is subtracted.',
        caveats=['Partitions are logical key partitions, not an automatic TP speedup; current query rows/card ownership remain separate.',
            'Local-union storage is a chosen retained-union representation. Resident-heap storage is a different chosen representation with generation/inflight access guards, not a universal capacity lower bound.',
            'Speculative QK may hide a dependency but performs A>=K rows; PV/softmax remain after required final selection/order. Reduced waiting is not reduced arithmetic or proven wall-clock gain.',
            'Keeping per-item logits avoids lossy PV retraction but still requires native score equality under retiling, final chronological replay, mask/sink-once, state/weight/cache generations and valid ABI.',
            'Existing scalar-score/exp GM queues, selected/candidate publications and v60/v83 interface credits must be revalued once; no extra credit for stages already fused.'],
        theory=dict(status='SYMBOLIC_CEILING',
            lower='max(L_selection_in_declared_score_model, F_INT8_index/P_INT8, F_BF16_attention_chosen_graph/P_BF16, L_Vector, Q_necessary_storage_by_tier/B_tier, L_communication_and_proved_tile_dependencies)',
            qualification='Only necessary work in the chosen graph/domain and valid typed capacity upper bounds can enter. Independent resources may overlap; same-resource necessary work must be combined without duplication. The speculative work ratio is not a speedup bound or a requirement for all algorithms.'),
        actual_valid_lengths_winner_sequences_native_precision_or_physical_speedup_known=False,overall_complete=False))
    write(O/'selection-attention-variants-v103.json',dict(variants=[
        dict(id='FINALIZED-HANDOFF',decision='EXISTING_CONDITIONAL_BASELINE',contract='Existing QLI/prepare/direct stable-bank handoff, preserve original sorted/-1 prefix and other candidate/index readers; no new gain counted.'),
        dict(id='CERTIFIED-TILE-EARLY',decision='CONDITIONAL_NEW_INTERFACE',contract='Only certified final winners with complete original-order tile membership can feed irreversible original attention updates early. Explicit unseen-score bounds/certificates or selected-set special domains, no mandatory full-operator barrier.'),
        dict(id='PROVISIONAL-LOGIT-HEAP',decision='CONDITIONAL',contract='Compute current TopK entrant QK, retain versioned per-key H logits, evict safely, then replay final original tile/softmax/PV. Live entries <=K in chosen schedule; total QK entries K..N. No actual native tile-equivalence proof.'),
        dict(id='LOCAL-TOPK-COVER-LOGITS',decision='CONDITIONAL_WITH_WORK_STORAGE_PENALTY',contract='Compute attention QK for the complete local TopK union; preserve complete indexer score comparator/masks/membership/fillers, then choose final winners and replay. A=sum min(K,n_partition), not extra TP division.'),
        dict(id='ROUNDED-SUMMARY-EVICTION',decision='EXCLUDED_AS_UNCONDITIONAL_SUBSTITUTION',contract='Rounded (m,l,a) plus removed item is insufficient in the reference collision. Full/augmented per-item state or recomputation/tolerance proof is a distinct candidate, not an append-only merge.'),
        dict(id='APPEND-PROVISIONAL-PV',decision='EXCLUDED_AS_UNCONDITIONAL_SUBSTITUTION',contract='A later better Indexer winner replaces an old selected key. Appending both attention contributions computes another set; QLI and attention scores are distinct. Preserve selection and precision contracts.')],
        existing_relation='REL83-08',existing_group='HG83-INDEX-ATTENTION',new_candidate_or_pair=False,
        all_native_or_semantic_branches_closed=False,overall_complete=False))
    write(O/'selection-attention-inputs-v103.json',dict(inputs=[pin(O/f)for f in prior]+[pin(Path(r['path']))for r in parts],
        prior_answers_not_new=['v60/v83 final handoff, key partition and independent SWA overlap','v83 distinct index/attention scores','v100 partial-attention state and numerical preservation'],
        new_scope='Provisional TopK replacement versus irreversible attention updates; certified final tile and per-key-logit speculative graphs/work/representation tradeoffs.',
        no_server_model_Torch_NPU_capture_implementation_or_framework_research=True,overall_complete=False))
    print(json.dumps(dict(status='ANALYZED',frontiers=640,TASKs=1920,partition_cases=enum,real_retraction_cases=real_cases,models=len(models),overall_complete=False)))

if __name__=='__main__':main()
