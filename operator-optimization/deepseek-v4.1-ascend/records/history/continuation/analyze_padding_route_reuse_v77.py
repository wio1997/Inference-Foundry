"""Saved-profile/source-only analysis of fully materialized constant padding routes.

No target changes, imports of Torch/models, remote work, or native experiments.
Byte references are deliberately not asserted to be physical HBM lower bounds.
"""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib, math, struct
from finalize_origin_coverage_v9 import N, O, W, read, write
from mirror_saved_profile_core_v53 import CACHE


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def partition(rows, cores=48):
    per = math.ceil(rows / cores)
    active = math.ceil(rows / per)
    return dict(rows=rows, available_programs=cores, active_programs=active,
                per_program_rows=per, last_program_rows=rows % per or per)


def main():
    assert not (O/'padding-route-domain-v77.json').exists()
    old=read(O/'router-predicate-source-index-v76.json')['source_files']
    suffixes=['moe_gating_top_k_hash_without_group.h','moe_gating_top_k_hash_tiling.cpp',
              'moe_gating_top_k_hash.cpp','ops/fused_moe/router/fused_topk_router.py',
              'ops/fused_moe/prepare_finalize.py']
    sources=[]
    for suffix in suffixes:
        r=next(r for r in old if r['path'].endswith(suffix))
        p=Path(r['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
        sources.append(dict(path=str(p),sha256=r['sha256'],binding='Reused exact frozen operator source receipt; loaded native binary not attested.'))
    consumer=N/'F21-callback-source-v52/01-routed_experts.py'
    sources.append(dict(path=str(consumer),sha256=hashlib.sha256(consumer.read_bytes()).hexdigest(),binding='Reused frozen v52 routed-expert callback source; local source clause, not binary/value attestation.'))
    kernel=Path(sources[0]['path']).read_text();host=Path(sources[1]['path']).read_text()
    router=Path(sources[3]['path']).read_text();prepare=Path(sources[4]['path']).read_text();cons=consumer.read_text()
    for phrase in ['Sort<float, true>','sumValue = maxValueTensor.GetValue(0) + tilingData_->eps','Muls(yOutTensor, yOutTensor, tilingData_->routedScalingFactor','tokenId = tokenId == static_cast<U1>(-1)']:
        assert phrase in kernel
    assert 'int64_t perCoreRows = CeilDiv(rows_' in host
    assert 'eps=1e-20' in router and 'out_flag=False' in router
    assert 'router_logits=nn.functional.pad(router_logits,(0,0,0,pad_size))' in ''.join(prepare.split())
    assert 'enable_force_load_balance = _EXTRA_CTX.in_profile_run' in cons
    assert cons.index('topk_ids = self.log2phy[topk_ids]') < cons.index('elif enable_force_load_balance:')

    pads=[r for r in read(O/'pad-owner-invocations-v72.json')['invocations'] if r['field']=='MC2_ROUTER']
    guards=read(O/'router-predicate-guard-models-v76.json')['regimes']
    target=next(r for r in guards if r['M']==816 and r['experts']==384)
    assert len(pads)==320 and target['executions']==640
    cap=read(CACHE/'analysis/profile-capabilities-index-v12-v2.json')
    groups=[];pairs=[]
    for db in sorted({r['database_index']for r in pads}):
        x=read(CACHE/f'analysis/native-execution-identity-v14/db-{db:02d}.json')
        ts={t['task_record_rowid']:t for t in x['TASK_records']}
        cs=defaultdict(set);apis=defaultdict(list);byid={a['api_rowid']:a for a in x['linked_PYTORCH_APIs']}
        for c in x['CONNECTION_IDS']:cs[c['connectionId']].add(c['id'])
        for a in x['linked_PYTORCH_APIs']:apis[a['connectionId']].append(a)
        roots=[r for r in cap['databases'][db]['scopes']if r['name'].startswith('wio_hostdiag::execute_model')]
        buckets=defaultdict(lambda:dict(pad=[],topk=[]))
        def add(kind,a,record):
            rs=[r for r in roots if r['globalTid']==a['globalTid']and r['startNs']<=a['startNs']and a['endNs']<=r['endNs']]
            assert len(rs)==1
            buckets[rs[0]['api_rowid']][kind].append((a,record))
        for p in pads:
            if p['database_index']==db:add('pad',byid[p['owner_API_rowid']],p)
        for d,tid in target['native_TASK_keys']:
            if d==db:
                aa=[a for c in cs[ts[tid]['connectionId']]for a in apis[c]if a['name_text']=='aclnnMoeGatingTopKHash']
                assert len(aa)==1;add('topk',aa[0],tid)
        for rootid,b in buckets.items():
            pp=sorted(b['pad'],key=lambda z:z[0]['startNs']);tt=sorted(b['topk'],key=lambda z:z[0]['startNs'])
            assert len(pp)==len(tt)==40
            root=next(r for r in roots if r['api_rowid']==rootid)
            for i,((pa,p),(ta,tid)) in enumerate(zip(pp,tt)):
                assert pa['endNs']<=ta['startNs']
                if i+1<len(pp):assert ta['endNs']<=pp[i+1][0]['startNs']
                pairs.append(dict(database_index=db,execute_model_API_rowid=rootid,source_order_ordinal=i,
                                  pad_owner_API_rowid=pa['api_rowid'],pad_MemSet_TASK_rowid=p['MemSet_TASK_rowid'],
                                  TopK_owner_API_rowid=ta['api_rowid'],TopK_TASK_rowid=tid,
                                  binding_grade='COMPLETE_SAME_THREAD_ROOT_CARDINALITY_AND_SOURCE_ORDER;NOT_POINTER_DEF_USE',
                                  native_input_pointer_or_layer_parameter_ID_attested=False))
            groups.append(dict(database_index=db,root=root,pad_owners=40,TopK_owners=40,
                               all_pad_then_TopK_then_next_pad_intervals=True,
                               repeated_same_layer_cache_uses_proved=False))
    assert len(groups)==8 and len(pairs)==len({(r['database_index'],r['TopK_TASK_rowid'])for r in pairs})==320
    total_elements=sum(r['M']*r['experts']*r['executions'] for r in guards)
    assert total_elements==306816000
    write(O/'padding-route-domain-v77.json',dict(known_MC2_router_pad_calls=320,
        original_valid_rows=800,padded_rows=816,experts=384,top_k=6,groups=groups,source_order_correspondences=pairs,
        M816_TopK_population=640,all_M816_calls_assumed_padded=False,
        explicit_padding_score_elements=320*16*384,all_observed_TopK_score_elements=total_elements,
        element_reference_fraction=320*16*384/total_elements,
        invocation_element_reference_fraction=16/816,
        cache_hits_or_request_time_fraction_proved=False,source_zero_logit_and_ID0_domain=True,
        functional_assignments_rewritten=False,overall_complete=False))

    phi=f32(math.sqrt(f32(math.log(f32(f32(math.exp(0.0))+1.0)))))
    summed=0.0
    for _ in range(6):summed=f32(summed+phi)
    ratio=f32(phi/f32(summed+1e-20));algebraic=f32(1/6)
    assert ratio!=algebraic
    b0=0.0;b1=2**-30
    assert b1>b0 and f32(phi+b0)==f32(phi+b1)
    write(O/'padding-route-semantic-proof-v77.json',dict(
        domain='Pinned nohash without-group source, normType2, FLOAT logits+weights/textbias, INT64 IDs, E384/k6, positive sentinel129257..129261, out_flagFalse. Literal +0 logits and ID0 only on explicitly identified alignment tail.',
        source_claims=[
            dict(kind='DERIVED',claim='ID0 chooses text bias; current nohash path has no row/core-dependent arithmetic apart from input/output addressing.'),
            dict(kind='DERIVED',claim='Literal-zero tail result is a function of text-bias value generation, all scalar attributes, dtype/native numerical and tie semantics. Router GEMM/FFN weights are not inputs to this literal-zero subproblem.'),
            dict(kind='DERIVED',claim='Exact native result reuse can avoid repeated row nonlinear/rank/weight calculation while still emitting every required raw P×k ID/weight before all unchanged postprocessors.'),
            dict(kind='UNKNOWN',claim='Loaded native determinism, exact bias values/scaling, cross-shape equivalence and native result bank are not attested; no actual template values obtained.')],
        reference_FP32_witness=dict(phi=phi,six_term_sum=summed,stage_rounded_weight=ratio,algebraic_one_over_k=algebraic,
            reference_model='Python binary32 rounding after explicit scalar stages. Not Ascend Exp/Ln/Sqrt/reduction measurement; demonstrates why algebra alone is insufficient.'),
        ranking_witness=dict(bias=[b0,b1],rounded_phi_plus_bias=[f32(phi+b0),f32(phi+b1)],
            implication='Bias-only sorting can reverse winner relative to equal rounded ranking scores and a stable input-index tie rule; do not replace native ranking by bias-only argsort.'),
        retained_precision_gates=['Keep Exp→+1→Ln→Sqrt stage semantics, unbiased gather, selected-k sum+eps, Div and scaling; do not inject1/k.',
                                  'Keep native tie/NaN/Inf and initialized expert-index semantics. Cache exact output bits rather than CPU-derived approximate results.',
                                  'out_flagFalse full score output is unwritten in source/caller discarded; no E-wide third-output HBM saving claimed.'],
        consumer_frontier=dict(placement='Inside raw inner-router output boundary; materialize full P×k before routed_experts._select_experts postprocessing.',
            unchanged=['logical→physical mapping','shared-expert ID/weight append','forced EPLB','full P-shaped RNG draws and argsort','all observers/capture alias lifetimes','MC2 hidden/mask/output dimensions and collective receiver work'],
            profile_run_flag='in_profile_run is dummy/profile-run lifecycle state; it is not equivalent to Torch profiler being enabled.',
            no_router_logits_after_select_in_scoped_forward=True,all_global_hooks_or_aliases_closed=False),
        overall_complete=False))

    n,p,e,k,d=800,816,384,6,16
    original=4*p*e+8*p+8*p*k+8*e
    integrated=4*n*e+8*n+8*p*k+8*e+8*k
    assembled=4*n*e+8*n+8*n*k+8*n*k+8*k+8*p*k+8*e
    pad_plus_original=4*n*e+4*p*e+original
    assert (original,integrated,assembled,pad_plus_original)==(1302144,1277488,1354288,3784320)
    costs=dict(N=n,P=p,E=e,k=k,original_TopK_reference_bytes=original,
        integrated_prefix_plus_tail_reference_bytes=integrated,integrated_reference_budget_bytes=original-integrated,
        native_prefix_plus_separate_assembly_reference_bytes=assembled,separate_assembly_extra_bytes=assembled-original,
        original_logitsPad_plus_TopK_reference_bytes=pad_plus_original,
        joint_no_logitsPad_prefix_plus_tail_reference_bytes=integrated,joint_reference_budget_bytes=pad_plus_original-integrated,
        joint_already_counted_direct_producer_valid_prefix_copy_budget_bytes=8*n*e,
        joint_incremental_reference_over_that_budget_bytes=pad_plus_original-integrated-8*n*e,
        output_template_bytes=8*k,full_fresh_output_bytes=8*p*k,padding_tail_output_bytes=8*d*k,
        forty_layer_template_payload_bytes_per_rank=40*8*k,
        optional_shared_weight_payload_bytes_per_rank=4*k+40*4*k,
        router_GEMM_padded_row_FLOPs_saved=0,
        bias_reference='Count two unique FLOAT bias planes once,3072B. Actual source loads both once per active program; no claim that unique-value references equal physical traffic.',
        boundary='TopK input logits/IDs, unique biases, materialized weights/IDs and template reference only. Router GEMM/ID preparation, allocation, helper packing, launches/events, state publication and cache generation are excluded and must be priced.',
        partition_original=partition(p),partition_prefix=partition(n),
        fixed_row_loop_model=dict(longest_row_count_before=17,longest_row_count_after=17,
            last_program_valid_rows_before=1,last_program_padding_rows_before=16,last_program_rows_after=1,
            claim='Removing16 rows reduces total work but does not reduce the maximum17-row loop count with48 programs. This fixed homogeneous-row scheduling reference is not a universal latency lower bound or proof of no benefit.'),
        theory=dict(state='SYMBOLIC_CEILING',
            legal_graph='Constant padding removes any necessity for per-use nonlinear/sort on those rows if exact result reuse is permitted. Fresh full raw-output materialization remains in this domain.',
            bound='For a declared legal graph and genuinely necessary resource work Q_r with capacity upper bounds C_r, T>=max_r(Q_r/C_r). Under a fresh dense raw-output store interface requiring every byte, Q_output_store>=8Pk; do not identify this with HBM without a hierarchy/initial-state proof.',
            numeric_Vector_Sort_Ln_Sqrt_store_bandwidth_capacity_known=False,request_time_lower_bound_ns=None,
            reference_bytes_are_not_necessary_HBM=False),
        actual_measured_speedup=None,overall_complete=False)
    # Explicit false flag says these reference bytes ARE NOT a proven HBM bound.
    costs['theory']['reference_bytes_are_proven_necessary_HBM']=False
    costs['theory'].pop('reference_bytes_are_not_necessary_HBM')
    write(O/'padding-route-cost-models-v77.json',costs)
    variants=[
        dict(id='PAD-TOPK-INTEGRATED-CONSTANT-TAIL',mechanism='New valid-prefix/output interface skips tail row calculation and writes exact cached native IDs/weights into full freshPoutput.',reference_bytes=integrated,status='CONDITIONAL_OPERATOR_REORGANIZATION'),
        dict(id='PAD-TOPK-NATIVE-PREFIX-THEN-ASSEMBLE',mechanism='Existing native N-row call followed by separate fullPassembly.',reference_bytes=assembled,status='COST_DISFAVORED_IN_FIXED_ROW_MODEL;NOT_UNIVERSALLY_EXCLUDED'),
        dict(id='PAD-TOPK-JOINT-NO-LOGITSPAD',mechanism='Keep router logits atN and combine valid-prefix consumer+constant-tail output while retaining hidden/maskP.',reference_bytes=integrated,status='CONDITIONAL_F20_DIRECT_PRODUCER_COMPOSITION;NO_DOUBLE_COUNT'),
        dict(id='PAD-TOPK-PERSISTENT-TAIL',mechanism='Reuse already filled tail in persistent raw-output bank.',reference_bytes=None,status='CONDITIONAL_ALL_WRITER_EPOCH;NOT_GENERIC_FRESH_OUTPUT'),
        dict(id='EXCLUDED-ALGEBRAIC-OR-POSTRNG-TEMPLATE',mechanism='Replace weights by1/k, sort bias alone, reuse mapped/randomized outputs, or shrink postprocessor RNG shape.',reference_bytes=None,status='EXCLUDED_WITHOUT_ADDITIONAL_EQUIVALENCE_PROOF')]
    for v in variants:v.update(new_main_candidate=False,actual_measured_speedup=None,native_validated=False)
    write(O/'padding-route-variants-v77.json',dict(variants=variants,variant_count=5,
        existing_candidate='F20 MC2-PAD-TOPK(v63), composed with existing F20 padding and F21 dispatch frontier;not a new main mechanism.',
        required_key=['layer/rank and text-bias value generation','E,k,norm2,eps,scaling,group attrs,output dtype/index type','ID0 sentinel classification and hash0/vision-bias mode','native implementation/precision/tie policy','shape/tiling class until cross-shape exactness proved'],
        rowplan_guard='OriginalN/P+explicitpadding provenance required. P816alone cannot distinguish N800→816 from N816→816.',
        invalidation=['Publish template with stream/event and capture replay ordering before readers. Sameaddress does not mean samevalue generation.',
                      'Parameter reload/mode/native changes invalidate. Cached postmapped IDs additionally require mapepoch;current proposal retains mapping outside cache.',
                      'N816realinput overwrites former tail in persistentbank; nextN800mustrestore unless all-writer epoch proves preserved.'],
        build_options=[dict(kind='NATIVE_M1_BUILD',row_evaluations=1,condition='Needs row/shape/native equivalence; symbolic16J−1 rowwork,setup/fill/launch not priced.'),
                       dict(kind='NATIVE_SAME_P816_BUILD',row_evaluations=816,strict_positive_rowwork_break_even_uses=52,condition='16J>816⇒J>51 even before setup/fill. Same-shape conservative option.'),
                       dict(kind='SNAPSHOT_FIRST_PADDED_NATIVE_RESULT',extra_TopK_row_evaluations=0,reference_copy_bytes=96,condition='Read48+write48template; first call not accelerated,needs later eligible reuse and publication.')],
        repeated_layer_hits_proved=False,all_EP_receiver_work_preserved=True,
        conditional_operator_analysis_complete_for_this_question=True,whole_candidate_all_variants_closed=False,overall_complete=False))
    write(O/'padding-route-source-index-v77.json',dict(source_files=sources,new_remote_source_read=False,new_source_files_obtained=0,
        documented_sort_reference=dict(url='https://asc.gitcode.com/api/SIMD-API/adv_api/sort_operations/Sort.html',
            checked_date='2026-11-01',sections='Product support, no-SortConfig signature, parameters and constraints',
            supports='A3/no-SortConfig Sort accepts input score/index; equal-score order follows input index order in current documentation.',
            version_limit='Web API guide not pinned to installed CANN9.2.0 or binary; stable-tie semantics remain an explicit native cache gate. New SortConfig APIs are950-only and not used.',
            used_as_numeric_capacity_or_loaded_binary_proof=False),
        no_model_Torch_NPU_capture_service_or_process_work=True,overall_complete=False))
    print(dict(domain_pad_calls=len(pads),source_order_correspondences=len(pairs),native_pointer_bindings=0,
               original_bytes=original,integrated_bytes=integrated,separate_bytes=assembled,
               variants=5,actual_gain=None,status='BOUNDED_ANALYSIS_READY'))


if __name__=='__main__':main()
