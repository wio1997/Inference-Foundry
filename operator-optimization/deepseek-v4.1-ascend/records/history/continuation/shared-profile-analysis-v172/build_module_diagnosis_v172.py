"""Materialize grouped module diagnosis and every module's P5 comparison route."""
import json
from pathlib import Path

H=Path(__file__).resolve().parent
N=H.parent
O=N/'profile-first-20261031'
def read(p):return json.loads(p.read_text())
def write(n,d):(H/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

d=read(H/'module-window-diagnosis.json')
a=read(H/'module-activity-review.json')
old=read(H/'module-activity-review-first.json')
mods={x['module']:x for x in a['modules']}
ledger=read(O/'module-profile-ledger-v154.json')['modules']
work=d['conditional_matrix_work_comparisons']
rates={'INT8':8192,'BF16':4096,'FP16':4096}

# Symbolic terms refer to necessary work AFTER legal candidate restructuring.
# A cached weight's size or a recorded workspace is not necessary HBM traffic.
notes={
'M00':('HOST_SELECTION','Implementation selection changes downstream operator contracts; startup host control has no comparable NPU counter here.','L_host(selection contract); downstream work accounted to its functional owner.'),
'M01':('HOST_REQUEST','Frontend was ignored by this profiler; tokenization and request semantics cannot be assigned zero time.','L_host(request preparation), and release-time constraints on downstream consumers.'),
'M02':('HOST_STATE_AND_MEMORY','Prefix reuse, allocation and rollback change required cache work; no universal weight/cache-byte floor from allocation sizes.','max(Q_state/R_state_upper, B_necessary(initial cache state, ownership)/BW_upper).'),
'M03':('HOST_SCHEDULING','No exclusive NPU execution is owned by the scheduler in this partition; CPU/control latency remains outside pipe ranking.','L_host(scheduling) and proved producer/consumer release constraints; do not serially add overlappable device work.'),
'M04':('VECTOR_METADATA','Input preparation has positive Vector counters; ambiguous generic worker calls remain separate.','max(Q_index_or_conversion/R_mode_upper, B_necessary/BW_upper), both symbolically conditioned on eliminated intermediates and retained readers.'),
'M05':('VECTOR_LOOKUP','Only selected embedding rows are demanded; the full table is not automatically read from HBM. Communication is owned by M23.','max(B_necessary(selected rows, initial cache state, output contract)/BW_upper, Q_mask/R_vector_upper).'),
'M06':('MIXED_HC','HC is a mixed arithmetic operator; FLOAT precision, internal rounding and residual generations must be preserved. Shape alone does not inventory its MACs.','max(Q_HC_MAC/(24*r_mode*f_upper), Q_HC_vector/R_vector_upper, B_necessary/BW_upper).'),
'M07':('MIXED_PROJECTION','Reported ordinary GEMM shapes provide conditional partial work, not all norm/RoPE/quantization work. Outputs and numerical rounds constrain fusion.','max(L_direct_GEMM, Q_norm_RoPE_quant/R_vector_upper, B_necessary/BW_upper).'),
'M08':('MIXED_COMPRESSOR_STATE','FLOAT mode is unresolved; report HF32-off/on cases separately. Ring/latent state and selected consumer generations remain part of the semantic contract.','max(L_GEMM_mode_case, Q_state_and_vector/R_vector_upper, B_necessary/BW_upper).'),
'M09':('MIXED_INDEXER','INT8 and BF16 GEMM partial work is conditional; ranking/top-k/index precision and sparse row demands remain separate necessary terms.','max(L_GEMM, Q_score_selection/R_selection_upper, B_necessary/BW_upper).'),
'M10':('MIXED_SPARSE_ATTENTION','SparseFlashMla is fused already. Necessary dot-product work depends on actual sparse edges, head dimensions and the allowed arithmetic algorithm.','max(Q_MAC(selected sparse edges)/(24*r_mode*f_upper), Q_softmax/R_vector_upper, B_necessary(cache residency)/BW_upper).'),
'M11':('MIXED_PROJECTION_REDUCTION','Projection GEMMs have conditional shape work; head restoration/redistribution and numeric accumulation cannot be made free by fusion.','max(L_GEMM, Q_restore/R_vector_upper, B_necessary/BW_upper, L_required_network_cut).'),
'M12':('MIXED_ROUTED_MOE','F21 already fuses dispatch, quantization, two GEMMs, SwiGLU and return. Receiver assignments, not local sender rows, set expert work. v168 receiver counts are not paired with v164 timings.','Q_F21=35389440*sum(receiver expert assignments) in the reviewed direct INT8 source domain; L=max(Q_F21/(24*8192*f_upper), Q_vector/R_vector_upper, B_necessary/BW_upper, L_required_network_cut).'),
'M13':('MIXED_SHARED_EXPERT','Shared experts remain a distinct functional owner even under routed-MoE wrappers; scale/add readers and rounding constrain branch merging.','max(L_shared_expert_GEMM, Q_gate_scale/R_vector_upper, B_necessary/BW_upper).'),
'M14':('MIXED_ENGRAM_HOST_UVA','Host/UVA lookup is not a Cube operation. Reusing the pre-rotation FP32 conversion extends lifetime; whole embedding-table bytes are not necessary traffic.','max(L_projection_GEMM, B_necessary(selected host/UVA rows)/BW_link_upper, Q_hash_gate/R_vector_upper, B_necessary_device/BW_device_upper).'),
'M15':('VECTOR_COLLAPSE_NORM','Final HC collapse and norm have Vector counters; auxiliary/MTP readers and residual generations decide which materializations can disappear.','max(Q_collapse_norm/R_vector_upper, B_necessary/BW_upper).'),
'M16':('MIXED_VOCABULARY_HEAD','Conditional BF16 vocabulary projection work is available. ID-only reduction requires tie rules, score/logprob readers and cross-rank ownership to remain valid.','max(L_vocab_GEMM, Q_required_selection/R_vector_upper, L_required_vocab_network_cut, B_necessary/BW_upper).'),
'M17':('VECTOR_SAMPLING','Positive greedy/rejection callsites prove execution, not every sampler flag. Sampling RNG, tie and logprob semantics define legal candidates.','max(Q_required_comparison_or_sampling/R_mode_upper, B_necessary/BW_upper, L_required_cross_rank_selection).'),
'M18':('MIXED_CONTEXT_AND_KV_STATE','Context preparation has exact pure-context ancestors, separate from generic decoder children; KV publication and generation are observable effects.','max(L_context_GEMM, Q_KV_transform/R_vector_upper, B_necessary/BW_upper, L_required_network_cut).'),
'M19':('HIERARCHICAL_DECODER','M19 is an inclusive parent for routed/HC/attention children; this ranking contains only its exclusive leftovers. Parent and child totals must not be added.','Parent lower bound uses nonduplicate necessary loads per proved shared pool and max across independent resources; exclusive leftovers retain Q_state/R_vector_upper and B_necessary/BW_upper.'),
'M20':('MIXED_MARKOV_RECURRENCE','Only reviewed exact recurrence callsites and DSparkMarkovHead descendants are certain. Selected IDs, not the whole embedding table, determine lookup demand.','max(L_Markov_GEMM, Q_recurrence_selection/R_vector_upper, B_necessary(selected IDs and retained logits)/BW_upper).'),
'M21':('VECTOR_METADATA','Slot/index/mask work has positive counters. The public BOOL padding boundary can be analyzed without decoding the private native MemSet packet.','max(Q_required_index_mask/R_vector_upper, B_necessary/BW_upper), with signed quotient, padding and alias contracts explicit.'),
'M22':('GRAPH_LIFECYCLE_CONTROL','No exclusive COMPUTE row is attributed to graph lifecycle here. Model/event/wait/copy tasks are not zero work and are outside this counter table.','L_control(required graph update/readiness), plus nonduplicate necessary copy work; selected-tile barriers are not universal whole-operator barriers.'),
'M23':('COMMUNICATION_AND_PARTITION','Very low Vector arithmetic activity is expected for data transport/polling and does not mean no useful communication. MTE2 activity is not HBM bandwidth.','For single-owner uncompressed raw AllGather across k cards, L_cut>=max_b((G-C_b)/B_b_upper); current TP8/four-card full case gives 6S/card,24S total conditional ingress. Initial replication/compression/consumer fusion changes this contract.'),
'M24':('VECTOR_CROSS_ITERATION_STATE','Accepted/draft IDs, counters, prefetch and rollback remain state effects. Missing trace operands do not establish independence or removable stores.','max(Q_state_index/R_vector_upper, B_necessary_generation_and_lifetime/BW_upper), with proved cross-iteration release dependencies.'),
'M25':('HOST_RESPONSE','Frontend/output work was not measured consistently; no exclusive COMPUTE attribution is not zero response cost.','L_host(serialization and termination), plus necessary result-copy/readiness terms when established.'),
}
rows=[]
for m in ledger:
    mid=m['id'];resource,note,formula=notes[mid];s=mods.get(mid)
    comparisons=[]
    for x in work:
        if x['module']!=mid:continue
        c={k:v for k,v in x.items() if k!='regimes'}
        if x['mode']=='FLOAT_MODE_UNKNOWN':
            q=x['conditional_MAC']
            c['FLOAT_restricted_mode_cases']=[{'mode':mode,'MAC_per_core_cycle':r,'conditional_device_cycle_floor':q/(24*r),'rated1800MHz_reference_floor_ns':q/(24*r*1.8),'physical_time_floor_expression':f'{q}/(24*{r}*f_upper)'} for mode,r in [('FLOAT_HF32_OFF',1024),('FLOAT_HF32_ON',2048)]]
        else:c['physical_time_floor_expression']=f"{x['conditional_MAC']}/(24*{rates[x['mode']]}*f_upper)"
        comparisons.append(c)
    primary={}
    for engine,rat in [('AIC_MAC','aic_mac_ratio'),('AIV_Vector','aiv_vec_ratio')]:
        primary[engine]=s['ratios'][rat] if s else {'numerator':None,'denominator':None,'valid_groups':0,'fraction':None}
    rows.append({'module':mid,'name':m['name'],'candidate_links':m['candidate_links'],'resource_scope':resource,'certain_native_executions':s['executions'] if s else 0,'attribution_scope':'EXCLUSIVE_FUNCTIONAL_LEAF_OR_PARENT_LEFTOVER','primary_reported_activity':primary,'primary_status':'MEASURED_POSITIVE_DENOMINATOR' if any(v['valid_groups'] for v in primary.values()) else 'NO_COMPARABLE_EXCLUSIVE_COUNTER_OBSERVATION_NOT_ZERO_COST','P5_status':'CONDITIONAL_PARTIAL_CYCLE_AND_REFERENCE_COMPARISON' if comparisons else 'SYMBOLIC_OR_HOST_SCOPE_COMPARISON','P5_time_lower_bound':formula,'P5_required_work_and_capacity_missing':'Necessary post-restructuring work/traffic and matching physical rate upper bounds remain symbolic where not bound; f_upper unknown;1800MHz reference is not a physical upper bound.','P5_numeric_partial_comparisons':comparisons,'diagnostic_scope_note':note,'whole_module_fusion_scan_closed_by_this_metric_review':False})
write('all26-module-metric-and-P5-routes.json',{'status':'ALL26_ROUTED_WITH_MEASURED_CONDITIONAL_SYMBOLIC_OR_HOST_SCOPE','modules':rows,'metric_unit':'sum(reported engine total cycles * reported pipe fraction) / sum(reported engine total cycles), dimensionless; not physical MFU or full-device busy','coverage':{'module_universe':26,'positive_AIC_MAC_modules':len(a['rankings']['AIC_MAC']),'positive_AIV_Vector_modules':len(a['rankings']['AIV_Vector']),'certain_native_executions':a['certain_execution_count'],'ambiguous_native_executions':a['ambiguous_execution_count'],'all_native_executions':a['native_executions'],'missing_PMU_executions':a['native_executions']-read(H/'extraction-observation.json')['review']['PMU_execution_groups']},'P5_model_rules':['Conditional ordinary-shape work is not an unconditional MAC lower bound across all algorithms or transpose attributes.','Partial GEMM comparison uses exactly its selected execution times, not whole-module or request time.','A source tensor/weight/workspace size is not a universal necessary HBM byte count.','Independent resource lower bounds combine by max; mode/shared-pool accumulation requires proof.','No comparison ratio is a reachable speedup.'],'overall_complete':False})

causes=[]
for stage,g in d['groups'].items():
    allm=g['modules']['M12'];f=next(r for r in d['selected_module_operator_diagnostics'] if r['window_cluster']==stage and r['module']=='M12' and r['opType']=='DispatchFFNCombine')
    fs=f['stats'];causes.append({'window_cluster':stage,'measured_lowest_Cube_module':'M12','minimum':g['rankings']['AIC_MAC'][0],'recorded_ambiguity_robust':g['rankings']['AIC_MAC_minimum_stable_under_recorded_ownership_ambiguity'],'F21_executions':fs['executions'],'F21_PMU_groups':fs['PMU_groups'],'F21_missing_PMU_groups':fs['missing_PMU_groups'],'F21_kernel_sum_ns':fs['kernel_sum_ns'],'M12_kernel_sum_ns':allm['kernel_sum_ns'],'F21_fraction_of_M12_cumulative_native_time':fs['kernel_sum_ns']/allm['kernel_sum_ns'],'F21_fraction_of_M12_reported_AIC_cycles':fs['ratios']['aic_mac_ratio']['denominator']/allm['ratios']['aic_mac_ratio']['denominator'],'F21_pipe_ratios':fs['ratios'],'F21_duration_ns':f['duration_ns']})
claims=[
{'claim':'M12 is lowest among measured MAC-bearing functional modules in each existing window and pooled population under the declared reported-counter metric.','state':'SUPPORTED','support':'Per-execution exact PMU containment; sum numerators/denominators; recorded ambiguous-membership bounds do not change either window minimum.','limit':'Recorded positive-cycle population only; missing PMU, unrecorded work, physical MFU and production profiler-free ranking not covered.'},
{'claim':'The lowest M12 ratio and most of its observed cumulative native time are located inside already-fused F21, rather than just the surrounding small kernels.','state':'SUPPORTED','support':causes,'limit':'Cumulative cross-device native durations are not request walltime or removable time.'},
{'claim':'Very small receiver tiles and rank/expert imbalance reduce useful dense-MAC opportunity relative to the full pool.','state':'HYPOTHESIS_WITH_SOURCE_AND_OTHER_COHORT_MECHANISM_SUPPORT','support':'v169/v170 receiver-domain and source work model; current window has small local sender shapes plus broad F21 duration distribution.','limit':'Local sender row counts do not determine incoming work. v168 receiver work is not matched to these v164 PMU calls; no causal fraction assigned.'},
{'claim':'Cross-rank arrival/protocol synchronization and polling contribute to F21 idle arithmetic pipelines.','state':'HYPOTHESIS','support':'Reviewed F21 includes dispatch and return; current full-kernel intervals have low reported arithmetic and MTE activity simultaneously.','limit':'Pipe ratios alone do not isolate wait reasons or time fractions; host-delayed peer arrival, resource contention and instrumentation remain competing causes.'},
{'claim':'HBM saturation alone explains the full F21 interval.','state':'NOT_SUPPORTED_BY_THIS_METRIC','support':'Whole-interval reported AIC/AIV MTE2 fractions are also low; no necessary-byte or bandwidth counter supports a quantitative saturation claim.','limit':'Short bandwidth-limited phases, cache misses, transfer imbalance and protocol stalls remain possible; MTE2 fraction is not physical bandwidth.'},
{'claim':'The collective module M23 is computationally useless because its Vector arithmetic fraction is near zero.','state':'REFUTED','support':'Exported communication roles and substantial scalar/MTE2 activity describe transport/polling, which does not require high Vector arithmetic occupancy.','limit':'This does not establish communication efficiency or link saturation.'},
{'claim':'One can obtain speedup by fusing dispatch/GMM1/SwiGLU/GMM2/combine as a new F21 opportunity.','state':'REFUTED_FOR_THE_PINNED_CURRENT_PATH','support':'That semantic group is already fused in the current DispatchFFNCombine path; reconsider cross-boundary scheduling/ownership or legal decomposition instead of double-counting an existing fusion.','limit':'No implementation, attainable schedule or speedup is claimed.'},
{'claim':'CPU/framework gaps cannot affect low arithmetic activity within a resident communication-compute kernel.','state':'REFUTED_AS_A_GENERAL_EXCLUSION','support':'Host delay can postpone peer arrivals while resident device code polls or waits.','limit':'This cohort does not attribute a measured amount to that cause; CPU+PMU recorder perturbation remains explicit.'},
]
write('lowest-module-cause-review.json',{'status':'GROUPED_MEASURED_MINIMA_AND_COMPETING_CAUSES_REVIEWED','measured_minima':causes,'claims':claims,'no_trustworthy_global_minimum_across_host_Cube_Vector_communication':True,'no_speedup_or_physical_MFU_claim':True,'full_scope_fusion_scan_closed':False,'overall_complete':False})

checks={
'all32_inputs_accounted':len(a['profiles'])==32 and len(d['inputs'])==32,
'all288846_native_executions_retained':a['native_executions']==288846==a['certain_execution_count']+a['ambiguous_execution_count'],
'two_windows_16_profiles_each':len(d['groups'])==2 and all(len(g['profiles'])==16 for g in d['groups'].values()),
'12_MAC_20_Vector_measured_groups':len(a['rankings']['AIC_MAC'])==12 and len(a['rankings']['AIV_Vector'])==20,
'M12_lowest_both_windows_robust_recorded_ambiguity':all(g['rankings']['AIC_MAC'][0]['module']=='M12' and g['rankings']['AIC_MAC_minimum_stable_under_recorded_ownership_ambiguity'] for g in d['groups'].values()),
'all26_modules_P5_routed':len(rows)==26 and all(r['P5_time_lower_bound'] for r in rows),
'classification_tightening_preserved_ratios':all(x['ratios']==next(y for y in old['modules'] if y['module']==x['module'])['ratios'] for x in a['modules']),
'all_semantic_source_overrides_are_explicit_communication_partition':all(x['semantic_owner']=='M23' and x['op'].startswith('hcom_') for x in a['source_semantic_conflict_groups']),
'no_duplicate_COMPUTE_metadata_IDs':all(not x['COMPUTE_metadata_duplicate_globalTaskIds'] for x in read(H/'block-dimension-decode.json')['profiles']),
'launch_dimensions_unknown_not_raw_ID_core_counts':all(set(x['maps']['blockDim'].values())=={'N/A'} for x in read(H/'block-dimension-decode.json')['profiles']),
'no_crosscohort_work_time_join':all('v168' not in str(x) for x in work),
}
assert all(checks.values()),checks
write('module-diagnosis-independent-checks.json',{'status':'PASS','checks':checks,'formal_proof':False,'metric_scope':'Recorded v164 positive-cycle population; two formal CPU+PMU windows, not startup model-load profiling. Export timestamp clusters are not proof of one uniform semantic phase.','protocol_interpretation':'MODULE_DIAGNOSIS permits grouped comparable activity minima and explicitly stating no credible global minimum. Physical MFU for every module is not an additional prerequisite. Missing and ambiguous work remains explicit; fusion coverage is a separate obligation.','errors_preserved':['Submission construction formatting error before remote execution.','First diagnosis exposed blockDim STRING_IDS keys rather than decoded labels; no capacity/core-count inference used them. All labels N/A now decoded; corrected diagnostics merge groups, primary metrics unchanged.','One correction script had a pre-execution SyntaxError; retained receipt; no new model or device workload.'],'overall_complete':False})
print(json.dumps({'status':'PASS','checks':len(checks),'modules':len(rows),'minimum':'M12','all_module_global_minimum':None,'overall_complete':False}))
