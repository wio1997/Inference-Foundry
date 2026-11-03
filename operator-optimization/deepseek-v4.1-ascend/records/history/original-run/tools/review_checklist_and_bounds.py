"""Apply primary-controller review mappings, and calculate conditional ceilings."""
from pathlib import Path
from fractions import Fraction
import hashlib,json,datetime,shutil
T=Path(__file__).resolve().parents[1]
def r(p):return json.loads((T/p).read_text())
def w(p,d):
 f=T/p;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
templates={d['id']:d for d in r('graphs/source-templates.json')['templates']};records=r('records.json');old=records['fusion_candidates'];records['existing_fusions']=[]
maps={
'F01':('MHC_BLOCK_AND_CROSS_LAYER',['attn_pre']),
'F02':('MHC_BLOCK_AND_CROSS_LAYER',['attn_pre','input_norm']),
'F03':('MHC_BLOCK_AND_CROSS_LAYER',['ffn_post','next_pre']),
'F04':('MHC_BLOCK_AND_CROSS_LAYER',['norm_cast']),
'F05':('ATTN_C2_SOURCE',['q_norm','q_quant','q_b']),
'F06':('ATTN_C2_SOURCE',['q_a','kv']),
'F07':('ATTN_C2_SOURCE',['q_b','q_rope']),
'F08':('ATTN_C2_SOURCE',['kv','kv_norm','kv_rope','swa_store']),
'F09':('ATTN_C2_SOURCE',['iq','iq_rope','iq_quant']),
'F10':('ATTN_C2_SOURCE',['iw','iw_scale']),
'F11':('ATTN_C2_SOURCE',['ik_norm','ik_rope','ik_quant','ik_store','scale_cast','scale_store']),
'F12':('ATTN_C2_SOURCE',['qli','fixup','indices_store']),
'F13':('ATTN_C2_SOURCE',['attention']),
'F14':('ATTN_C2_SOURCE',['attention','inverse_rope','pad','restore_heads','o_a']),
'F15':('ATTN_C2_SOURCE',['pad','restore_heads','o_a']),
'F16':('ATTN_C2_SOURCE',['o_a','o_b','reduce_scatter']),
'F17':('ATTN_C2_SOURCE',['x_cast','compress_proj','score_proj']),
'F18':('ATTN_C2_SOURCE',['ring_pool','compress_norm']),
'F19':('ATTN_C2_SOURCE',['compress_norm','ik','ik_norm','ik_rope','cmp_rope','cmp_store']),
'F20':('MOE_WRAPPER',['router','topk','dispatch_ffn_combine']),
'F21':('MC2_INTERNAL_KNOWN_STAGES',['dispatch','gmm1','swiglu','gmm2','combine']),
'F22':('MOE_WRAPPER',['shared_gate_up','swiglu_quant','shared_down']),
'F23':('MOE_WRAPPER',['shared_down','dispatch_ffn_combine','merge']),
'F24':('ENGRAM',['hash','lookup','ready','rows']),
'F25':('ENGRAM',['proj','gate']),
'F26':('MHC_BLOCK_AND_CROSS_LAYER',['collapse','final_norm']),
'F27':('DSPARK_BOUNDARIES',['aux','main_proj','main_norm']),
'F28':('DSPARK_BOUNDARIES',['embedding','repeat','block0']),
'F29':('DSPARK_SHARED_KV',['proj','norm','rope','cat','slots','scatter']),
'F30':('DSPARK_BOUNDARIES',['collapse','norm','logits','markov','confidence','accept'])}
existing_only={'F01','F04','F13'};source_review=[]
for c in old:
 fid=c['id']
 if fid=='F31':
  c.update(status='UNKNOWN',semantic_equivalence_argument='Device metadata source shows existing asynchronous tasks and deferred waits; complete scheduler/metadata graph is not expanded',unknown_ids=['U_PRIMITIVE_MAPPING'],implementation_availability='PARTIAL_EXISTING_ASYNC_PATH');source_review.append({'id':fid,'status':'UNKNOWN','reason':'Explicit metadata graph and exact consumers remain to be expanded'});continue
 tid,selected=maps[fid];d=templates[tid];nodes={n['id']:n for n in d['nodes']};assert set(selected)<=set(nodes)
 ins=[e for e in d['edges'] if e['consumer'] in selected and e['producer'] not in selected];outs=[e for e in d['edges'] if e['producer'] in selected and e['consumer'] not in selected];inside=[e for e in d['edges'] if e['producer'] in selected and e['consumer'] in selected]
 graph={'id':fid+'_BOUNDARY_CANDIDATE','original_template_graph_id':tid,'original_node_mapping':{x:x for x in selected},'nodes':[nodes[x] for x in selected],'internal_edges':inside,'external_inputs':ins,'external_outputs':outs,'retained_persistent_consumers':'All future cache/index consumers in G1 must retain persistent outputs; template list alone does not enumerate all invocations','rewrite':'Candidate composite boundary preserving original semantic operations and value versions; no reassociation or removal proved','status':'ALREADY_IMPLEMENTED' if fid in existing_only else 'CONDITIONAL','precision_conditions':d['precision_rule'],'new_performance_result':False}
 path='candidates/'+fid+'.json';w(path,graph);c.update(status='OPPORTUNITY' if fid in existing_only else 'CONDITIONAL',original_node_ids=selected,module_ids=[tid],rewrite_rule_ids=['SOURCE_BOUNDARY_CONTRACTION_OR_REGROUP'],candidate_graph_path=path,candidate_graph_sha256=hashlib.sha256((T/path).read_bytes()).hexdigest(),original_node_mapping=graph['original_node_mapping'],other_consumers=outs,implementation_availability='ALREADY_IMPLEMENTED' if fid in existing_only else 'UNKNOWN',semantic_equivalence_argument='Preserving every operation, rounding and state write in original order preserves semantics as a composite boundary. Eliminating I/O, changing order or gaining speed is conditional and unproved.',new_liveness_and_resource_concerns=['Preserve ALL external and persistent consumers','Prove tile-capacity bound before claiming an intermediate can remain on-chip','Do not serialize independent Q/KV or query-quant overlap as an accidental consequence','Current API boundary is not a universal physical barrier'],scan_pass_id='SOURCE_REVIEW_31',source_paths=[d['source']],review_type='PRIMARY_CONTROLLER_SOURCE_REVIEW')
 if fid in existing_only:records['existing_fusions'].append({'check_id':fid,'source':d['source'],'reason':'Existing fused primitive; no new fusion gain claimed'})
 source_review.append({'id':fid,'status':graph['status'],'template_id':tid,'candidate_graph_path':path,'semantic_elimination_proved':False})
 c['potential_removed_work_or_io']={'upper_opportunity_expression':'For a materialized internal edge with no external users: up to one write and one read, 2*bytes(edge), conditional on compatible tiling and capacity','guaranteed_removed_bytes':None,'persistent_or_shared_outputs_cannot_be_deleted':True,'original_check_text':c['potential_removed_work_or_io']}
records['fusion_candidates']=old;w('records.json',records);w('evidence/source-review-31.json',{'items':source_review,'review_scope':'Boundary/consumer review; not machine-checked equivalence or exhaustive primitive scan','whole_model_scan_complete':False})

info=r('evidence/profile-device-capacity-info.json')['device_info'];assert len(info)==16
values=[x['device_info'][0] for x in info];assert all(int(x['ai_core_num'])==24 and Fraction(x['aic_frequency'])==1800 for x in values)
ops_per_cycle=16*32*16*2;capacity_ops=24*1800*1000000*ops_per_cycle;capacity_mac=Fraction(capacity_ops,2)
conditional={'status':'CONDITIONAL_CEILING','physical_unconditional_bound_proven':False,'capacity_mac_per_second':str(capacity_mac),'capacity_ops_per_second':str(capacity_ops),'per_logical_device':True,'observed_profiler_info_devices':16,'observed_frequency_MHz':'1800','observed_aicore_count':24,'official_nominal_ops_per_cycle':ops_per_cycle,'primary_source':{'url':'https://github.com/Ascend/msprof-analyze/blob/6b8f431d4724043f44a39eef000fd5c6600784d8/msprof_analyze/cluster_analyse/cluster_kernels_analysis/operator_mfu/chip_peak_flops.py','commit':'6b8f431d4724043f44a39eef000fd5c6600784d8','snapshot':'sources/primary/msprof-analyze/chip_peak_flops.py','sha256':hashlib.sha256((T/'sources/primary/msprof-analyze/chip_peak_flops.py').read_bytes()).hexdigest()},'conditions':['The official nominal dense INT8 ops/cycle model applies to the actual Ascend910_9382 resource set','Frequency is capped at or below1800MHz for the declared evaluation domain; observed1800MHz alone is not a max-frequency proof','MAC counts are necessary in standard dense algorithm domain; no omitted alternate algorithm/answer-cache escape','Sustained capacity cannot exceed the official nominal model; this premise remains explicit, not independently proven here']}
w('evidence/conditional-nominal-capacity.json',conditional)
def exact(x):return {'exact_fraction':str(x),'decimal_approx_not_certificate':float(x)}
comparisons=[];rank0=r('evidence/TRACE-SUMMARY.json')['ranks'][0]
for v in rank0['shape_variants']:
 if v['type']=='QuantBatchMatmulV3' and '1024,80,16,32' in v['shapes']:
  M=int(v['shapes'].strip('"').split(',')[0]);F=41943040*M;L=Fraction(F,capacity_mac);obs=Fraction(str(v['median_task_duration_us']))/1000000
  comparisons.append({'object_id':'Wq_b_rank0_M'+str(M),'object_kind':'INT8_Q_projection_family_target_draft_not_separated','case_id':'OBSERVED_SHAPE_M'+str(M),'implementation_domain':'Standard dense Wq_b, per logical device, fixed conditional nominal1800MHz capacity','scope_match_status':'MATCHED_KERNEL_BOUNDARY_WITH_EXPLICIT_CLOCK_CONDITION','effective_work_Q':M,'work_unit':'query_row','necessary_MAC':F,'measured_time_T':exact(obs),'time_lower_L':exact(L),'time_unit':'second','measured_throughput':exact(Fraction(M,obs)),'throughput_upper_U':exact(Fraction(M,L)),'measured_over_ceiling_ratio':exact(L/obs),'time_gap_ratio':exact(obs/L),'observed_median_us':v['median_task_duration_us'],'samples':v['calls'],'bound_status':'CONDITIONAL_CEILING','dominant_constraint_ids':['DENSE_INT8_COMPUTE_ONLY'],'utilization_metric_reference':{'mac_pipeline_ratio':v['aic_mac_ratio_duration_weighted_reported'],'mte2_pipeline_ratio':v['aic_mte2_ratio_duration_weighted_reported'],'not_theoretical_efficiency':True},'source_ids':['profile-device-capacity-info','chip_peak_flops','existing kernel_details'],'profile_overhead':'UNKNOWN; measurement observer-affected; ratio not a speedup promise','candidate_only':False})
Fframework=398862791475200;Lframework=Fraction(Fframework,16*capacity_mac);baseline=r('evidence/baseline-result.json');obs=Fraction(str(baseline['total_formal_duration_seconds']))/2
comparisons.append({'object_id':'TARGET_SELECTED_INT8_C1','object_kind':'framework_relaxation','implementation_domain':'Standard dense target evaluation, non-dropped top6 plus1shared, fixed conditional nominal1800MHz sixteen device resource set','effective_work_Q':8192,'work_unit':'returned output token per C8 round','necessary_MAC':Fframework,'time_lower_L':exact(Lframework),'throughput_upper_U':exact(Fraction(8192,Lframework)),'measured_time_T':exact(obs),'measured_throughput':baseline['aggregate_output_tokens_per_second'],'measured_over_ceiling_ratio':exact(Lframework/obs),'time_gap_ratio':exact(obs/Lframework),'bound_status':'CONDITIONAL_CEILING','relaxations':['All draft/rejected work, BF16/FP32 work, attention, memory, communication and scheduler costs omitted optimistically'],'not_an_achievable_speedup':True})
w('evidence/conditional-performance-comparisons.json',{'capacity':conditional,'comparisons':comparisons,'unconditional_numeric_ceiling':'UNKNOWN'})
module=r('module-analysis.json');module['performance_comparisons']=comparisons;module['status']='PARTIAL';module['observed_graph_id']='G1_REFINED_MODULE_GRAPH';module['unknown_ids']=['U_MODULE_ATTRIBUTION','U_HARDWARE_CAPACITY','U_PROFILE_BOUNDARY'];w('module-analysis.json',module)
state=r('STATE.json');state.update(current_phase='P2',fusion_scan_status='SCANNING',theoretical_bound_status='CONDITIONAL_CEILING',performance_comparison_status='PARTIAL_CONDITIONAL_NUMERIC_AND_SYMBOLIC',next_read_only_action='Finish semantic coverage review, resolve module/API/task correlation or prepare bounded targeted markers only if needed',last_source_review={'reviewed_prior_items':31,'structural_candidate_count':r('candidates/structural-index.json')['count'],'new_application_fusions':0,'unconditional_hardware_bound':'UNKNOWN'});w('STATE.json',state)
print(json.dumps({'reviewed_check_items':len(source_review),'conditional_comparisons':len(comparisons),'nominal_INT8_TMAC_per_second_per_logical_device':float(capacity_mac/10**12),'framework_conditional_upper_tokens_per_second':float(Fraction(8192,Lframework)),'unconditional_physical_bound_proven':False},indent=2))
