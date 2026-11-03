"""Exclusive functional attribution and complete reported-counter accounting."""
import collections
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

A=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis')
O=A/'joint172'
IDX=json.loads((O/'PROFILE-FRAME-SYMBOL-DELTA.json').read_text())['all_frame_index']
MATRIX=['M06','M07','M08','M09','M10','M11','M12','M13','M14','M16','M18','M20']
DEVICE_MODULES=['M'+str(i).zfill(2) for i in range(4,26)]
ENGINES={'AIC_MAC':('aic_total_cycles','aic_mac_ratio'),'AIV_Vector':('aiv_total_cycles','aiv_vec_ratio')}
RATIOS=['aic_mac_ratio','aic_scalar_ratio','aic_mte1_ratio','aic_mte2_ratio','aic_fixpipe_ratio','aiv_vec_ratio','aiv_scalar_ratio','aiv_mte1_ratio','aiv_mte2_ratio','aiv_mte3_ratio']

def union(intervals):
    last=None;total=0
    for a,b in sorted(intervals):
        if last is None:last=[a,b]
        elif a<=last[1]:last[1]=max(last[1],b)
        else:total+=last[1]-last[0];last=[a,b]
    return total+(last[1]-last[0] if last else 0)

def module_from_source(frames,regime):
    named=[IDX[f] for f in frames if f in IDX]
    syms=[x['enclosing_qualified_symbol'] for x in named]
    # Pure context preparation has its own boundary,not generic decoder children.
    if any(s and (s.startswith('DeepseekV41DSparkModel.') or s.startswith('DSparkDeepseekV41ForCausalLM.')) and any(q in s for q in ['combine_hidden_states','precompute_and_store_context_kv','_project_shared_kv','_store_standard_swa_kv']) for s in syms):
        return ['M18'],'EXACT_CONTEXT_OPERATOR_ANCESTOR'
    if any(s and s.startswith('DSparkMarkovHead.') for s in syms):return ['M20'],'EXACT_MARKOV_OPERATOR_ANCESTOR'
    for x in named:
        s=x['enclosing_qualified_symbol'] or '';p=x['path'];op=regime['opType']
        if '/engram/' in p or 'prepare_engram' in s:return ['M14'],'EXACT_ENGRAM_OPERATOR_FRAME'
        if 'shared_experts.py' in p or s.startswith('AscendSharedExperts.'):return ['M13'],'EXACT_SHARED_EXPERT_FRAME'
        if '/sample/' in p or '/sample/' in p.replace('/v1/','/') or 'reject_sample' in p:return ['M17'],'EXACT_SAMPLING_FRAME'
        if any(q in s for q in ['update_num_computed_tokens_for_batch_change','prepare_next_token_ids_padded']):return ['M24'],'EXACT_CROSS_ITERATION_STATE_FRAME'
        if s.startswith('AscendLogitsProcessor.') or s.startswith('LogitsProcessor.'):return ['M16'],'EXACT_VOCABULARY_HEAD_FRAME'
        if any(q in s for q in ['MetadataBuilder.','DeviceMetadataExecutor.','compute_slot_mapping','compressed_slot_mapping','get_cos_and_sin_dsa','build_dspark_swa_indices','build_draft_attn_metadata']):return ['M21'],'EXACT_METADATA_OPERATOR_FRAME'
        if s in ['_ascend_sp_padding_mask_impl','sp_padding_mask']:return ['M21'],'EXACT_PADDING_MASK_METADATA_FRAME'
        if s in ['_ascend_sp_shard_impl','sp_shard','sp_all_gather','sp_reduce_scatter','_ascend_sp_reduce_scatter_impl']:return ['M23'],'EXACT_PARTITION_OR_COLLECTIVE_HELPER_FRAME'
        if s.startswith('DeepseekV41Compressor.') or s=='compressor_from_projected':return ['M08'],'EXACT_COMPRESSOR_FRAME'
        if s.startswith('DeepseekV41Indexer.') or s in ['prepare_indexer_indices','quantize_indexer_query']:return ['M09'],'EXACT_INDEXER_FRAME'
        if any(s.endswith(q) for q in ['.hc_pre','.hc_post']):return ['M06'],'EXACT_HC_FRAME'
        if s.endswith('.hc_collapse'):return ['M15'],'EXACT_FINAL_HC_COLLAPSE_FRAME'
        if s.endswith('.rms_norm_cast'):return ['M12'],'EXACT_FFN_NORMALIZATION_FRAME'
        if s.startswith('AscendMoERunner.') or s.startswith('AscendRoutedExperts.') or '/ops/fused_moe/' in p:return ['M12'],'EXACT_ROUTED_MOE_OPERATOR_FRAME'
        if s in ['restore_tp_heads','AscendDSACPImpl._forward_o_proj','AscendDSACPImpl._apply_wo_b','AscendDSAV41CPImpl._project_output']:return ['M11'],'EXACT_ATTENTION_OUTPUT_FRAME'
        if s in ['AscendDSAV41Impl._forward_attention']:return ['M10'],'EXACT_ATTENTION_FRAME'
        if s in ['AscendDSAV41Impl._write_compressed_source']:return ['M08'],'EXACT_COMPRESSED_KV_SOURCE_FRAME'
        if s in ['AscendDSAV41Impl._write_swa_cache','AscendDSAV41Impl._project_kv']:return ['M07'],'EXACT_SWA_OR_KV_PROJECTION_FRAME'
        if s in ['AscendDSAV41Impl._select_sparse_indices','AscendDSAV41CPImpl._select_sparse_indices','AscendDSAV41Impl._quantize_indexer_query','AscendDSAV41Impl._prepare_indexer_inputs']:return ['M09'],'EXACT_ATTENTION_INDEXER_STAGE_FRAME'
        if s in ['AscendDSAV41Impl.multistream_preprocess','AscendDSAV41Impl._preprocess','DeepseekV41Attention.forward']:
            expr=' '.join(x.get('call_expressions_on_profile_line',[]))
            if 'compress' in expr:return ['M08'],'SOURCE_CALLSITE_COMPRESSOR_EXPRESSION'
            if any(q in expr for q in ['q_a','q_b','kv_a','q_norm','kv_norm','_project_kv']):return ['M07'],'SOURCE_CALLSITE_QKV_EXPRESSION'
            return ['M07','M08'],'QKV_COMPRESSOR_STAGE_NOT_EXCLUSIVELY_BOUND'
        if s=='DeepseekV41DecoderLayer.forward':
            expr=' '.join(x.get('call_expressions_on_profile_line',[]))
            if op=='RmsNorm' and 'attn_norm' in expr:return ['M07'],'SOURCE_ATTENTION_INPUT_NORM_EXPRESSION'
            return ['M06','M07','M12'],'DECODER_GENERIC_CALLSITE'
        if s in ['DeepseekV41DSparkModel.compute_logits','DSparkDeepseekV41ForCausalLM.compute_logits','DSparkDeepseekV41ForCausalLM.compute_draft_logits']:
            return (['M16'] if 'MatMul' in op or 'Matmul' in op else ['M15','M16','M20']),'DRAFT_HEAD_CALLSITE'
        if s.startswith('DeepseekV41DSparkModel.') or s=='DSparkDeepseekV41ForCausalLM.forward':return ['M19'],'DS_PARK_DECODER_EXCLUSIVE_NON_CHILD_FRAME'
        if s in ['DeepseekV41Model.embed_input_ids','AscendVocabParallelEmbedding.forward','AscendVocabParallelEmbedding._forward_standard','AscendVocabParallelEmbedding._mask_input_for_vocab_range']:return ['M05'],'EXACT_EMBEDDING_FRAME'
        if s=='DeepseekV41Model.forward':
            return (['M15'] if op=='RmsNorm' else ['M05','M15']),'TARGET_MODEL_GENERIC_CALLSITE'
        if s=='AscendSpecDecodeBaseProposer._run_merged_draft':
            if x['line'] in [1584,1587,1588,1589,1599,1602] and op in ['ArgMaxV2','Add','ViewCopy','Slice','Cast']:return ['M20'],'MARKOV_RECURRENCE_EXACT_CALLSITE'
            return ['M18','M19','M20'],'MERGED_DRAFT_GENERIC_CALLSITE'
        if s in ['AscendSpecDecodeBaseProposer.build_draft_attn_metadata','AscendSpecDecodeBaseProposer._pad_draft_buffers','AscendSpecDecodeBaseProposer.prepare_inputs_padded']:return ['M21'],'DRAFT_METADATA_OR_PADDING_FRAME'
        if s in ['AscendDSparkProposer.set_inputs_first_pass','AscendDflashProposer.build_model_inputs_first_pass']:return ['M18','M19','M21'],'DRAFT_FIRST_PASS_INPUT_PUBLICATION'
        if s in ['UpdatableGraph.capture_begin','NPUModelRunner.capture_model']:return ['M22'],'GRAPH_CONTROL_FRAME'
        if s in ['NPUModelRunner._prepare_inputs','GPUModelRunner._prepare_input_ids','GPUModelRunner._get_positions','GPUModelRunner._preprocess','NPUModelRunner._sanitize_placeholder_input_ids_for_forward','NPUModelRunner._dummy_run']:return ['M04'],'WORKER_INPUT_PREPARATION_FRAME'
        if s.startswith('NPUModelRunner.') or s.startswith('GPUModelRunner.'):return ['M04','M17','M21','M24','M25'],'WORKER_GENERIC_CALLSITE'
    return [],'NO_REVIEWED_SOURCE_ROLE'

def semantic_role(r):
    op=r['opType'];name=r['name'];resource=r['taskType']
    if resource=='COMMUNICATION' or op.startswith('hcom_') or name.startswith('hcom_'):return 'M23'
    return {'DispatchFFNCombine':'M12','HcPre':'M06','HcPost':'M06','RmsNormCast':'M12','MoeGatingTopKHash':'M12','DequantSwigluQuant':'M13','SparseFlashMla':'M10','VllmAscendQuantLightningIndexerV2':'M09','SparseFlashMlaMetadata':'M21','VllmAscendQuantLightningIndexerV2Metadata':'M21'}.get(op)

def stats():
    return {'executions':0,'kernel_sum_ns':0,'PMU_groups':0,'ratios':{n:{'numerator':0.0,'denominator':0.0,'valid_groups':0} for n in RATIOS},'missing_PMU_groups':0,'zero_cycle_or_missing_metric':collections.Counter(),'source_reasons':collections.Counter()}

def update(s,nr,metrics,reason):
    s['executions']+=1;s['kernel_sum_ns']+=nr[3]-nr[2];s['source_reasons'][reason]+=1
    if metrics:s['PMU_groups']+=1
    else:s['missing_PMU_groups']+=1
    for name in RATIOS:
        engine='aic' if name.startswith('aic_') else 'aiv';cycles=metrics.get(engine+'_total_cycles');value=metrics.get(name)
        if cycles is not None and cycles>0 and value is not None and 0<=value<=1:
            m=s['ratios'][name];m['numerator']+=cycles*value;m['denominator']+=cycles;m['valid_groups']+=1
        else:s['zero_cycle_or_missing_metric'][name]+=1

def finish(s):
    for name,r in s['ratios'].items():r['fraction']=r['numerator']/r['denominator'] if r['denominator'] else None
    s['zero_cycle_or_missing_metric']=dict(s['zero_cycle_or_missing_metric']);s['source_reasons']=dict(s['source_reasons']);return s

def membership_bounds(base_n,base_d,optional):
    if not optional:return {'lower':base_n/base_d if base_d else None,'upper':base_n/base_d if base_d else None}
    bounds=[]
    for reverse in [False,True]:
        n,d=base_n,base_d;best=n/d if d else None
        for w,r in sorted(optional,key=lambda x:x[1],reverse=reverse):
            n+=w*r;d+=w;q=n/d
            best=q if best is None else max(best,q) if reverse else min(best,q)
        bounds.append(best)
    return {'lower':bounds[0],'upper':bounds[1]}

def matrix_work(r):
    if r['opType'] not in ['MatMulV2','MatMulV3','BatchMatMulV2','QuantBatchMatmulV3']:return None
    try:
        a=[int(x) for x in r['inputShapes'].strip('"').split(';')[0].split(',')];o=[int(x) for x in r['outputShapes'].strip('"').split(';')[0].split(',')]
        if len(a)<2 or len(o)<2 or a[:-2]!=o[:-2] or a[-2]!=o[-2] or min(a+o)<=0:return None
        dtype=r['inputDataTypes'].split(';')[0];mode={'INT8':'INT8','DT_BF16':'BF16','BF16':'BF16','DT_FLOAT16':'FP16','FLOAT16':'FP16','FLOAT':'FLOAT_MODE_UNKNOWN'}.get(dtype)
        if not mode:return None
        return {'mode':mode,'MAC':math.prod(o)*a[-1],'K':a[-1],'M':a[-2],'N':o[-1]}
    except (TypeError,ValueError):return None

started=time.time();global_stats={m:stats() for m in DEVICE_MODULES};optional={m:{e:[] for e in ENGINES} for m in DEVICE_MODULES};profiles=[];by_reason=collections.Counter();ambiguity=collections.Counter();unresolved_examples=[];source_conflicts=[];native_count=0;role_eligible=collections.Counter();module_work=collections.defaultdict(collections.Counter);dummy_counts=collections.Counter();source_frames=set()
for p in sorted((O/'profiles').glob('*.json')):
    d=json.loads(p.read_text());frames=d['source_frame_strings'];chains=d['callchains'];cpu=d['CPU_rows'];metric_names=d['metric_names'];local={m:stats() for m in DEVICE_MODULES};intervals=collections.defaultdict(list);unbound_intervals=[];assignments=[]
    for nr in d['native_rows']:
        native_count+=1;r=d['regimes'][nr[9]];metrics=dict(zip(metric_names,nr[11])) if nr[11] is not None else {};fs=[]
        if len(nr[10])==1:
            c=cpu[str(nr[10][0])];fs=[frames[str(x)] for x in chains.get(str(c['callchain_id']),[])];source_frames.update(fs)
        sem=semantic_role(r);candidates,reason=module_from_source(fs,r) if fs else ([], 'NO_CPU_CREATOR_CONNECTION')
        if sem:
            if candidates and sem not in candidates:source_conflicts.append({'profile':d['profile'],'TASK_ROW':nr[0],'op':r['opType'],'semantic_role':sem,'source_candidates':candidates})
            candidates=[sem];reason='PINNED_OPERATOR_SEMANTIC_FAMILY' if sem!='M23' else 'EXPORTED_COMMUNICATION_ROLE'
        certain=len(candidates)==1
        if not candidates:
            candidates=MATRIX if matrix_work(r) else DEVICE_MODULES
            if len(unresolved_examples)<40:unresolved_examples.append({'profile':d['profile'],'TASK_ROW':nr[0],'regime':r,'source_frames':fs[:5]})
        by_reason[reason]+=1
        dummy=any('execute_dummy_batch' in f for f in fs)
        if dummy:dummy_counts[r['opType']]+=1
        if certain:
            owner=candidates[0];update(global_stats[owner],nr,metrics,reason);update(local[owner],nr,metrics,reason);intervals[owner].append((nr[2],nr[3]));work=matrix_work(r)
            if work:module_work[owner][work['mode']]+=work['MAC']
        else:
            owner=None;ambiguity[json.dumps(candidates)]+=1;unbound_intervals.append((nr[2],nr[3]))
            for e,(cyc,rat) in ENGINES.items():
                c=metrics.get(cyc);v=metrics.get(rat)
                if c is not None and c>0 and v is not None and 0<=v<=1:
                    for m in candidates:optional[m][e].append((c,v))
        assignments.append([nr[0],owner,candidates,reason,dummy])
    for m,s in local.items():finish(s);s['native_interval_union_ns']=union(intervals[m])
    ap=O/'assignments'/(d['profile']+'.json');ap.write_text(json.dumps({'profile':d['profile'],'assignments_columns':['TASK_ROW','certain_owner','candidate_owners','reason','runtime_dummy_context'],'rows':assignments},separators=(',',':'))+'\n')
    profiles.append({'profile':d['profile'],'modules':local,'unbound_native_interval_union_ns':union(unbound_intervals),'source_input':str(p),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'assignment_artifact':str(ap)})
rows=[]
for m,s in global_stats.items():
    finish(s);s['module']=m;s['shape_qualified_direct_MAC_by_mode']=dict(module_work[m]);s['primary_ambiguity_bounds']={}
    for e,(_,rat) in ENGINES.items():
        b=s['ratios'][rat];s['primary_ambiguity_bounds'][e]={**membership_bounds(b['numerator'],b['denominator'],optional[m][e]),'optional_positive_cycle_groups':len(optional[m][e]),'scope':'Allsubsetsofrecordedpositive-denominatorambiguousrowsinthisprofilepopulation;missingPMU/zerocycle/unrecordedpartsnotbounded. Coupledrolecountconstraintsmaytighten.'}
    rows.append(s)
ranking={}
for e,(_,rat) in ENGINES.items():
    eligible=[r for r in rows if r['ratios'][rat]['fraction'] is not None and (e!='AIC_MAC' or r['module'] in MATRIX)]
    ranking[e]=[{'module':r['module'],'fraction':r['ratios'][rat]['fraction'],'numerator':r['ratios'][rat]['numerator'],'denominator':r['ratios'][rat]['denominator'],'valid_groups':r['ratios'][rat]['valid_groups'],'source_certain_executions':r['executions'],'ownership_bounds':r['primary_ambiguity_bounds'][e]} for r in sorted(eligible,key=lambda r:r['ratios'][rat]['fraction'])]
review={'status':'CURRENT_COHORT_EXCLUSIVE_FUNCTIONAL_REPORTED_ACTIVITY_RANKING','profiles':profiles,'native_executions':native_count,'certain_execution_count':sum(r['executions'] for r in rows),'ambiguous_execution_count':sum(ambiguity.values()),'reason_counts':dict(by_reason),'ambiguous_sets':dict(ambiguity),'source_semantic_conflicts_count':len(source_conflicts),'source_semantic_conflict_samples':source_conflicts[:20],'source_semantic_conflict_groups':[{'op':k[0],'semantic_owner':k[1],'source_candidates':list(k[2]),'count':v} for k,v in collections.Counter((x['op'],x['semantic_role'],tuple(x['source_candidates'])) for x in source_conflicts).items()],'unresolved_examples':unresolved_examples,'modules':rows,'rankings':ranking,'runtime_dummy_context_op_counts':dict(dummy_counts),'used_source_frames':len(source_frames),'scope':'32v164CPU+PMUwindows only. Exclusivefunctionalleafpartition;M19parenthasexclusiveleftoverworkonly. Allrouted/HC/etcchildrenacrosstargetanddraftretaincanonicalfunctionalowners. No v168counts/timejoined. ReportedpipefractionsnotMFU/wholedevicebusy. Ambiguoussource andmissingPMUpopulationsretained.','elapsed_seconds':time.time()-started,'no_new_device_work':True,'whole_modules_closed':0,'overall_complete':False}
f=O/'MODULE-ACTIVITY-REVIEW.json';f.write_text(json.dumps(review,indent=2)+'\n');print(json.dumps({k:v for k,v in review.items() if k in ['status','native_executions','certain_execution_count','ambiguous_execution_count','source_semantic_conflicts_count','elapsed_seconds']}))
