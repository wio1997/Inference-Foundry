"""Profile-bound IndexQ/IndexWeight projections, upstream input panels and precise joins."""
from pathlib import Path
from collections import Counter,defaultdict
from fractions import Fraction as F
from itertools import product
import hashlib,json,math,statistics,struct
from finalize_origin_coverage_v9 import N,O,W,read,write

CACHE=N/'original-profile-cache-v53'
def pin(p):
    p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def dims(s):return [None if x=='N/A'else list(map(int,x.split(',')))if x else []for x in s.strip('"').split(';')]
def f16(x):return struct.unpack('>e',struct.pack('>e',float(x)))[0]
def f32(x):return struct.unpack('>f',struct.pack('>f',float(x)))[0]
def stat(x):return dict(n=len(x),minimum=min(x),median=statistics.median(x),maximum=max(x))
def section(p,a,b):return dict(**pin(p),first_line=a,last_line=b,source=Path(p).read_text().splitlines()[a-1:b])

def main():
    assert not(O/'REVIEW-CHECKPOINT-v106.json').exists()
    root=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend'
    model=root/'models/deepseek_v41/indexer.py';dsa=root/'attention/dsa_v41.py'
    vec=N/'attention-operator-source-v60/vllm-workspace/vllm-ascend/csrc/attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_service_vector_arch22.h'
    parts=[section(model,51,74),section(model,167,179),section(model,253,260),section(dsa,350,408),section(vec,714,759)]
    write(O/'index-projection-source-contract-v106.json',dict(parts=parts,facts=[
        'IndexQ is quant-config ReplicatedLinear QR1280→32*128=4096;IndexWeight is unquantized ReplicatedLinear hidden5120→32. Equal row count does not make their inputs or arithmetic the same.',
        'Ordinary source preprocess passes the same query_hidden expression to QA and later IndexWeight. QA normalizes1280 outputs;the32 IndexWeight outputs are a separate unnormalized branch. Packed/alternate/CP row mappings and mutation/epoch identity remain separate gates.',
        'Current _prepare_indexer_inputs calls IndexQ project,then records weights_start;aux waits that event before weight projection,while main applies IndexQ RoPE and then waits weights_done. Weight input is logically independent of the projected query once both source inputs are valid.',
        'Starting Weight before IndexQ can remove that control edge in a new schedule;deferring its join until the first true Weight reader can release independent query/compression work. Preserve input-ready events,all other readers/aliases,exception domains and bank lifetimes.',
        'Weight work and IndexQuery quantization are submitted to the same overlap-stream helper;the saved640 frontiers also place both on the same stream. A late main-stream join alone does not make these two auxiliary stages concurrent.',
        'Weight source contract is BF16 projection round→FLOAT widen*2^-6→FP16 conversion at select_projected. Native QLI then multiplies FP16 weight*FP16 query scale and broadcasts for Cube head reduction. A fused implementation may retain rounds in registers;dropping them needs an additional value-domain proof.',
        'v64 already priced Cast/Mul/Cast interfaces;v82 IndexQ RoPE/quant extension andv83 distinct-input join remain prior answers. This turn adds shape-specific projection/combined resource cuts,precise early-start/late-join source edges,and an upstream QA/Weight common-input panel.'],
        new_target_source_files=0,new_public_documents=0,no_framework_or_remote_model_device_capture_work=True,overall_complete=False))
    graph_path=CACHE/'analysis/native-execution-identity-v14/graph-blocks-PMU-and-MC2-v14.json'
    gs=defaultdict(list)
    for g in read(graph_path)['graph_executions']:gs[g['database_index']].append(g)
    old_front_path=O/'attention-indexer-profile-and-edges-v60.json'
    qquant={(r['database_index'],r['graph_ordinal'],r['block_ordinal']):r['query_quant']for r in read(old_front_path)['graph_QLI_chains']}
    paths=[];front=[];cohorts=defaultdict(list);tasks_unique=set()
    for db in range(32):
        p=CACHE/f'analysis/native-execution-identity-v14/db-{db:02d}.json';paths.append(p);data=read(p)
        ds={d['globalTaskId']:d for d in data['COMPUTE_definitions']};ts={t['task_record_rowid']:t for t in data['TASK_records']}
        def de(t):return ds[t['globalTaskId']]
        def ins(t):return dims(de(t)['inputShapes_text'])
        def outs(t):return dims(de(t)['outputShapes_text'])
        def qmatch(t):
            d=de(t)
            if d['opType_text']!='QuantBatchMatmulV3':return False
            a=ins(t);z=outs(t)
            return d['opType_text']=='QuantBatchMatmulV3'and z and len(z[0])==2 and z[0][1]==4096 and a[0]==[z[0][0],1280]and a[1]==[128,80,16,32]
        def wmatch(t):
            d=de(t)
            if d['opType_text']!='MatMulV2':return False
            a=ins(t);z=outs(t)
            return d['opType_text']=='MatMulV2'and len(a)==2 and a[1]==[32,5120]and z==[[a[0][0],32]]
        for t in ts.values():
            if t['globalTaskId']not in ds:continue
            role='IndexQ'if qmatch(t)else'IndexWeight'if wmatch(t)else None
            if role:
                d=de(t);m=outs(t)[0][0]
                cohorts[(role,m,d['inputDataTypes_text'],d['outputDataTypes_text'])].append([db,t['task_record_rowid'],t['endNs']-t['startNs']])
        def stage(t):return dict(task_record_rowid=t['task_record_rowid'],operator=de(t)['opType_text'],streamId=t['streamId'],startNs=t['startNs'],endNs=t['endNs'],duration_ns=t['endNs']-t['startNs'],signature={k:de(t)[k]for k in ['inputShapes_text','inputDataTypes_text','outputShapes_text','outputDataTypes_text']})
        for g in gs[db]:
            for b in g['blocks']:
                tt=[ts[i]for i in b['task_record_rowids']];q=[t for t in tt if qmatch(t)]
                if not q:continue
                assert len(q)==1;qi=q[0];w=[t for t in tt if wmatch(t)];assert len(w)==1;wp=w[0]
                qa=[t for t in tt if de(t)['opType_text']=='MatMulV2'and ins(t)==[[3,5120],[1280,5120]]];assert len(qa)==1
                assert de(qa[0])['inputDataTypes_text']==de(wp)['inputDataTypes_text']=='DT_BF16;DT_BF16'
                assert de(qa[0])['outputDataTypes_text']==de(wp)['outputDataTypes_text']=='DT_BF16'
                assert de(qi)['inputDataTypes_text']=='INT8;INT8;DT_BF16;FLOAT'and de(qi)['outputDataTypes_text']=='DT_BF16'
                out32=lambda t:outs(t)==[[3,32]]
                wc=[t for t in tt if de(t)['opType_text']=='Cast'and out32(t)and de(t)['outputDataTypes_text']=='FLOAT'];assert len(wc)==1
                mul=[t for t in tt if de(t)['opType_text']=='Mul'and out32(t)and de(t)['outputDataTypes_text']=='FLOAT'];assert len(mul)==1
                half=[t for t in tt if de(t)['opType_text']=='Cast'and out32(t)and de(t)['outputDataTypes_text']=='FLOAT16'];assert len(half)==1
                rope=[t for t in tt if de(t)['opType_text']=='InplacePartialRotaryMul'and ins(t)[0]==[3,1,32,128]];assert len(rope)==1
                qli=[t for t in tt if de(t)['opType_text']=='VllmAscendQuantLightningIndexerV2'];assert len(qli)==1
                qq=ts[qquant[(db,g['graph_ordinal'],b['block_ordinal'])]['task_record_rowid']]
                assert qq['streamId']==wp['streamId']and mul[0]['endNs']<=qq['startNs']
                stages={k:stage(t)for k,t in dict(QA=qa[0],IndexQ=qi,IndexWeight=wp,WeightFLOAT=wc[0],WeightScale=mul[0],WeightFP16=half[0],IndexRoPE=rope[0],IndexQuant=qq,QLI=qli[0]).items()}
                assert qi['endNs']<=wp['startNs']and wp['endNs']<=wc[0]['startNs']<=mul[0]['startNs']and mul[0]['endNs']<=half[0]['startNs']<=qli[0]['startNs']
                for t in stages.values():tasks_unique.add((db,t['task_record_rowid']))
                front.append(dict(database_index=db,graph_ordinal=g['graph_ordinal'],block_ordinal=b['block_ordinal'],root_bound=bool(g['prepare_root_keys']),stages=stages,
                    IndexWeight_overlaps_IndexRoPE=wp['startNs']<rope[0]['endNs']and rope[0]['startNs']<wp['endNs'],
                    causal_input_or_alias_identity_proved=False))
    assert len(front)==640
    cohorts=[dict(role=k[0],M=k[1],input_types=k[2],output_types=k[3],execution_count=len(v),TASK_keys=[[x[0],x[1]]for x in v],native_TASK_ns=stat([x[2]for x in v]))for k,v in sorted(cohorts.items())]
    for r in cohorts:
        f=2*r['M']*(1280*4096 if r['role']=='IndexQ'else 5120*32)
        r['full_output_dense_reference_OPS']=f
        r['dense_reference_OPS_per_native_TASK_second']=f/(r['native_TASK_ns']['median']*1e-9)
        r['rate_is_not_physical_MFU_or_request_critical_path']=True
    rows=sorted({r['M']for r in cohorts});assert rows==[3,800,816,830,834]
    write(O/'index-projection-native-frontiers-v106.json',dict(frontiers=front,frontier_count=640,root_bound=512,phase_unbound=128,
        reused_unique_TASKs=len(tasks_unique),projection_cohorts=cohorts,observed_M=rows,
        query_before_weight_frontiers=640,weight_rope_overlap_frontiers=sum(r['IndexWeight_overlaps_IndexRoPE']for r in front),
        weight_and_IndexQuant_same_stream_frontiers=640,
        definition_and_shape_scope_not_new_functional_attribution=True,new_native_promotions=0,
        limits='Observed task order/shape/dtype only;native scores/aliases/input row epochs/counterfactual readiness/internal tiles and current cache residency are not attested. Direct cohorts are shape-bound,not promoted module/phase assignments.',overall_complete=False))
    models=[];warm=24*65536;rate=24*128
    for m in rows:
        fq=2*m*1280*4096;fw=2*m*5120*32;wq=1280*4096;ww=2*5120*32
        cut=math.ceil(max(0,wq+ww-warm)/rate)
        models.append(dict(M=m,IndexQ_classical_INT8_ops=fq,IndexWeight_classical_BF16_ops=fw,
            IndexQ_INT8_weight_value_bytes=wq,IndexWeight_BF16_weight_value_bytes=ww,
            IndexQ_scale_values_bytes=2*4096+4*m,
            IndexQ_input_output_value_bytes=m*1280+2*m*4096,
            IndexWeight_input_output_value_bytes=2*m*5120+2*m*32,
            IndexQ_useful_compute_cycle_cut=str(F(fq,24*16384)),IndexWeight_useful_compute_cycle_cut=str(F(fw,24*8192)),
            one_global_initial_L0B_grant=warm,shared_fixedB_rate_upper_bytes_per_cycle=rate,combined_fixedB_weight_ingress_cycle_cut=cut,
            combined_cycle_cut=math.ceil(max(F(fq,24*16384),F(fw,24*8192),cut)),
            IndexQ_full_M16_reference_issued_ops=2*math.ceil(m/16)*16*1280*4096,
            IndexWeight_full_M16_reference_issued_ops=2*math.ceil(m/16)*16*5120*32,
            CastMulCast_interface_budget_from_v64_bytes=16*m*32,
            projection_epilogue_interface_budget_from_v64_bytes=20*m*32,
            cycle_to_time='T>=cycle_cut/f_max only with a proven physical max frequency and the declared path/scope;no clock number assumed.',
            domain='Declared full ordinary named products and distinct uncompressed weights on fixedB L1→L0B path,24AIC,one initial L0B grant,no cached computed outputs/pruning/transposed operand route. INT8 andBF16 separate compute cuts use max;no undocumented mixed-issue simplex. Logical values not HBM.'))
    panels=[]
    for old in read(O/'projection-panel-work-and-resource-models-v99.json')['models']:
        ops=old['useful_classical_ops']+2*3*5120*32;weights=old['distinct_weight_value_bytes']+2*5120*32
        naive=old['global_rectangular_classical_ops']+2*24*5120*32
        panels.append(dict(C1_source_branch=old['C1_source_branch'],query_rows=3,global_KV_rows=24,
            groups=old['groups']+[dict(id='IndexWeight',M=3,N=32)],new_useful_classical_BF16_ops=ops,
            global_rectangular_reference_ops=naive,rectangular_to_ragged_work_ratio=str(F(naive,ops)),
            distinct_weight_bytes=weights,one_initial_L0B_grant=warm,shared_fixedB_rate_upper_bytes_per_cycle=rate,
            fixedB_weight_ingress_cycle_cut=math.ceil((weights-warm)/rate),BF16_compute_cycle_cut=str(F(ops,24*8192)),
            incremental_input_read_budget_vs_same_v99_independent_baseline_bytes=2*3*5120,
            total_input_read_budget_vs_declared_all_independent_baseline_bytes=old['conditional_input_read_difference_bytes']+2*3*5120,
            IndexWeight_not_in_QA_RMS=True,
            domain='Same source query_hidden feeds QA andIndexWeight;optional global KV uses sameJ map/generation gates asv99. KeepQA1280 RMS separate fromWeight32 postscale,no joint1312 RMS,independent rounding/row-ready outputs. No existing interface credit counted twice.'))
    write(O/'index-projection-work-and-resource-models-v106.json',dict(models=models,upstream_panel_models=panels,
        fixed_path_capacity_evidence=['typed-capacity-primary-contract-v89.json','projection-panel-work-and-resource-models-v99.json'],
        status='CONDITIONAL_CEILING',actual_native_path_numeric_wallclock_MFU_or_speedup_proved=False,
        all_algorithms_scope=False,overall_complete=False))
    split=[]
    for m in rows:
        for partitions in [1,2,4,8,16,32]:
            assert (5120//partitions)%16==0
            split.append(dict(M=m,K=5120,N=32,K_partitions=partitions,K_per_partition=5120//partitions,
                dense_arithmetic_OPS=2*m*5120*32,
                independent_full_M16_N16_reference_tiles=math.ceil(m/16)*2*partitions,
                materialized_FLOAT_partial_write_read_reference_bytes=8*partitions*m*32 if partitions>1 else 0,
                chosen_FLOAT_merge_add_count=(partitions-1)*m*32,
                one_final_BF16_output_value_bytes=2*m*32,
                same_distinct_weight_bytes=2*5120*32,
                no_partial_BF16_round=True,
                reference_tiles_not_actual_active_cores=True))
    seq=f32(f32(2**25-2**25)+1);other=f32(2**25+f32(-2**25+1));assert seq==1 and other==0
    write(O/'index-weight-partition-and-precision-v106.json',dict(models=split,
        source_order_counterexample=dict(BF16_representable_products=[2**25,-2**25,1],ordered_FLOAT_sum=seq,regrouped_FLOAT_sum=other,
            conclusion='A split-K FLOAT merge may change the final BF16 result even without BF16 partial rounding;native accumulation order/tolerance remains a separate proof.'),
        scope='Chosen conventional split-K alternative for narrow32-column Weight projection,not a claim current native MatMul uses only2cores. More independent product tiles may trade workspace/reduction/numeric order for parallelism;original input union/weights do not multiply by partitions.',
        choices_not_additive='Standalone split-K,QA/Weight common-input panel,and query/weight stream-hoist are alternative/composable graphs only after one shared cost/reduction/lifetime audit;do not add independent gains.',
        actual_native_split_K_or_speedup_proved=False,overall_complete=False))
    b=3*2**-20;q=4096;first=f16(b/64);correct=f16(first*q);dropped=f16((b/64)*q)
    assert correct!=dropped and f16((127*q)/127)==q
    safe=0
    for e in range(-8,21):
        for sig in [1,1.25,1.5,1.9921875]:
            x=sig*2**e;y=x/64
            if 2**-14<=abs(y)<=65504 and f16(y)==y:
                for scale in [2**-24,2**-10,1,32]:
                    if abs(y*scale)<=65504:
                        assert f16(f16(y)*scale)==f16(y*scale);safe+=1
    dags=[]
    for qi,r,w,u in product([0,1,4,16],repeat=4):
        old=qi+max(r,w)+u
        early=max(qi+r,w)+u
        late=max(qi+r+u,w)
        dags.append(dict(IndexQ=qi,RoPE=r,Weight=w,independent_tail=u,
            old_finish=old,early_weight_start_finish=early,early_start_late_join_finish=late,
            ideal_unlimited_resource_example_only=True))
    write(O/'index-projection-rounding-and-readiness-v106.json',dict(
        prior_rounds='Keep BF16 Weight projection round and FP16(weight/64) before FP16 product with FP16Qscale.',
        intermediate_FP16_counterexample=dict(BF16_projection=b,weight_scale=1/64,FP16_query_scale=q,
            valid_query_amax=127*q,required_first_FP16_weight=first,required_FP16_product=correct,
            product_if_first_FP16_round_dropped=dropped,
            scope='Finite BF16/F16 round-to-nearest reference,no nativeFTZ/CANNcast attestation. Qamax is a BF16 representable value producing this valid FP16 scale in the declared query quantizer.'),
        sufficient_single_round_domain='If scaled BF16 Weight is exactly representable finite FP16,and Qscale is finite FP16,its exact product fits FP32 precision/range in this declared domain;the first FP16 round is identity. Unknown actual values do not prove this domain.',
        finite_safe_domain_cases=safe,DAG_examples=dags,
        no_new_source_dependency='Weight only needs query_hidden;IndexQ needs qr. Current weights_start comes after IndexQ;new early start and late first-reader join keep input publication/readers/events/capture generations.',
        caution='DAG service examples are feasible ideal schedules,not lower bounds,actual overlap availability or measured speedups. Independent tail denotes eligible separate main-stream work;Weight andIndexQuant share one auxiliary stream in all640 savedfrontiers and remain serialized unless the graph/stream contract is separately changed. Both matrix paths compete for resources;changing branch/control/error timing must fit the declared valid domain.',overall_complete=False))
    clause=dict(id='REL106-01',a='F05',b='F10',relation='UPSTREAM_QA_INDEX_WEIGHT_COMMON_HIDDEN_PANEL',
        operator_claim='Extend the QA producer variant of F05 with a separate32-output BF16IndexWeight panel sharing query_hidden;then release1280QA toitsownRMS andWeight32 toitsownpostscale.',
        applicable_domain='Exact source hidden/row/weight/config/generation equality;observed ordinaryBF16QA andWeight projections. Different QR1280IndexQ remains separate. Optional globalSWA/C1 joins retainv99Jmap andindependentRMS/round/release contracts.',
        evidence=['index-projection-source-contract-v106.json','index-projection-native-frontiers-v106.json','index-projection-work-and-resource-models-v106.json','index-projection-rounding-and-readiness-v106.json'],
        conditional_compatibility=True,all_runtime_gates_proved=False,speedup_additive=False,critical_path_gain_us=None)
    write(O/'index-projection-composition-v106.json',dict(clauses=[clause],refine_existing_relation='REL83-02',
        extend_group='HG75-Q-KV-JOIN',new_members=['F10'],new_groups=0,new_main_candidates=0,
        keep_old_distinct_IndexQ_Weight_input_contract=True,whole_pair_all_variants_closed=False,overall_complete=False))
    prior=['index-context-fusion-costs-v64.json','Q_B-index-epilogue-extension-v82.json','candidate-relation-clauses-v105.json',
        'projection-panel-work-and-resource-models-v99.json','typed-capacity-primary-contract-v89.json','whole-scope-requirement-audit-v105.json']
    write(O/'index-projection-inputs-v106.json',dict(inputs=[pin(p)for p in paths+[graph_path,old_front_path]]+[pin(O/n)for n in prior]+[pin(Path(p['path']))for p in parts],
        prior_answers_not_new=['v64Weight cast/scale/FP16interface','v82IndexQepilogueextension','v83IndexQ/Weightdistinct-inputjoin','v99QA/SWA/C1raggedpanel'],
        no_server_model_Torch_NPU_capture_or_implementation=True,overall_complete=False))
    print(json.dumps(dict(status='ANALYZED',frontiers=640,reused_TASKs=len(tasks_unique),M_domains=rows,projection_cohorts=len(cohorts),new_pairs=1,overall_complete=False)))

if __name__=='__main__':main()
