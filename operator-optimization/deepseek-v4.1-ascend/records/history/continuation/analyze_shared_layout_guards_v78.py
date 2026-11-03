"""Operator-only shared-MLP applicability and precision-preserving repartition.

Offline pinned-source, typed-descriptor and arithmetic evidence. No models,
Torch, remote work, device work, implementation, or performance experiments.
"""
from pathlib import Path
from collections import Counter
from itertools import product
import ast,hashlib,struct
from finalize_origin_coverage_v9 import N,O,W,read,write


def bf16(x):
    b=struct.unpack('<I',struct.pack('<f',float(x)))[0]
    b=(b+0x7fff+((b>>16)&1))&0xffff0000
    return struct.unpack('<f',struct.pack('<I',b))[0]


def main():
    assert not(O/'shared-layout-guard-cases-v78.json').exists()
    prior=read(O/'shared-operator-numerical-and-library-contract-v57.json')
    p=Path(prior['caller']['path']);model=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py'
    cfg=Path(prior['model_text_config']['path'])
    pins=[(p,prior['caller']['sha256']),(model,'101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91'),(cfg,prior['model_text_config']['sha256'])]
    for path,h in pins:assert hashlib.sha256(path.read_bytes()).hexdigest()==h
    s=p.read_text();mt=model.read_text();tree=ast.parse(mt)
    mlp=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='DeepseekV41MLP')
    mlptext=ast.get_source_segment(mt,mlp)
    assert mlptext.count('disable_tp=is_sequence_parallel')==2
    assert 'self.act_fn = SiluAndMulWithClamp(swiglu_limit)'in mlptext and 'expert_gate'not in mlptext
    assert 'is_sequence_parallel=self.is_sequence_parallel'in mt
    assert 'not has_lora(self.lora_context)'in s and 'self.quant_type in (QuantType.W8A8, QuantType.W4A8)'in s
    assert 'tp_size = self.moe_config.tp_group.world_size'in s
    textcfg=read(cfg)['text_config'];assert (textcfg['n_shared_experts'],textcfg['hidden_size'],textcfg['moe_intermediate_size'],textcfg['swiglu_limit'])==(1,5120,2304,10.0)
    quantcases=[]
    for lora,upscale,downscale,qt in product([False,True],[False,True],[False,True],['W8A8','W4A8','W4A8MXFP','W8A8MXFP','W8A8FP','W4A4MXFP','OTHER']):
        eligible=not lora and upscale and downscale
        result='A8_INT_FUSED'if eligible and qt in ['W8A8','W4A8']else 'A8_MXFP_FUSED'if eligible and qt=='W4A8MXFP'else 'LINEAR_WRAPPER'
        quantcases.append(dict(lora=lora,up_scale_present=upscale,down_scale_present=downscale,quant_type=qt,path=result))
    modes=[]
    for tp_gt1,sp,rep in product([False,True],repeat=3):
        mode='TENSOR_PARALLEL'if not tp_gt1 else 'SEQUENCE_PARALLEL_SEDP'if sp and rep else 'SEQUENCE_PARALLEL_ONLY'if sp else 'SHARED_EXPERT_DATA_PARALLEL_ONLY'if rep else 'TENSOR_PARALLEL'
        modes.append(dict(physical_TP_gt1=tp_gt1,sequence_parallel=sp,weights_replicated_flag=rep,controller_mode=mode,
                          fixed_model_SP_constructor_creates_full_weights=sp,
                          controller_mode_alone_proves_parameter_layout=False))
    assert Counter(x['path']for x in quantcases)=={'A8_INT_FUSED':2,'A8_MXFP_FUSED':1,'LINEAR_WRAPPER':53}
    assert Counter(x['controller_mode']for x in modes)=={'TENSOR_PARALLEL':5,'SEQUENCE_PARALLEL_SEDP':1,'SEQUENCE_PARALLEL_ONLY':1,'SHARED_EXPERT_DATA_PARALLEL_ONLY':1}
    regs=[r for r in read(O/'operator-shape-ceiling-ledger-v55.json')['observed_regimes']if r['module_candidates']==['M13']]
    typed=[];stages=Counter();rows=Counter()
    for r in regs:
        sig=r['signature'];op=r['operator']
        stage='GATE_UP'if op=='QuantBatchMatmulV3'and sig['outputDataTypes_text']=='INT32'else 'DOWN'if op=='QuantBatchMatmulV3'else op
        stages[stage]+=r['executions'];rows[(r['origin'],r['context'])]+=r['executions']
        if stage in ['GATE_UP','DOWN']:assert sig['inputDataTypes_text'].split(';')[:2]==['INT8','INT8']
        typed.append(dict(regime_id=r['regime_id'],stage=stage,origin=r['origin'],context=r['context'],executions=r['executions'],signature=sig))
    assert len(regs)==42 and sum(r['executions']for r in regs)==22360 and all(v==4472 for v in stages.values())
    write(O/'shared-layout-guard-cases-v78.json',dict(quantization_selector_cases=quantcases,parallel_controller_cases=modes,
        finite_source_case_counts=dict(quantization=56,parallel=8),typed_regimes=typed,typed_executions=22360,typed_stage_executions=dict(stages),
        descriptor_origin_context_counts=[dict(origin=a,context=b,executions=n)for (a,b),n in sorted(rows.items())],
        fixed_model=dict(shared_experts=1,H=5120,I=2304,swiglu_limit=10.0,declared_activation='SiluAndMulWithClamp',
            separate_shared_ctor_guard='n_shared_experts is not None and not mix_placement',
            SP_disables_TP_in_both_linears=True,SP_only_weight_sharding_requires_new_construction_and_ownership=True),
        applicability=[
            dict(domain='Observed42regimes/22360executions',grade='MEASURED_TYPED_DESCRIPTOR',claim='4472each DQ,INT8-weightGateUp→INT32,DSQ,INT8-weightDown→BF16,merge. Current observed typed contract does not justify using INT4/MXFP/LoRA-wrapper precision or kernels.'),
            dict(domain='Pinned shared controller',grade='DERIVED_SOURCE_CASES',claim='A8 means activation width,not weight width; W8A8 and W4A8 share selector but observed weights are INT8. ActiveLoRA or missing scales selects wrappers.'),
            dict(domain='Pinned fixed DeepseekV41MLP constructor',grade='DERIVED_SOURCE',claim='SP true sets disable_tp=True in bothlinears. Generic SP-only controller branch does not establish TP-sharded fixed-model weights. Flag changes alone cannot price a legal weight-sharded rewrite.'),
            dict(domain='Savedconfiguration+source+typedgeometry',grade='CONDITIONAL_APPLICABILITY',claim='Current capturedSP/localrows+fullI weights with sharedDP and MC2 serial-path evidence are consistent with SP+SEDP. No new live-boolean or loadedobject snapshot inferred.'),
            dict(domain='Othermodel/quant/LoRA/Situ/expert-gate/MXFP branches',grade='COUNTERFACTUAL_OR_GUARDED',claim='Retain their correct generic branches when applicable; fixed source constructor and current typed executions do not prove their global absence in every unrecorded execution.')],
        source_case_enumeration_is_not_runtime_branch_hit_count=True,implementation=False,overall_complete=False))

    # A local-shard amax changes the current full-I row quantizer.
    activations=[1.0]*7+[64.0]
    fullscale=max(activations)/127;globalq=[round(x/fullscale)for x in activations]
    localq=[127]*8
    assert globalq!=localq
    a,b=257,-256;before=bf16(a+b);premature=bf16(bf16(a)+bf16(b))
    assert before==1 and premature==0
    write(O/'shared-shard-precision-proof-v78.json',dict(
        fullrow_scale_witness=dict(shard_maxima=activations,one_global_scale=fullscale,global_quantized_values=globalq,
            independently_quantized_shard_values=localq,scope='Eight positive shard maxima;arithmetic reference,not native input replay. Values lie strictly inside the clamp-related activation magnitude envelope;this is not proof of a captured full-model input. A per-shard scale changes full-I quantization.'),
        early_BF16_partial_witness=dict(partial_INT32_dots=[a,b],common_effective_scale=1.0,full_sum_then_BF16=before,BF16_partials_then_BF16_sum=premature,
            realizability='257=16*16+1*1;-256=(-16)*16,remaining integer dot terms zero. Abstract valid INT8 dot/scale witness,not a captured model tensor. Extrazero-weight q127 can set a legal rowmax without changing dot.',
            implication='BF16 ReduceScatter of separately rounded partial Down results is not generally identical to current full INT32-dot→dequant→BF16 contract.'),
        conventional_INT32_domain=dict(K_up=5120,K_down=2304,max_abs_INT8_product=16384,full_up_bound=83886080,full_down_bound=37748736,INT32max=2147483647,
            proof='Triangle bound applies to every subset/intermediate reduction of at mostK terms;no INT32 overflow in this bias-free conventional dot domain. Does not prove loadednative algorithm or FP32 dequant sequence.'),
        precision_preserving_graph=[
            'Construct TP-sharded GateUp/Down weights with correct gated-half/output-column packing;owner/checkpoint/capture contracts must change consistently.',
            'Quantize each original full-H input row once, optionally gather its INT8 values+FLOATrowScale instead of BF16;retain exact quantizer and all BF16 readers.',
            'Compute paired local GateUp columns in INT32, apply the original FP32 dequant/clamped activation, then all-reduce full-row amax across I-shards before original INT8 rounding.',
            'Compute rawINT32 partial Down dots;integerSUM ReduceScatter before any FP32 dequant or BF16round. Owner applies original common scale/dequant ordering and one BF16round.',
            'Join routed branch and execute original BF16merge/HC rounds with full rowmaps, output masks and stream/rank generations.'],
        gates=['New raw-INT32 Down accumulator/output interface;current BF16-output ABI is not a drop-in.',
               'Native dequant operation/order,rounding/NaN/zero/amax and stored BF16weight-scales must match. RawINT32 integer-domain proof alone is insufficient.',
               'INT32SUM ReduceScatter and FLOATMAX AllReduce legality/capture/native collective contracts not attested by this proposed graph.',
               'If sharedgroup has uneven validrows/padding,prove validN/P owner map and preserve padding/fulloutput contracts rather than assumeN=P*m.',
               'No all-rank parameteridentity,exactloadedquantizer,actualnetworkuppercapacity or speedup proof.'],
        status='CONDITIONAL_OPERATOR_REPARTITION_AND_SYMBOLIC_CEILING',implementation=False,native_test=False,overall_complete=False))
    H,I,P=5120,2304,8;models=[]
    for m in [2,3,800,816,834]:
        repops=6*m*H*I;shardops=6*(P*m)*H*(I//P)
        assert repops==shardops
        models.append(dict(local_materialized_rows=m,balanced_global_materialized_rows=P*m,P=P,H=H,I=I,
            replicated_GEMM_INT8_ops_per_rank=repops,sharded_GEMM_INT8_ops_per_rank=shardops,
            replicated_INT8_weight_payload_bytes_per_rank=3*H*I,sharded_INT8_weight_payload_bytes_per_rank=3*H*I//P,
            weight_payload_difference_bytes_per_rank=3*H*I*(P-1)//P,
            additional_naive_BF16_AG_plus_BF16_RS_ring_send_bytes=(P-1)*m*4*H,
            additional_precision_graph_BF16_AG_MAX_INT32_RS_ring_send_bytes=(P-1)*m*(6*H+8),
            additional_precision_graph_quant_AG_MAX_INT32_RS_ring_send_bytes=(P-1)*m*(5*H+12),
            full_global_amax_FLOAT_elements_per_rank=P*m,raw_INT32_RS_input_elements_per_rank=P*m*H,raw_INT32_RS_output_elements_per_rank=m*H,
            shared_replicated_layout_extra_collective_count=0,precision_shard_graph_extra_collective_count=3,
            replicated_input_quant_elements=m*H,quant_before_gather_input_quant_elements=m*H,quant_after_gather_input_quant_elements=P*m*H,
            reference_INT8_dense_compute_us=repops/752e12*1e6,
            replicated_cold_weight_only_1p6TB_reference_us=3*H*I/1.6e12*1e6,
            sharded_cold_weight_only_1p6TB_reference_us=(3*H*I//P)/1.6e12*1e6,
            balanced_model_is_not_actual_unpadded_N_or_all_ranks_layout_proof=True))
    assert models[1]['additional_precision_graph_quant_AG_MAX_INT32_RS_ring_send_bytes']==537852
    write(O/'shared-layout-work-communication-models-v78.json',dict(models=models,
        modeled_domain='Balanced dense materializedrows m/rank,N=P*m,conventional INT8dot,full-row quantization,common output dequant perrow/column. Current uneven/padded cohorts require separate N/P ownership;these are comparison domains,not measured alternate executions.',
        observations=['Weight payload perrank35,389,440→4,423,680B;fullmodel storage/extra metadata or actual VRAM savings not proved.',
                      'GEMM operations perrank remain6mHI. Sharding weights does not divide current SP+replicated token-shard compute by8.',
                      'Preserving full-I quantization adds cross-rank amax;preserving one Down BF16round moves integerSUM beforedequant and widens RS messages. Naive two-BF16-collective cost is not an exact-equivalence budget.',
                      'Cold weight-only reference falls22.1184→2.7648us atpublished1.6TB/s;not actual required HBM traffic or request speedup. AtM800 compute reference exceeds bothweight-only references,so no8xaggregate densecompute ceiling improvement.'],
        communication_formula='Ordinary fixed-width ring send reference:AG_INT8+scale=(P−1)m(H+4);AR_MAX_FLOAT=8(P−1)m;RS_INT32=4(P−1)mH. Sum=(P−1)m(5H+12). Algorithm servicebytes,not a universal network lower bound.',
        theory=dict(state='SYMBOLIC_CEILING',formula='For the declared legalgraph,use necessary work and genuine capacity upperbounds:T>=max(Q_cube/P_INT8,Q_Vector/P_Vector,Q_HBM/B_HBM,Q_network/B_network,dependencybounds). Current bytes/752TOPS/1.6TBs are qualified references;actualtyped/network/caches remain unknown.',
            actual_physical_numeric_ceiling=None,actual_measured_speedup=None),
        no_new_main_candidate=True,implementation=False,overall_complete=False))
    write(O/'shared-layout-source-index-v78.json',dict(source_files=[dict(path=str(p),sha256=h,binding='Reused frozen exact model/operator/config clause;loadedbinary/object not attested.')for p,h in pins],
        source_clauses=['DeepseekV41MLP104–150;MoEsharedconstructor191–208','Sharedcontrollerparallel208–227,input363–423,selector639–661,finish500–637/output670–716'],
        reused_evidence=['shared-operator-numerical-and-library-contract-v57.json','shared-expert-shape-ceiling-v57.json','shared-overlap-activation-contract-v21.json','merge-to-HC-contract-v22.json'],
        new_remote_source_read=False,new_source_files_obtained=0,local_operator_clause_read=True,no_model_Torch_NPU_capture_service_or_process_work=True,overall_complete=False))
    print(dict(shared_regimes=42,shared_executions=22360,selector_cases=56,layout_cases=8,
               BF16_witness=[before,premature],M3_ring_reference=537852,compute_ratio=1,new_main_candidates=0,status='BOUNDED_ANALYSIS_READY'))


if __name__=='__main__':main()
