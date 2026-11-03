"""Reprice mixed projection graphs with typed ingress, communication and precision gates."""
import math,struct
from finalize_origin_coverage_v9 import O,read,write

def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
def bf16(x):
    b=struct.unpack('I',struct.pack('f',x))[0];b=(b+0x7fff+((b>>16)&1))&0xffff0000
    return struct.unpack('f',struct.pack('I',b))[0]

def main():
    assert not(O/'mixed-SWA-C2-cost-models-v88.json').exists()
    evidence=['mixed-SWA-C2-source-contract-v88.json','mixed-SWA-C2-native-frontiers-v88.json',
      'A3-datapath-primary-evidence-v80.json','projection-operator-numerical-contract-v61.json','joint-operator-resource-cut-proof-v83.json']
    models=[]
    for n in[24,6400,6528,6668,8192]:
        H,D=5120,512
        models.append(dict(N=n,H=H,D=D,
          exact_new_native_triple_join=(n==24),capacity_or_prior_row_geometry_reference_only=(n!=24),
          source_input_plane_GM_reference_bytes=16*n*H,
          FLOAT_pair_joined_plus_SWA_input_plane_GM_reference_bytes=12*n*H,
          already_F17_FLOAT_pair_join_credit_bytes=4*n*H,
          hypothetical_all_BF16_exact_reencoding_input_plane_reference_bytes=2*n*H,
          hypothetical_credit_vs_source_if_reencoding_and_all_ownership_gates_hold_bytes=14*n*H,
          original_SWA_BF16_dense_ops=2*n*H*D,C2_two_FLOAT_dense_ops=4*n*H*D,
          hypothetical_reencoded_three_BF16_dense_ops=6*n*H*D,
          retained_SWA_BF16_Cube_to_Vector_bridge_reference_bytes=4*n*D,
          retained_two_C2_FLOAT_Cube_to_Vector_bridge_reference_bytes=16*n*D,
          one_all_FLOAT_output_plus_UB_BF16round_Norm_extra_vs_segmented_output_reference_bytes=4*n*D,
          one_all_FLOAT_output_then_separate_BF16_cast_plane_extra_vs_segmented_output_reference_bytes=8*n*D,
          simultaneously_entire_BF16_and_FLOAT_A_L1_plan_bytes=6*n*H,
          simultaneously_all_three_FP32_accumulator_outputs_L0C_plan_bytes=12*n*D,
          actual_HBM_or_latency_gain=None))
    weights=dict(SWA_BF16=2*5120*512,C2_wkv_FLOAT=4*5120*512,C2_wgate_FLOAT=4*5120*512)
    weights_reencoded=dict(SWA_BF16=2*5120*512,C2_wkv_BF16=2*5120*512,C2_wgate_BF16=2*5120*512)
    cuts=[]
    for copies in[1,3]:
        for representation,w in [('original_typed_planes',weights),('conditional_exact_BF16_reencoding',weights_reencoded)]:
            total=copies*sum(w.values());cores=24
            grantB=cores*65536;grantAB=cores*(65536+65536)
            cuts.append(dict(C2_source_windows=copies,representation=representation,distinct_uncompressed_weight_bytes=total,
              max_AIC_in_declared_die_model=cores,fixed_B_initial_L0B_capacity_grant_bytes=grantB,
              fixed_B_L1_to_L0B_max_bytes_per_aggregate_cycle=cores*128,
              fixed_B_min_aggregate_elapsed_cycles=math.ceil(max(total-grantB,0)/(cores*128)),
              flexible_orientation_initial_L0A_L0B_grant_bytes=grantAB,
              flexible_orientation_relaxed_max_bytes_per_aggregate_cycle=cores*(256+128),
              flexible_orientation_relaxed_min_aggregate_elapsed_cycles=math.ceil(max(total-grantAB,0)/(cores*(256+128))),
              actual_wallclock_lower_bound_us=None))
    write(O/'mixed-SWA-C2-cost-models-v88.json',dict(models=models,weights=weights,conditional_reencoded_weights=weights_reencoded,
      theory_status='CONDITIONAL_CEILING',universal_algorithm_lower_bound=False,
      weight_ingress_cuts=cuts,
      input_plane_derivation='Source:BF16SWAread2NH + cast(BF16read2NH,FLOATwrite4NH) + twoFLOATprojection reads8NH=16NH. JoiningFLOATpair reduces its input read reference to4NH,total12NH. This4NH is existingF17 benefit,not a new extra SWA cast-CSE. Cross-owner BF16/FLOAT loads are not one physical transfer merely because source values match.',
      reencoding_domain='2NH reference needs a different all-BF16 operand graph:actual C2weights exactly BF16-representable,original FLOAT/HF32 reduction/accumulation/rounding matched,C2outputs stillFLOAT,and source-sharing/row/readiness/layout gates. FLOATparameter dtype doesnot attest this. Alternative lawful Cube ingress conversion is separate and not proved by a newer UB→L1 API.',
      output_domain='SWA BF16512values and C2 FLOATkv/score512values are different products and readers. A single FLOAT1536-output surface needs a correct SWA BF16 boundary beforeRMS. Segmented mixed-dtype output is a new producer ABI;plain concatenate doesnot make output conversion/bridges free.',
      CV_domain='For all fresh defined outputs crossing conventional AIC→AIV,store+read payload is4ND for SWA BF16 plus16ND for twoC2FLOATplanes. Queue/tile storage may change peak,not automatically eliminate requiredGM ownership service. Padding/unused rows,other algorithms or owners need their own model;these are not physicalHBM bytes.',
      ingress_cut_domain='Declared conventional dense uncompressed weights viaL1→L0,24-AIC die/fixed aggregate-cycle model,all distinct planes required,no imported pre-window in-flight service. Grant initial L0 capacity ONCE for whole window. DirectGM→L0 or compression/weight-reuse/other algorithms are outside this L1path bound. Published rate is capacity upper in this model,not observed bandwidth.',
      ingress_consequence='Original3typed weight planes remain26,214,400B with source,F17join,or owner-local projector. Fewer launches or8times fewer row dot-products doesnot lower this fixed weight-ingress floor. Reencoding changes the domain and Q;it is conditional,not a proven actual speedup.',
      capacity_specific_plan=dict(N=24,H=5120,D=512,full_dual_input_L1_bytes=737280,L1_per_AIC=524288,dual_input_fits=False,
        full_three_FP32_accumulator_bytes=147456,L0C_per_AIC=131072,full_accumulator_fits=False,
        consequence='Only these whole-matrix single-AIC plans fail. Row/K/column tiles,streaming and different ownership remain candidates.'),
      additional_encoded_C2_weight_bank_if_original_FLOAT_parameters_retained_bytes_per_source=4*5120*512,
      actual_encoded_weight_identity_or_native_new_kernel_or_speedup_proved=False,overall_complete=False))
    n,m,H,D,P=24,3,5120,512,8
    old=2*(P-1)*m*H;swa=2*(P-1)*m*D;c2=8*(P-1)*m*D
    write(O/'mixed-SWA-C2-project-before-gather-v88.json',dict(N=n,M=m,P=P,H=H,D=D,
      old_full_hidden_BF16_per_rank_ring_send_reference_bytes=old,
      projected_SWA_BF16_send_reference_bytes=swa,projected_two_C2_FLOAT_send_reference_bytes=c2,
      projected_joint_typed_send_reference_bytes=swa+c2,joint_send_ratio=(swa+c2)/old,
      individual_SWA_only_reference_ratio=swa/old,individual_C2_only_reference_ratio=c2/old,
      one_shared_input_gather_cannot_credit_two_full_baselines=True,
      owner_local_dense_ops=dict(SWA_BF16=2*m*H*D,C2_FLOAT=4*m*H*D),
      original_dense_ops=dict(SWA_BF16=2*n*H*D,C2_FLOAT=4*n*H*D),
      local_CAST_FLOAT_GM_store_read_reference_bytes=8*m*H,
      forcing_QA_into_original_global_N_plan_additional_BF16_dense_ops=2*(n-m)*H*1280,
      domain='Same SP→CP owner-normalized BF16 source rows;project complete SWA/C2 outputs atowners,retain localQ,exchange exact typed planes before global C2pool/cache consumers. N24=8M only;actual padded6668/6672/830/834 and ragged maps remain separate.',
      conditions=['All numerical full-hidden consumers covered;source-pointer/value/weight epoch and new M-vs-N arithmetic policy preserved.',
        'C2pool/globalroworder/oldring/score and all current/future KV/index/cache writes receive identical values.',
        'TwoFLOATandoneBF16 planes need multiple typed collectives or bit-preserving packed ABI;startup/packing/layout/queue costs not zero.',
        'Byte ratio1/2 and arithmeticN/M=8 are separate from runtime speedup;weight-ingress plane set is unchanged.',
        'Q_A source localM is not globalN. A same-row jointGEMM on globalN adds redundant Qwork unless ownership algorithm changes.'],
      actual_new_collective_or_native_precision_or_critical_path_proved=False,overall_complete=False))
    w=f32(1+2**-9);x=256.;original=f32(w*x);encoded=f32(bf16(w)*x)
    assert(original,encoded)==(256.5,256.)
    left=f32(f32(2**24+1.)-2**24);other=f32(f32(2**24-2**24)+1.)
    assert(left,other)==(0.,1.)
    write(O/'mixed-SWA-C2-precision-witnesses-v88.json',dict(weight_reencoding=dict(FLOAT_weight=w,BF16_weight=bf16(w),BF16_input=x,
      original_FLOAT_result=original,reencoded_result=encoded),
      exact_BF16_operands_different_reduction_order=dict(operands=[2**24,1,-2**24],weights=[1,1,1],left_result=left,other_tree_result=other),
      gates='Exact weight representability is necessary for this lossless reencoding route but not sufficient for original native reduction parity. Preserve C2FLOATkv/score/ring values even if later pooled output isBF16.',
      scope='Finite scalarFP32/BF16-RNE semantic witnesses;not actual checkpoint values,loaded kernel behavior or a formal/device proof.',overall_complete=False))
    conditions=['Named BF16/FLOAT source value/row/weight generation and all readers/readiness/capture aliases preserved.',
      'All A3 ingress/GM bridges and output BF16/FLOAT boundaries included;new Cube/AIV ownership or encoding needs explicit proof.',
      'Current native FLOAT/HF32 mode and actual weights/reduction order are unbound;input/weight byte counts are not latency.']
    variants=[
      ('MIX88-ORIGINAL-TYPED-JOIN','CONDITIONAL','Coordinate BF16SWA and joined FLOATC2 pair with existing representation;keep GM-mediated cast/typed ingress. ExistingF17input-read credit,not14NH extra.'),
      ('MIX88-ALL-BF16-OPERANDS','CONDITIONAL_REPRESENTABILITY_AND_REDUCTION_GATE','Encode C2weights asBF16 only if exact and match original accumulation;emit original FLOATC2results and rounded BF16SWA results.'),
      ('MIX88-LOCAL-PROJECT-TYPED-EXCHANGE','CONDITIONAL_REPARTITION','Move complete typed projections before shared input gather,retain localQ,send5120typedB/row rather than10240hiddenB/row in declared case.'),
      ('MIX88-FLOAT-WIDE-OUTPUT','CONDITIONAL_EXTRA_OUTPUT_CONVERSION_COST','A single FLOAT1536output needs original SWABF16round. Price new mixed segmented stores or extraCtoV/cast output service.'),
      ('MIX88-WHOLE-DUAL-A-AND-C','EXCLUDED_SPECIFIC_CAPACITY_PLAN','N24full BF16+FLOAT A cannot both reside oneL1,and all3FP32accumulators exceed oneL0C. Tiled alternatives remain open.')]
    write(O/'mixed-SWA-C2-variants-v88.json',dict(variants=[dict(id=i,decision=s,graph=g,conditions=conditions,evidence=evidence,
      implementation=False,native_test=False,critical_path_gain_us=None)for i,s,g in variants],new_maincandidates=0,overall_complete=False))
    print(dict(models=5,variants=5,weight_cuts=4,typed_projected_send=[old,swa+c2],precision_witnesses=2,actual_speedup=False))

if __name__=='__main__':main()
