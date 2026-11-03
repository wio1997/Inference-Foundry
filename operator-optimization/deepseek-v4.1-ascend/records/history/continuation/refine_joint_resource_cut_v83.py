"""Close the named composition-model evidence gaps without new source/device work."""
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import O,read,write


def main():
    output=O/'joint-operator-resource-cut-proof-v83.json'
    assert not output.exists()
    ring_names=['ring-storage-source-contract-v79.json','ring-state-layout-and-lifetime-proof-v79.json','ring-storage-capacity-models-v79.json']
    receipts=[]
    for name in ring_names:
        path=O/name;blob=path.read_bytes()
        receipts.append(dict(path=str(path),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()))
    ring=read(O/ring_names[0]);capacity=read(O/ring_names[2])
    assert ring['owner_topology']['number_of_ring_objects']==3
    assert capacity['standalone_even_compaction_pool_divisor_change']==0
    write(O/'ring-persistence-input-supplement-v83.json',dict(files=receipts,
        resolves_missing_optional_name='C2-physical-ring-storage-models-v79.json was a stale retrieval name, not an absent operator analysis. Actual source/layout/capacity artifacts inspected and pinned here.',
        facts=dict(ring_objects=3,ring_owners_are_not_all_ratio2_layers=True,
            standalone_even_compaction_global_pool_divisor_change=0,
            private_and_paired_bank_lifetime_required=True),
        scope='Existing frozen operator-state records only; no new framework or server query.',overall_complete=False))
    # A valid joint resource bound can be expressed without guessing whether
    # two instruction classes share lanes, modes, pipeline issue or frequency.
    proof=dict(
        declaration='Fix algorithm/precision/representation domain D and one physical shared resource cut. Let W_j be a lower bound on necessary completed work of typed class j during the measured window, after removing duplicate work and granting allowed initial residency once.',
        capacity_premise='For nonnegative lambda_j, prove for every legal instantaneous allocation/mode/frequency in D that sum_j lambda_j*r_j(t) <= beta(lambda). beta must be a capacity UPPER, not measured best throughput or independent-peak compatibility guess.',
        derivation=['Integrate the capacity inequality over the entire window: integral sum(lambda_j*r_j(t))dt <= T*beta(lambda).',
            'Necessary service is sum_j lambda_j*W_j <= integral sum(lambda_j*r_j(t))dt, including only work proven necessary in D.',
            'Therefore T >= sum_j lambda_j*W_j / beta(lambda). Take the supremum over valid nonnegative weights, and max with other resource/cut/dependency bounds.',
            'Overlapping kernel execution and distributing modes among different AICs do not invalidate an aggregate cut if beta bounds that whole device cut. Initial warm capacity/duplicate work must not be granted once per kernel.'],
        independent_peaks_limit='Knowing Q8<=T*C8 and Qhalf<=T*Chalf alone gives max(Q8/C8,Qhalf/Chalf). It does NOT prove Q8/C8+Qhalf/Chalf<=T; that requires the joint simplex/exclusive-service capacity premise. Do not claim free overlap either.',
        trivially_safe_but_weak='From valid individual rate upper bounds one may use beta(lambda)=sum(lambda_j*C_j), but this may be weaker than choosing an axis. A tighter beta needs architecture/issue/service proof.',
        matching_Vector_rule='Exp/Div/Sort/cast/INT64/FP32 classes need their own joint capacity support. A Vector label does not prove serial issue or a common full-rate arithmetic peak.',
        theorem_grade='DERIVED_ANALYTIC_CONDITIONAL_BOUND;NO_SOLVER_OR_DEVICE_PROOF',
        physical_numeric_beta_for_actual_A3_INT8_plus_BF16_or_Vector=None,
        published_752TOP_and_376TF_joint_simplex_not_proved=True)
    example=dict(domain='Illustrative3validqueries,eachfullQLIscan17408keys andSMLA128SWA+512CMPkeys. Conventional stated exact typed algorithms,not inferred device-visible lengths.',
        valid_queries=3,QLI_keys_per_query=17408,SMLA_keys_per_query=640,
        QLI_sum_L=3*17408,SMLA_sum_n=3*640,
        QLI_INT8_direct_ops=2*32*128*3*17408,
        QLI_useful_FP16_headreduce_ops=2*32*3*17408,
        SMLA_BF16_direct_QK_PV_ops=4*64*512*3*640,
        QLI_scalar_score_GM_interface_bytes=8*3*17408,
        SMLA_score_and_exponential_GM_interfaces_bytes=12*64*3*640,
        query_INT8_VC_GM_interface_bytes=2*3*32*128,
        private_query_scale_plus_weight_interface_budget_bytes=8*3*32,
        joint_Cube_bound='T >= [lambda8*427819008 + lambdah*(3342336+251658240)] / beta_Cube(lambda8,lambdah), only if the useful headreduce class is FP16 and beta jointly bounds the selected mixed modes. Native M16 padding belongs to current implementation, not universal necessary work.',
        only_the_named_scale_interfaces_removed=True,internal_GM_interfaces_are_not_HBM_bytes=True,
        actual_capacity_bound_or_latency_or_speedup=None)
    assert example['QLI_INT8_direct_ops']==427819008
    assert example['QLI_useful_FP16_headreduce_ops']==3342336
    assert example['SMLA_BF16_direct_QK_PV_ops']==251658240
    assert example['QLI_scalar_score_GM_interface_bytes']==417792
    assert example['SMLA_score_and_exponential_GM_interfaces_bytes']==1474560
    write(output,dict(proof=proof,illustrative_joint_window=example,
        relation_to_previous='Refines the max-of-separate-types model with an optional joint support-function capacity cut; no actual joint capacity is invented. Fixed current GM queue volume stays a stated-domain interface model, not an all-algorithm lower bound.',
        conditional_operator_analysis_deliverable=True,overall_complete=False))
    write(O/'index_attention-distinct-score-proof-v83.json',dict(
        index_score='INT8 width128,32heads,DEQF16/ReLU,FP16weights*Qscale headreduce,keyscale and TopK/membership.',
        attention_score='BF16 width512,64heads,selectedSWAcmp positions,onlineSoftmax/sink andPV.',
        conclusion='QLI dot/head score is not SMLA attention score. Sharing positions/latent generation or composing kernels does not remove either ordinary computation without an additional algebraic/precision proof for the fixed weights and query transforms.',
        abstract_projection_counterexample=dict(index_projection=[1,0],attention_query=[0,1],
            latent_A=[1,0],latent_B=[1,1],index_value_A=1,index_value_B=1,attention_dot_A=0,attention_dot_B=1,
            scope='Structural counterexample to a general score-reuse rule; not an assertion about particular frozen weight entries or native instructions.'),
        evidence=['attention-indexer-native-numerical-contract-v60.json','attention-indexer-work-traffic-models-v60.json'],
        all_native_score_reuse_ruled_out=False,overall_complete=False))
    print(dict(joint_capacity_bound='SYMBOLIC_WITH_PROOF',illustrative_ops=[427819008,3342336,251658240],
        retained_internal_GM_bytes=417792+1474560,private_scale_budget=768,stale_input_name_resolved=True))


if __name__=='__main__':main()
