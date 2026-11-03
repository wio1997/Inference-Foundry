"""Source-domain tail rewrites, row-local proof, resource work and transfer credits."""
from finalize_origin_coverage_v9 import O,read,write
from model_Engram_WKV_frontier_v84 import f32,bf16
from math import sqrt


def main():
    assert not(O/'DSpark-tail-reorganization-variants-v85.json').exists()
    proof=read(O/'DSpark-tail-reader-closure-v85.json');assert not proof['raw_buffer_source_value_read_in_this_pipeline']
    conditions=[
        'Pinned method=dspark normal Ascend serving callgraph; no extra out-of-band raw getter client/custom hook. Other MTP/nonaux/GPU caller cases retain their own observers.',
        'All participating TP ranks use the same selected rewrite and unchanged padded row ownership/concat/trim. Preserve per-row BF16 R, FLOAT premix, original within-row reduction/rounding policy and initialized storage/error behavior.',
        'Keep all normalized target output rows,3aux checkpoint means/context inputs,draft raw confidence inputs,current sampling/raw/probability/RNG/acceptance outputs. Removing target raw HC is not evidence those values are dead.',
        'Native numerical equivalence under changed batch shape/tiling and new fused ownership/output ABI/capture liveness is not attested here. Source row-local identities do not prove device bitwise parity or speedup.'
    ]
    variants=[
        dict(id='F26-DSPARK-SNAPSHOT-COPY-DEAD',state='SOURCE_VALUE_READER_CLOSED_IN_STATED_PIPELINE',
            graph='Keep current buffer/control/RAW-HC+premix gathers and arithmetic; omit only the unused raw snapshot copy.',
            conditions=conditions,arithmetic_reordered=False,requested_buffer_capacity_unchanged=True,actual_gain=None),
        dict(id='F26-DSPARK-NO-RAW-BUFFER',state='CONDITIONAL_ROW_LOCAL_REPARTITION_REVIEWED',
            graph='Existing bufferNone tail: local BF16 HC collapse -> BF16 collapsed AllGather -> original global-row RMS -> unchanged target outputs',
            conditions=conditions+['NoRaw is a buffer-presence/control rewrite, not just deleting a copy. Norm remains after gather with its original per-rank gamma/eps and global N shape.'],
            source_existing_alternate_path=True,actual_gain=None),
        dict(id='F26-LOCAL-NORM-BEFORE-GATHER',state='CONDITIONAL_REPLICATED_NORMALIZER_REPARTITION_REVIEWED',
            graph='Local HC collapse BF16 -> local RMS BF16 -> normalized BF16 AllGather -> unchanged full normalized output',
            conditions=conditions+['Gamma,eps and numerical normalization policy must be equal across participating TP ranks. Native M-versus-N reduction policy must preserve the required BF16 result. This extra premise is not needed by the preceding collapsed-AG/global-norm plan.'],actual_gain=None),
        dict(id='F03-F26-LAST-POST-COLLAPSE',state='CONDITIONAL_FINAL_OUTPUT_READER_FRONTIER_REVIEWED',
            graph='Last FFN HcPost BF16 tiles -> final weighted collapse BF16 -> optional local RMS -> one chosen BF16 AllGather',
            conditions=conditions+['Only final FFN post output is private after raw observer closure. Its input residual/current post+comb and final premix remain inputs. Interior post outputs still feed later post/aux/Engram and cannot inherit this elision.',
                'This final collapse is pure weighted HC sum, not another HcPre: do not invent HF32/A2/Sinkhorn coefficient work after the final post. Retile/co-own all4HC streams per output H tile, preserve separate post Mul/Add order and both BF16 barriers.'],actual_gain=None),
        dict(id='F23-F26-LAST-MERGE-POST-TAIL',state='CONDITIONAL_HIGHER_ORDER_VALUE_AND_RESOURCE_FRONTIER_REVIEWED',
            graph='shared+routed BF16 merge -> last HCPost BF16 -> HCcollapse BF16 -> local RMS BF16 -> selected AG/head path',
            conditions=conditions+['Keep effective routed scaling/row transforms,merge BF16 rounding and the post/collapse/norm rounds. Count merge,post and collapse temporaries once; optional DownAdd is a separate Vector-owner/readiness domain with INT32 CV bridge retained.',
                'Head selection/processors/fullscore/probability/entropy/bonus output protocols remain separate contracts. Raw HC snapshot closure alone does not authorize a best-ID-only head.'],actual_gain=None)]
    write(O/'DSpark-tail-reorganization-variants-v85.json',dict(variants=variants,common_conditions=conditions,
        source_observer_question_closed=True,native_alias_or_new_numerical_pipeline_proved=False,
        no_new_maincandidate=True,implementation=False,native_test=False,overall_complete=False))
    write(O/'DSpark-tail-row-local-proof-v85.json',dict(
        domain='Ordered raw rows R_p[r,j,h] and premix a_p[r,j] from same SP group and row generation; C preserves original per-row h-reduction and BF16 rounding. G concatenates rank rows in the same order then trims the same suffix.',
        collapse_identity='For every retained output row k=p*M+r, C(G(R),G(a))[k,:] is exactly C(R_p[r,:,:],a_p[r,:]); hence C(G(R),G(a))=G(C(R,a)) for the same row function, including casts/rounding.',
        norm_identity='G(N_p(C_p))=N_receiver(G(C_p)) only when the row-normalizer gamma/eps/numerical policy of every owner equals the receiver. The existing localC->G->globalN plan does not need this crossrank-normalizer equality premise.',
        final_post_identity='P computes rounded BF16 R_j using original separate FP32 Mul/Add order. C(P(...)) may consume those rounded values locally; replacing P by unrounded partials before C is not equivalent.',
        padding_and_empty_domain='Preserve exact initialized physical row range/group participation and final trim; a locally empty logical row set is not permission to drop a collective. New computations on padded rows must stay within valid storage and preserve observable numerical/error behavior.',
        protected_readers=['full normalized target hidden','three pre-layer aux mean tensors','draft raw collapsed hidden for confidence','last post input residual,post/comb/final premix'],
        native_kernel_batchshape_or_tiling_equivalence_attested=False,formal_solver_or_device_test=False,overall_complete=False))
    models=[]
    for global_rows in[24,6400,6528,6672]:
        local=global_rows//8;h=4;hidden=5120
        raw_send=7*local*h*hidden*2;mix_send=7*local*h*4;new_send=7*local*hidden*2
        base_vec=36*local*hidden+(11*hidden+1)*global_rows
        existing_none_vec=43*local*hidden+(4*hidden+1)*global_rows
        local_norm_vec=(47*hidden+1)*local
        models.append(dict(N=global_rows,M=local,P=8,h=h,H=hidden,observed_tail_shape=True,
            snapshot_copy_logical_read_write_bytes=4*global_rows*h*hidden,
            ring_reference_raw_HC_send_bytes=raw_send,ring_reference_premix_send_bytes=mix_send,
            ring_reference_collapsed_or_normalized_send_bytes=new_send,
            ring_reference_send_reduction_bytes=raw_send+mix_send-new_send,
            current_lastPost_plus_globalTail_FP32_ordinary_ops=base_vec,
            existing_None_localCollapse_globalNorm_FP32_ordinary_ops=existing_none_vec,
            localCollapse_localNorm_FP32_ordinary_ops=local_norm_vec,
            current_norm_rsqrt_elements=global_rows,local_norm_rsqrt_elements=local,
            current_per_rank_collapse_rows=global_rows,new_owner_collapse_rows=local,
            current_tail_noncollective_value_reference_bytes=(22*h+14)*global_rows*hidden+4*global_rows*h+2*hidden+4*global_rows,
            None_localCollapse_globalNorm_noncollective_value_reference_bytes=(18*h+10)*local*hidden+4*local*h+4*global_rows*hidden+2*hidden+4*global_rows,
            fused_localCollapseNorm_noncollective_value_reference_bytes=2*local*(h+1)*hidden+4*local*h+2*hidden+4*local,
            finalPost_output_plane_private_reference_bytes=4*local*h*hidden,
            partial_rstd_outputs_in_references_are_legacy_API_cost_not_universal_needed_stores=True,
            actual_valid_rows_HBM_wire_or_latency_gain=None))
    write(O/'DSpark-tail-work-communication-models-v85.json',dict(models=models,
        buffer_requested_capacity_bytes=335544320,buffer_requested_capacity_MiB=320,
        capacity_is_tensor_request_not_measured_reclaimable_memory=True,
        comparison_graphs=['G0current rawHC+premixAG,snapshot,globalC/globalNorm','G0Sonly omitdead snapshot copy',
            'G1existingNone path localC/AGcollapsed/globalNorm','G2localC/localNorm/AGnormalized','G3post+C+optionalNorm fusion after localization'],
        vector_work_domain='Conventional evaluation of original formulas on every physical row in the chosen graph:Post=36MH ordinaryFP32;C=7rows*H;RMS=(4H+1)rows plus1rsqrt/row. Not issued instruction counts or all-algorithm universal lower bounds.',
        value_reference_domain='Noncollective semantic value reads/writes of the explicit source-like graph, including retainedweight/rstd APIcost; communication staging/caches/repeatloads/mapping not included. References across different graphs are differences, not independent gains to sum.',
        ring_domain='Declared P8 ringlike AllGather per-rank send reference=(P-1)*localinputbytes. Actual captured raw/premix count/dtype/P used. Not physical HCCS/SIO/SDMA byte measurements or a universal topology lower bound.',
        symbolic_joint_bound='With necessary node-work in a declared algorithm/reader/capacity domain: T>=max(sum_j(lambda_j*W_j)/beta_Vector(lambda),Q_GMread/Bread,Q_GMwrite/Bwrite,Q_linkcut/Bcut,L_true_tile_dependencies). beta/B must be capacity uppers; cast/reduction/rsqrt modes and actual rates remain unbound.',
        data_dependency_order='G1movesCbeforecommunication;G2movesRMSbeforecommunication. This changes overlap/deadlines. Fewerbytes/ops/collectivecalls do not prove latency dominance. Never sum old native TASK/COMM durations into a speedup.',
        actual_numeric_capacity_or_speedup_proved=False,overall_complete=False))
    # A scalar counterexample to dropping the last post's BF16 round before collapse.
    x=2.;post=[0.501953125]*4;res=[256.,-256.,0.,0.];raw=[]
    for j in range(4):
        value=f32(x*post[j])
        for i in range(4):value=f32(value+f32(res[i]*(1. if i==j else 0.)))
        raw.append(value)
    rounded=[bf16(v)for v in raw]
    total=lambda vs:f32(f32(f32(vs[0]+vs[1])+vs[2])+vs[3])
    assert(bf16(total(rounded)),bf16(total(raw)))==(5.,4.)
    norm=lambda vs:[bf16(f32(v*f32(1/sqrt(f32(f32(f32(vs[0]*vs[0])+f32(vs[1]*vs[1]))/2)+1e-20))))for v in vs]
    y=[257.5,256.5];required=norm([bf16(v)for v in y]);unrounded=norm(y)
    assert required!=unrounded
    write(O/'DSpark-tail-rounding-witnesses-v85.json',dict(
        post_BF16_before_collapse=dict(post_input_BF16=x,residual_BF16=res,post_FP32=post,comb='Identity4 in scalar operator example',
            unrounded_post_values=raw,rounded_post_values=rounded,required_Y=5.,bypass_round_Y=4.),
        collapse_BF16_before_RMS=dict(unrounded_Y=y,required_Y_BF16=[bf16(v)for v in y],required_output=required,bypass_output=unrounded),
        scope='Explicit finite f32/BF16-RNE scalar examples with modeled RMS, not native kernels/fixed-weight values or tolerated equivalence. Preserve the original native row function and rounding contracts.',
        native_or_formal_validation=False,overall_complete=False))
    transfers=[
        ('merge.write',2,'M*H'),('merge.read',2,'M*H'),
        ('finalPost_R.write',2,'M*h*H'),('localC_R.read',2,'M*h*H'),
        ('C_Rfloat.write',4,'M*h*H'),('C_Rfloat.read',4,'M*h*H'),
        ('C_Mulfloat.write',4,'M*h*H'),('C_Mulfloat.read',4,'M*h*H'),
        ('C_Yfloat.write',4,'M*H'),('C_Yfloat.read',4,'M*H'),
        ('C_Ybf16.write',2,'M*H'),('Norm_Ybf16.read',2,'M*H')]
    write(O/'DSpark-tail-transfer-credit-v85.json',dict(
        graph_domain='After choosing G1/G2 localization and matching original row/generation. G0snapshot/globalAG credits and G1/G2privatefusion credits are sequential graph comparisons, not independent original-graph transfer removals.',
        transfer_atoms=[dict(id=i,bytes_per_extent=b,extent=e)for i,b,e in transfers],
        conditional_private_interface_models=[dict(M=m,
            merge_Post_localC_private_interface_budget_bytes=92*m*5120,
            merge_Post_localC_localNorm_private_interface_budget_bytes=96*m*5120,
            above_plus_VectorDownAdd_private_S_budget_bytes=100*m*5120,
            chosen_localC_to_localNorm_Y_interface_is_already_in_76MH_collapseNorm_budget=True,
            final_normalized_or_collapsed_local_write_to_AG_is_retained=True,
            previous_input_residual_and_coefficients_retained=True,
            original_down_INT32_CV_bridge_not_deleted=True)for m in[3,800,816,834]],
        derivation='LocalC/Norm private intermediates=8MhH(Rfloat)+8MhH(Mulfloat)+8MH(Yfloat)+4MH(Ybf16)=76MH;finalPostR=4MhH=16MH;merge=4MH. Union96MH;withoutlocalNormYsurface=92MH. OptionalDownS4MHonlyinitsseparatepostcastVectorowner/allreader/readinessdomain.',
        no_sum_with_full_76NH_global_tail_credit_or_old_snapshot_copy_credit_as_independent_same_graph_savings=True,
        actual_HBM_or_critical_path_gain=None,overall_complete=False))
    print(dict(snapshot_reader='SOURCE_CLOSED_FIXED_DSPARK',shape_models=4,
        M3_N24_copy_bytes=1966080,M3_ring_send_reduction_bytes=645456,
        buffer_requested_MiB=320,post_round_counterexample=[5,4],Y_round_outputs=[required,unrounded],
        M3_private_selected_joint_budget=96*3*5120,actual_gain_proved=False))


if __name__=='__main__':main()
