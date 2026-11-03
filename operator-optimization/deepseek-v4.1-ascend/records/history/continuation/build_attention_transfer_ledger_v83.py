"""Concrete transfer identities for union-based joint credit, in a stated boundary model."""
from finalize_origin_coverage_v9 import O,write


def main():
    assert not(O/'attention-transfer-credit-ledger-v83.json').exists()
    # Coefficients multiply M, sum_L, sum_n, dirty rows, or Mproj separately.
    rows=[
      ('Qscale.write','M',64,'PRIVATE_FP16_SCALE',['F09','F10','F12'],True),
      ('Qscale.read','M',64,'PRIVATE_FP16_SCALE',['F09','F10','F12'],True),
      ('W_FP16.write','M',64,'PRIVATE_FP16_WEIGHT',['F10','F12'],True),
      ('W_FP16.read','M',64,'PRIVATE_FP16_WEIGHT',['F10','F12'],True),
      ('Q_INT8.write','M',4096,'RETAINED_A3_VC',['F09','F12'],False),
      ('Q_INT8.read','M',4096,'RETAINED_A3_VC',['F09','F12'],False),
      ('QLI_score.write','sum_L',4,'RETAINED_CURRENT_CV',['F12','F13'],False),
      ('QLI_score.read','sum_L',4,'RETAINED_CURRENT_CV',['F12','F13'],False),
      ('SMLA_score.write','sum_n',256,'RETAINED_CURRENT_CV',['F13','F14'],False),
      ('SMLA_score.read','sum_n',256,'RETAINED_CURRENT_CV',['F13','F14'],False),
      ('SMLA_exp.write','sum_n',128,'RETAINED_CURRENT_VC',['F13','F14'],False),
      ('SMLA_exp.read','sum_n',128,'RETAINED_CURRENT_VC',['F13','F14'],False),
      ('SMLA_unpacked.write','M',65536,'PRIVATE_OUTPUT_LAYOUT_BASELINE',['F13','F14','F15','F16'],True),
      ('inverse_tail.read','M',8192,'PRIVATE_OUTPUT_LAYOUT_BASELINE',['F13','F14','F15'],True),
      ('inverse_tail.write','M',8192,'PRIVATE_OUTPUT_LAYOUT_BASELINE',['F13','F14','F15'],True),
      ('pack_unpacked.read','M',65536,'PRIVATE_OUTPUT_LAYOUT_BASELINE',['F13','F14','F15','F16'],True),
      ('final_packed.write','M',65536,'RETAINED_SELECTED_SEND_LAYOUT',['F14','F15','F16'],False),
      ('selected_indices.publish','M',2048,'RETAINED_LIVE_BANK',['F12','F13'],False),
      ('C1_candidates.publish','M',8192,'RETAINED_MODE1_LIVE_BANK',['F12'],False),
      ('IndexK.publish','dirty_index',128,'RETAINED_LIVE_CACHE',['F11','F12'],False),
      ('IndexScale.publish','dirty_index',2,'RETAINED_LIVE_CACHE',['F11','F12'],False),
      ('compressed_KV.publish','dirty_cmp',1024,'RETAINED_LIVE_CACHE',['F13','F19'],False),
      ('SWA_KV.publish','dirty_swa',1024,'RETAINED_LIVE_CACHE',['F08','F13'],False),
      ('OA_BF16.write','Mproj',2048,'CONDITIONAL_CC_BF16_INTERFACE',['F14','F15','F16'],True),
      ('OA_BF16.read','Mproj',2048,'CONDITIONAL_CC_BF16_INTERFACE',['F14','F15','F16'],True)]
    transfers=[dict(id=identifier,extent_symbol=symbol,bytes_per_extent=coefficient,
        role=role,candidate_claimants=claimants,conditionally_removable_in_its_domain=removable,
        native_address_or_transfer_count_attested=False)for identifier,symbol,coefficient,role,claimants,removable in rows]
    scalar_ids={'Qscale.write','Qscale.read','W_FP16.write','W_FP16.read'}
    output_ids={'SMLA_unpacked.write','inverse_tail.read','inverse_tail.write','pack_unpacked.read'}
    byid={r['id']:r for r in transfers}
    value=lambda ids,m:sum(byid[i]['bytes_per_extent']*m for i in ids)
    examples=[]
    for count in [2,3,800,816,830,834]:
        examples.append(dict(M=count,private_scalar_frontend_union_bytes=value(scalar_ids,count),
            output_boundary_report_A_bytes=value(output_ids,count),
            output_boundary_report_B_same_transfer_bytes=value(output_ids,count),
            wrong_sum_of_two_output_reports_bytes=2*value(output_ids,count),
            correct_unique_output_transfer_union_bytes=value(output_ids|output_ids,count),
            joint_scalar_plus_output_union_bytes=value(scalar_ids|output_ids,count),
            selected_layout_publication_still_required_bytes=65536*count))
    write(O/'attention-transfer-credit-ledger-v83.json',dict(transfers=transfers,
        output_boundary_domain='SMLA writes unpacked BF16 once; inverse RoPE updates tail64 in place; separate pack reads unpacked plane and writes final packed BF16. A joint epilogue preserves both BF16 rounds and emits final packed plane. These event identities are a conditional model, not observed addresses or a universal source layout.',
        scalar_boundary_domain='Only private original FP16 Qscale and processed weight store/read are removed when their exact rounded values/readers are fully absorbed; INT8 query and current QLI/SMLA internal queues remain.',
        OA_domain='Same OA invocation, Mproj rows after exact CP/head/padding mapping, width1024 BF16 intermediate. CC fusion requires consistent owner/tiling/BF16 rounding. Mproj is not automatically query M and is never substituted in examples.',
        publication_domain='Selected/candidate banks only in active mode with live readers; dirty unique cache rows under unchanged raw disjoint GM-bank ABI. Different modes/extents are not all simultaneously mandatory.',
        examples=examples,
        compute_credit_atoms=[dict(id='QLI.INT8_QK',conventional_ops='8192*sum_L',distinct_from='SMLA.BF16_QK_PV'),
            dict(id='QLI.FP16_headreduce',conventional_useful_ops='64*sum_L',native_M16_padding_not_universal=True),
            dict(id='SMLA.BF16_QK_PV',conventional_ops='131072*sum_n'),
            dict(id='OA',conventional_ops='2*Mproj*4096*1024',claimants=['F14','F15','F16'],count_same_invocation_once=True)],
        joint_credit_rule='First choose one legal transformed graph, partition and value generation. Credit each original transfer/compute atom once by set union; preserve all retained/new transfers and readiness/resource costs. Same role name across invocations is not identity.',
        actual_HBM_traffic_or_joint_gain=None,conditional_operator_analysis_deliverable=True,overall_complete=False))
    print(dict(transfer_atoms=len(transfers),compute_atoms=4,output_M3_union_bytes=value(output_ids,3),
        scalar_plus_output_M3_union_bytes=value(output_ids|scalar_ids,3),not_actual_gain=True))


if __name__=='__main__':main()
