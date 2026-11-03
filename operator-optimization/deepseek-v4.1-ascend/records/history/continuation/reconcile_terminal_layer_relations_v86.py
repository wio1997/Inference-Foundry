"""Record conditional last-layer row demands across existing operator candidates."""
from finalize_origin_coverage_v9 import O,read,write

def main():
    assert not(O/'terminal-layer-composition-clauses-v86.json').exists()
    old=read(O/'candidate-relation-clauses-v85.json');pairs={frozenset((c['a'],c['b']))for c in old['clauses']}
    evidence=['terminal-layer-prior-evidence-v86.json','terminal-layer-operator-source-v86.json',
      'terminal-layer-native-frontier-v86.json','terminal-layer-demand-cut-v86.json',
      'terminal-layer-owner-and-work-models-v86.json','terminal-layer-weight-and-dependency-proof-v86.json']
    records=[
      ('F01','ATTENTION_PRE_FULL_KV_DEMAND_VS_SELECTED_FFN_PRE','First attention HcPre/RMS inputs retain full required KV rows; FFN HcPre after selected attention outputs can be restricted conditionally.',
       'Preserve current HF32/numerical stage semantics,coefficients,prior premix/residual,3aux and all state writers; equal shapes/row-local formulas do not prove native batch-shape parity.'),
      ('F04','SELECTED_FFN_NORM_DUAL_OUTPUT_REMAINS','Apply final-layer row demand consistently to fused RmsNormCast and all its BF16 expert/shared and widened-FLOAT router readers.',
       'Both outputs from rounded normalization remain for each retained row. Full router/state observers or EPLB may demand additional rows; no use of unrounded FLOAT norm in place of widened BF16.'),
      ('F14','SELECTED_ATTENTION_QUERY_OUTPUT_AND_ORIGINAL_COORDINATES','Demand can move from final tail back through attention output only with exact original query/position/sequence metadata.',
       'Keep all required historical/current KV and causal/sink/window semantics. Inverse RoPE uses original row positions,not compact ordinals; state publication is a separate full-demand branch.'),
      ('F15','SELECTED_WO_A_ROWS_AND_FULL_HEAD_GROUPS','Retile selected output rows through attention-head exchange and WO_A while retaining all head-group terms for each row.',
       'Current TP8 G1 weights give Uglobal WO_A rows on every head-weight owner; Q owner u_p is different. New ragged A2A/metadata/order/capture ABI and BF16 intermediate preservation are required.'),
      ('F16','SELECTED_WO_B_REDUCESCATTER_AND_HC_ROWS','Compose selected WO_B rows,TP partial reduction and attention Post with the final-layer demand graph.',
       'Preserve each per-rank BF16 partial and native SUM/output round. ReduceScatter target owner/inverse mapping must agree on all ranks; old full-M copy/pad and new selected budgets cannot be added independently.'),
      ('F20','SELECTED_ROUTER_WITH_FULL_SIDE_EFFECT_GUARDS','Final-layer row demand may restrict router work only when exact kept logits/IDs/TopK and all routing state observations are retained.',
       'Keep -1/vision/hash predicate,unbiased weights/scaling,tie/NaN semantics,weight generations and input-ID mapping. EPLB counters/load/heat may observe omitted rows; current false/empty settings are not inferred from trace absence.'),
      ('F21','SELECTED_MC2_ROWS_ARE_NOT_RECEIVER_PAD_COUNTS','Price selected source rows jointly with fused DispatchFFNCombine and all EP16 receiver demands.',
       '6U is source-group assignment reference,not each receiver count. Sum all DP/TP source groups,route maps,padded expert blocks/quant/combine,rank-uniform metadata and state epochs. Retain zero/empty writer contracts.'),
      ('F22','SELECTED_SHARED_ROWS_WITH_NONPROPORTIONAL_WEIGHT_COST','Shared-expert source rows may reduce to retained u_p while cold shared weight planes and all retained-row quantization remain.',
       'INT8 dense work and DynamicQuant/activation scales are typed references. Any positive u_p may still traverse the full gate/up/down weights; no proportional latency or weight savings assumption.'),
      ('F32','SELECTED_QUERY_BRANCH_DOES_NOT_PRUNE_KV_INPUT_GATHER','Separate selected Q from full-required KV input demand; project-before-gather is a different existing F32 graph.',
       'Exact SP→CP row ownership,all normalized-input/cache/index/compressor readers remain. Query pruning alone earns no full input-gather or KV-projection credit; joint F32 representation must be repriced once.'),
      ('F33','TERMINAL_QUERY_DEMAND_AND_COMPRESSION_STATE_CUT','Output-demand restriction does not by itself remove C1/C2/SWA projection/cache or compression state writer branches.',
       'Captured terminal schema has an explicit SWA KV/cache path; alternative layers/branches retain their own C1/C2/index/ring writers. No last-layer35node count is proof against other branches or future-state readers.')]
    clauses=[]
    for i,(peer,relation,claim,guard)in enumerate(records,1):
        assert frozenset(('F26',peer))not in pairs
        clauses.append(dict(id=f'REL86-{i:02d}',a=peer,b='F26',relation=relation,operator_claim=claim,
          applicable_domain='Conditional final target layer39 with complete output-reader union and raw/normalized observer/alias/lifetime guards; '+guard,
          evidence=evidence,conditional_compatibility=True,all_runtime_gates_proved=False,speedup_additive=False,critical_path_gain_us=None))
    groups=[dict(id='HG86-TERMINAL-QUERY-STATE-CUT',members=['F01','F03','F04','F14','F15','F16','F20','F21','F22','F23','F26','F32','F33'],
      contract='Selected final outputs can induce a smaller Q/attention-output/FFN graph under all-reader/numeric/metadata/state guards. Full required KV/cache/aux/previous-layer input demand remains. Typed work/weight and final-Post transfer unions are one changed graph,not additive independent speedups.'),
      dict(id='HG86-SELECTED-OWNER-PUBLICATION',members=['F26','F27','F31','F35'],
      contract='Prepared head I and optional prompt J define distinct union U with ordered inverse maps; context acceptance I is later and distinct. Owner u_p/maxq controls equal-count publication. Keep actual headK,all prompt/probability/RNG/aux/state readers; quotas and physical capacity remain conditional.')]
    for g in groups:g.update(evidence=evidence,scope_state='CONDITIONAL_OPERATOR_DEMAND_AND_TYPED_COST_REVIEWED',all_subsets_analyzed=False,legal_joint_fusion_proved=False)
    write(O/'terminal-layer-composition-clauses-v86.json',dict(clauses=clauses,higher_order_groups=groups,new_pair_clauses=10,
      new_higher_order_groups=2,new_maincandidates=0,refine_existing_relations=['REL73-21','REL74-08','REL85-01','REL85-02','REL85-05'],overall_complete=False))
    print(dict(new_pairs=10,scoped_pairs=105,higher_order_groups=21,central_pending=490,native_fusion=False))

if __name__=='__main__':main()
