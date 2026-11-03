"""Add concrete mixed-projection relations discovered from current source-graph adjacency."""
from finalize_origin_coverage_v9 import O,read,write

def main():
    assert not(O/'mixed-SWA-C2-composition-clauses-v88.json').exists()
    before=read(O/'candidate-relation-clauses-v87.json');pairs={frozenset((c['a'],c['b']))for c in before['clauses']}
    hints=read(O/'operator-graph-candidate-bindings-v88.json')['structural_pair_hints']
    assert any({h['a'],h['b']}=={'F08','F17'}for h in hints)
    evidence=['operator-graph-coverage-audit-v88.json','operator-graph-candidate-bindings-v88.json',
      'mixed-SWA-C2-source-contract-v88.json','mixed-SWA-C2-native-frontiers-v88.json','mixed-SWA-C2-cost-models-v88.json',
      'mixed-SWA-C2-project-before-gather-v88.json','mixed-SWA-C2-precision-witnesses-v88.json','mixed-SWA-C2-variants-v88.json']
    specs=[
      ('F08','F17','SAME_INPUT_DIFFERENT_BF16_FLOAT_INGRESS',
       'Coordinate SWA BF16 and two C2FLOAT projections using the same declared cache_hidden,without equating products/precision/ingress paths.',
       'Original typed input reference16NH becomes12NH with existing F17FLOATpair join. A2NH all-BF16 route needs exact C2weight encoding and original reductions/output types. A3GM bridges/weight ingress remain;C2FLOAT cannot becomeSWA output or be dropped after later BF16norm.'),
      ('F06','F17','LOCAL_QUERY_VS_GLOBAL_MIXED_PROJECTION_DOMAIN',
       'Compose Q/SWA siblings and C2 projections only after reconciling localQ and global cache row ownership.',
       'N24/M3 captured query/cache domains differ. Forcing oneglobal-rowGEMM adds Qwork;owner-local projection+typed exchange is a separate F32composition. SWA1024B+C2twoFLOAT4096B per row shares one old10240B hidden-gather baseline,not additive independent reductions.'),
      ('F08','F18','SWA_NORM_AND_C2_POOL_NORM_DISTINCT_VALUES',
       'Joint input projections must preserve the different SWA direct512norm and C2perchannel pool→BF16norm branches.',
       'SWA current cache visibility and C2oldring/completion/indexkey/longcache dependencies remain. Equal512width doesnot make norm statistics/latents equal. Mixed output segmentation or additionalFLOAT→BF16service must be priced;no unpriced directAIC→AIV path.')]
    clauses=[]
    for i,(a,b,r,claim,domain)in enumerate(specs,1):
        assert frozenset((a,b))not in pairs
        clauses.append(dict(id=f'REL88-{i:02d}',a=a,b=b,relation=r,operator_claim=claim,applicable_domain=domain,
          evidence=evidence,conditional_compatibility=True,all_runtime_gates_proved=False,speedup_additive=False,critical_path_gain_us=None))
    group=dict(id='HG88-SWA-C2-TYPED-PROJECTION',members=['F06','F08','F17','F18','F32','F33'],
      contract='Choose one original-typed/reencoded/owner-local graph. Include all reader/row/numerical/communication/output costs. One shared initial weight-ingress grant;unchanged cold weight set can bound local-row arithmetic gains. F17input sharing and F32communication credits refer to one baseline.',
      evidence=evidence,scope_state='CONDITIONAL_MIXED_PROJECTION_AND_RESOURCE_CUT_REVIEWED',all_subsets_analyzed=False,legal_joint_fusion_proved=False)
    write(O/'mixed-SWA-C2-composition-clauses-v88.json',dict(clauses=clauses,higher_order_groups=[group],new_pairs=3,new_groups=1,
      refine_existing_relations=['REL73-16','REL74-15'],new_maincandidates=0,overall_complete=False))
    print(dict(new_pairs=3,scoped_pairs=115,higher_order_groups=24,central_pending=480,new_maincandidates=0))

if __name__=='__main__':main()
