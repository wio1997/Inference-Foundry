"""Named HC/Engram/FFN composition clauses from bounded operator evidence."""
from finalize_origin_coverage_v9 import O,read,write


def main():
    assert not(O/'HC-Engram-composition-clauses-v84.json').exists()
    prior=read(O/'candidate-relation-clauses-v83.json')
    previous_pairs={tuple(sorted([r['a'],r['b']]))for r in prior['clauses']}
    evidence=['Engram-HC-operator-source-contract-v84.json','Engram-HC-native-frontiers-v84.json',
        'Engram-WKV-hoist-and-memo-variants-v84.json','Engram-WKV-storage-and-work-models-v84.json',
        'Engram-HC-rounding-witnesses-v84.json','Engram-HC-joint-frontier-models-v84.json',
        'HC-operator-numerical-and-dependency-contract-v59.json','shared-quant-owner-CV-interfaces-v82.json',
        'joint-operator-resource-cut-proof-v83.json']
    rows=[
      ('F01','F03','NEXT_HCPRE_IS_SHARED_WORK_NOT_ANOTHER_COPY',
       'Join HcPost->next HcPre including current coefficient output and delayed external-premix collapse.',
       'F03 may contain the same current HcPre as F01. Compute each coefficient/collapse once, keep BF16 post round and later residual/aux/Engram readers. At configured Engram boundaries insert gate update before HcPre; elsewhere preserve original scope.'),
      ('F01','F25','ENGRAM_UPDATED_HIDDEN_IS_HC_COEFFICIENT_INPUT',
       'Compose Engram output with attention HcPre readiness at target layers1/14.',
       'Both HcPre coefficients and Y consume post-Engram BF16 hidden. WKV/key-rstd can be early from lookup, but current coefficient cannot use pre-gate hidden.160source/graphfrontiers include a MEMCPY without native addresses; no pointer-elision or time gain proved.'),
      ('F02','F25','POST_GATE_BF16_COLLAPSE_NORM_HANDOFF',
       'Gate BF16 tiles feed delayed external-premix collapse and RMS while preserving updated residual publication.',
       'Keep gate BF16 round then HC Y BF16 round before RMS; skipping gate round has5vs4modelcounterexample. Retain updated R for current HCPost/coefficient/other readers. Full-H gates/norm and events/aliases/extra live buffers remain.'),
      ('F03','F25','CROSS_LAYER_HC_CHAIN_HAS_ENGRAM_BARRIER',
       'Place lookup-only WKV preparation ahead of hidden chain, then compose post-hidden rotation/gate and next HcPre.',
       'HcPost->Pre cannot bypass Engram at target layers1/14. Hidden rotation/residual reads original BF16 post, while next HC residual is updated BF16 gate output. WKV hoist does not remove work; Vector->Cube rotation/coefficient GM paths remain in fixed A3 owner domain.'),
      ('F01','F24','LOOKUP_READY_IS_NOT_UPDATED_HC_HIDDEN_READY',
       'Separate Engram lookup/TP/SP readiness from hidden/current coefficient readiness.',
       'Hash/oneDPproductionalreadyfeedsbothEngramlayers. Ready table rows enable pure WKV, not complete gate or HcPre coefficients. Respect token/lookup/row generation and all ranks; no new gain for existing shared hash/collective.'),
      ('F04','F21','ROUNDED_NORM_TO_NATIVE_DISPATCH_QUANT',
       'Compose rounded BF16 norm output with native routed dispatch quantization where ABI/quantizer contracts match.',
       'Current RMSNormCast already has BF16 plus FLOAT widened from rounded BF16. Shared branch/router may keep those outputs live. Full-H quant scale, TopK/route readiness, MC2 pad/row maps and original native quantizer/order must match; deleting public planes need not remove internal A3 GM bridges.'),
      ('F20','F22','ROUTER_SHARED_FORK_HAS_DISTINCT_TYPED_PROJECTIONS',
       'Jointly price router/TopK and shared DQ/GateUp/Down from one rounded normalized BF16 source.',
       'Router FLOAT path and shared INT8 path use different weights/precision and cannot be concatenated without a new numerical contract. Existing one norm is baseline. Source/layout/pad rows must match; finite mixed Cube/Vector/memory capacity, branch deadlines and current serial/overlap domains are priced jointly.'),
      ('F20','F23','ROUTE_WEIGHTS_AND_OUTPUT_TRANSFORM_PRECEDE_MERGE',
       'Compose router-selected weights/dispatch/combine into shared+routed merge and optional HC.',
       'Keep nonlinear unbiased selected-weight normalization, effective routed scaling/transform and BF16 output rounds. Config literal is not effective gamma. Native padded MC2 rows do not directly equal final merge M800; all row mappings, communication and old-R readers remain.'),
      ('F02','F20','DELAYED_Y_NORM_ROUTER_VS_CURRENT_COEFF_DEADLINE',
       'Advance external-premix collapsed Y and rounded norm toward router while current HC coefficients are computed separately.',
       'Y still needs actual current residual, including active Engram update. NewPre/Post/comb remain live and due by next consumers; original whole-HcPre SyncAll needs a new visibility contract. Price HF32/A2 vs router/experts mixed resource cuts, full-H RMS and rounding.'),
      ('F02','F21','HC_Y_ROUTER_DISPATCH_CHAIN_NOT_FREE_OVERLAP',
       'Analyze HC collapse/norm/router/TopK/dispatch as one subgraph with coefficient branch deadline.',
       'Keep route IDs/weights and native quant/send-ready dependencies plus rounded output contracts. Early Y is not early full MC2 until all these inputs are ready. Existing MC2 internal fusion is baseline; runtime host barriers alone are not universal dependencies.'),
      ('F02','F22','DELAYED_Y_NORM_TO_SHARED_QUANT_CHAIN',
       'Feed rounded collapse/norm tiles to shared DynamicQuant/GateUp while retaining coefficient branch.',
       'Current norm has routed/shared fanout and separate BF16/rounded FLOAT roles. Keep full-row RMS/amax, GateUp INT32, DSQ clamp/quant and Down BF16. Current SP full-weight replicas and alternative TP raw-INT32 reductions are distinct domains; no extra8xcomputecredit.'),
      ('F24','F31','HASH_REQUEST_MAPPING_REUSE_REQUIRES_EXACT_METADATA_VALUE',
       'Prepare or reuse request-index/row maps at metadata production instead of repeating pure mapping work.',
       'No same-field value is assumed from a name. Match ragged starts/current versus lookback IDs, mapped/DEAD/negative conventions, DP/SP row maps, accepted-prefix/cache generation and original write-before-fallback reads. Existing once-per-token request search/all-layer hash is baseline; new out ABI and work cost required.'),
      ('F25','F31','PURE_WKV_CACHE_AND_HIDDEN_BANK_GENERATIONS',
       'Publish immutable WKV/key-rstd banks early or by certified content-keyed memo for the current gate.',
       'Metadata must identify layer/table/storage/ownership/row/weight/numeric generations, collision-safe ID equality and capture/lastreader/lastwriter retirement. Actual current hidden still gates freshly. Equal pointer/shape or unchanged slot does not certify a WKV cache hit; native algorithms varying with M need exact-result policy.'),
      ('F02','F24','TOKEN_INPUT_SHARING_NOT_EMBEDDING_HASH_VALUE_IDENTITY',
       'Coordinate embedding/HC input preparation and target Engram ID/history lookup from token-ready inputs.',
       'Embedding raw IDs and Engram mapped ngram/history/dead semantics are different; only proven raw loads or readiness can be shared. Current first-layer HC initialization is separate from target Engram layers1/14. Preserve image/negative/dummy/validcount guards, per-layer table and all collective publication.'),
      ('F03','F04','ATTN_POST_TO_FFN_PRE_NORM_WITH_LIVE_RESIDUAL',
       'Compose attention HCPost->FFN HcPre external-premix collapse->RMSNormCast in the same decoder.',
       'Keep post BF16 then collapse Y BF16 then rounded norm BF16/FLOAT. Residual remains for current FFN HCPost/current coefficients. Collapse-read and Y-interface credit are distinct, same nextPre/one norm computed once; finite HF32/Vector/INT8 and new output visibility/sync remain.')]
    clauses=[]
    for index,(a,b,relation,claim,gates)in enumerate(rows,1):
        assert tuple(sorted([a,b]))not in previous_pairs
        clauses.append(dict(id=f'REL84-{index:02d}',a=a,b=b,relation=relation,operator_claim=claim,
            applicable_domain=gates,evidence=evidence,conditional_compatibility=True,all_runtime_gates_proved=False,
            speedup_additive=False,critical_path_gain_us=None))
    groups=[dict(id='HG84-HC-ENGRAM',members=['F01','F02','F03','F24','F25','F31'],
        contract='Lookup/rows-only WKV can precede hidden; key-rstd can precede hidden but original FP32 dot association stays. Gate BF16 updates residual before both HcPre branches. Preserve old-hidden read retirement, new hidden/current coefficients, full-H synchronization, table/weight/capture bank and cache generations.'),
      dict(id='HG84-HC-FFN-FORK',members=['F02','F03','F04','F20','F21','F22','F23'],
        contract='DelayedY collapse/norm fans to FLOAT router and BF16 shared/routed quant while current coefficients remain due. Pick consistent row/pad/quant/partition/output-owner graph; count norm/MC2/nativefusions/merge once. Joint typed capacity and true tile dependencies bound any overlap; original BF16 and effective scaling survive.')]
    for group in groups:group.update(evidence=evidence,scope_state='SCOPED_CONDITIONAL_OPERATOR_DAG_AND_RESOURCE_CONTRACT_REVIEWED',
        all_subsets_analyzed=False,legal_joint_fusion_proved=False)
    write(O/'HC-Engram-composition-clauses-v84.json',dict(clauses=clauses,higher_order_groups=groups,
        refined_existing_F24_F25_relation='REL73-20:lookup-row/weight-generation pure-WKV hoist/memo are compatible only after selecting original row/head partition and output/capture banks; table/weight/projection layers differ.',
        new_pair_clauses=15,new_higher_order_groups=2,new_maincandidates=0,overall_complete=False))
    print(dict(new_pair_clauses=15,total_scoped_pairs=84,central_pending=511,higher_order_groups=16))


if __name__=='__main__':main()
