"""Compose delayed KV publication with existing fusion and state candidates."""
from finalize_origin_coverage_v9 import O,read,write

def main():
    assert not(O/'KV-publication-composition-clauses-v87.json').exists()
    old=read(O/'candidate-relation-clauses-v86.json');pairs={frozenset((r['a'],r['b']))for r in old['clauses']}
    evidence=['KV-publication-prior-evidence-v87.json','KV-publication-source-contract-v87.json','KV-publication-native-scope-v87.json',
      'KV-publication-prefix-and-dependency-proof-v87.json','KV-publication-variants-v87.json',
      'KV-publication-value-work-models-v87.json','KV-publication-memory-page-models-v87.json',
      'KV-publication-state-plane-conditions-v87.json','KV-publication-shared-reader-topology-v87.json']
    specs=[
      ('F08','F31','CURRENT_KV_READ_VS_ACCEPTED_PREFIX_COMMIT',
       'Acceptance-delayed SWA commit requires a valid current reader route and exact retained input prefix;not a late Scatter alone.',
       'Reuse exact rejection/prefill/no-draft/row/slot/request epoch. Retained-bank overlay or PA-COW are alternatives. Compare source3Rb with retained(R+2A)b and direct-fusedRb;A is future-reader demand,not sampled output IDs or count blindly.'),
      ('F11','F31','CURRENT_INDEX_KEY_SCALE_PAIR_PUBLICATION',
       'Delayed key-cache publication must preserve matched current key-code/scale banks before current QLI reads.',
       'Last layer39 uses source20 indexK/scale and index-selection source36. Preserve preRoPE fork/quant/FP16scale policy,logical key rows/physical pages and paired generations;SWA-only commit does not cover this plane.'),
      ('F18','F31','POOL_OUTPUT_READY_IS_NOT_ACCEPTED_RING_STATE_READY',
       'Distinguish current pooled-latent/norm readiness from later accepted-state ring publication.',
       'v24 pool/update fork is preexisting. Deferring to acceptance adds old-history/current-pool/future-reader and source-bank-lifetime gates. Preserve BF16 pool output/RMS and all required completion rows;no free delay or accepted-count-only state set.'),
      ('F19','F31','PRE_ROPE_INDEX_FORK_AND_LONG_KV_COMMIT_EPOCH',
       'A transactional compressed-cache graph must preserve current preRoPE index-key fork and postRoPE longKV readers together.',
       'Layer20 source data is read by later layers before final acceptance. Keep BF16 latent/norm/RoPE boundaries,source sharing/duplicate-slot maps and code-scale/longKV generations. No substitution by final normalized hidden.'),
      ('F31','F33','COW_PAGEMAP_VS_SHARED_C1_SWA_TUPLE',
       'Page adoption can retain the existing PA attention interface but needs coherent ownership for all affected tuple-alias planes.',
       'v25 V41 descriptors/offsets/global-ID coupling retained. Raw128KiB SWA page payload is not isolated allocator cost. New map must preserve current/future C1/SWA/index/state values,pergroup metadata and last-reader/queued-writer lifetimes.'),
      ('F08','F34','SWA_PRIVATE_BANK_AND_C2_RING_GENERATION_LIFETIME',
       'Price SWA delayed publication separately from C2 ring accepted-history representation and reuse.',
       'Do not apply q-r to all ring slots without the v24/v79 history/rollback observer proof. Retain old-read-before-write and all last queued writers;private bank/page identities and ring owner/half/epoch remain separate.'),
      ('F08','F26','RAW_SNAPSHOT_CREDIT_VS_RETAINED_KV_LIVENESS',
       'Raw-HC capacity credit and delayed-KV banks compete under one actual memory budget.',
       'Under explicit unaliased40layer8192rowBF16D512 retention,absolute bank reference is320MiB,equal requested raw snapshot capacity. Neither is actual incremental/reclaimed memory. Deduct metadata/WKV/privateC2 and all peak allocations once;no double pool remainder.')]
    clauses=[]
    for i,(a,b,relation,claim,domain)in enumerate(specs,1):
        assert frozenset((a,b))not in pairs
        clauses.append(dict(id=f'REL87-{i:02d}',a=a,b=b,relation=relation,operator_claim=claim,
          applicable_domain='Conditional exact current/future operator readers,numerics,metadata,ownership,capture and state epochs. '+domain,
          evidence=evidence,conditional_compatibility=True,all_runtime_gates_proved=False,speedup_additive=False,critical_path_gain_us=None))
    groups=[dict(id='HG87-KV-CURRENT-READ-AND-COMMIT',members=['F08','F11','F13','F18','F19','F31','F33','F34'],
      contract='Current original/compressed/index data must be readable before acceptance. Retained/copiedbank overlays,existing-PA COW,on-chip retention and recomputation are different domains. Keep exact prefix/seed identity,source20 sharing,all planes and last reader+writer generations;no direct delayed-scatter rewrite.'),
      dict(id='HG87-KV-ROW-PAGE-AND-MEMORY',members=['F08','F25','F26','F34'],
      contract='GMvalue savings vs sourceScatter must be compared with direct-cache fusion. Row-bank absolute liveness,COWtuple reservation,recompute inputs,WKV and C2banks share one actual budget. No automatic latency/token capacity improvement from fewer permanent accepted writes.')]
    for g in groups:g.update(evidence=evidence,scope_state='CONDITIONAL_OPERATOR_PUBLICATION_AND_RESOURCE_DOMAIN_REVIEWED',all_subsets_analyzed=False,legal_joint_fusion_proved=False)
    write(O/'KV-publication-composition-clauses-v87.json',dict(clauses=clauses,higher_order_groups=groups,new_pairs=7,new_groups=2,
      refine_existing_relations=['REL83-01','REL73-27','REL79-01','REL83-21','REL85-09','REL85-10'],new_maincandidates=0,overall_complete=False))
    print(dict(new_pairs=7,scoped_pairs=112,higher_order_groups=23,central_pending=483,all_readers_closed=False))

if __name__=='__main__':main()
