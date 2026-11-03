"""Target-tail reader closure composed with final FFN and DSpark verifier/context."""
from finalize_origin_coverage_v9 import O,read,write


def main():
    assert not(O/'DSpark-tail-composition-clauses-v85.json').exists()
    previous=read(O/'candidate-relation-clauses-v84.json')
    oldpairs={tuple(sorted([c['a'],c['b']]))for c in previous['clauses']}
    evidence=['DSpark-tail-source-index-v85.json','DSpark-tail-reader-closure-v85.json','DSpark-tail-native-boundaries-v85.json',
        'DSpark-tail-reorganization-variants-v85.json','DSpark-tail-row-local-proof-v85.json',
        'DSpark-tail-work-communication-models-v85.json','DSpark-tail-rounding-witnesses-v85.json','DSpark-tail-transfer-credit-v85.json',
        'heads-sampling-operator-contract-v65.json','index-context-operator-contract-v64.json','joint-operator-resource-cut-proof-v83.json']
    rows=[
      ('F03','F26','FINAL_POST_OUTPUT_PRIVATE_AFTER_DSPARK_RAW_READER_CLOSURE',
       'Compose final FFN HcPost BF16 output with final HC collapse and optional local RMS.',
       'Final boundary has no next HcPre/coefficient stage; only final collapse remains after scoped raw snapshot closure. Retain HcPost input residual/post/comb and final premix. Interior post outputs still have next-post/aux/Engram readers. Preserve post BF16 then collapse BF16 and native row policy; new4-stream tile ownership/ABI/capture required.'),
      ('F23','F26','LAST_MERGE_POST_COLLAPSE_NORM_WITH_DISTINCT_TRANSFER_CREDIT',
       'Join last shared+routed merge through HCPost to localized final collapse/norm.',
       'Keep effective routed scaling/mapped rows, merge BF16, post BF16, collapse BF16 and norm BF16. Snapshot,localization and private fusion are successive graph changes. Count merge plane,finalR plane and72/76MHcollapse intermediates once. OptionalDownAdd retains raw INT32 CV and all routed-ready/progress gates.'),
      ('F26','F29','TARGET_RAW_HC_IS_NOT_DSPARK_AUX_CONTEXT',
       'Remove scoped unused target raw snapshot while preserving3aux->main->perlayer contextKV pipeline.',
       'DSpark combines3aux means, not final4Hraw snapshot or target normalized head value. Preserve all three checkpoint gathers/currentselectedI and context cache publication. Current raw-HC gather role labelMTP does not imply active methodmtp. Existing contextKV reuse is baseline.'),
      ('F26','F30','TARGET_RAW_SNAPSHOT_AND_DRAFT_CONFIDENCE_RAW_HIDDEN_DIFFER',
       'Apply target raw-reader closure only to the target tail; price draft collapse/head/confidence separately.',
       'Draft confidence consumes raw draft hidden and orderedseed/input embeddings; it is not the target fullHCsnapshot or normalized target output. Removing target snapshot does not license deletion of draft raw collapse or full probability/entropy/recovery outputs. Target/draft widths/row generations are distinct.'),
      ('F26','F31','METHOD_GUARD_BUFFER_PRESENCE_AND_TAIL_PROTOCOL',
       'Select no-copy or no-buffer/localcollapse tail based on actual observer demand with coherent rank/capture generation.',
       'Pinned dspark source consumes no raw value;othermethods/getterclients retain their branches. Buffer presence also chooses communication. All participating ranks must transition together with same row/pad/trim mappings;capture/output banks and native numerical variants remain explicit. Keep allnormalized targetrows until a separate all-reader/Iproof.'),
      ('F27','F35','ACCEPTANCE_SELECTOR_FEEDS_AUX_CONTEXT_NOT_RAW_FINAL_HC',
       'Connect target verify/acceptance output to ordered3aux selection and context main projection.',
       'prepare_inputs(_padded) I is produced after sampling/rejection. Preserve acceptedcounts/rowduplicates/bounds/positions and raw/probability/RNG/sentinel branches; no unconditional earlyI. Source actual selection remains before mainGEMM, while late aux gather needs its own communication/liveness cost. Raw target snapshot is not this argument.'),
      ('F28','F30','DRAFT_INITIALIZATION_TO_RAW_CONFIDENCE_AND_NORMALIZED_HEAD_FORK',
       'Compose draft embedding/HC initialization/decoder/final collapse with the two draft head consumers.',
       'Preserve row/pad/onehot and all decoder queries. Final raw collapsed draft hidden feeds confidence; rounded norm feeds base head. Target raw snapshot closure does not make either draft branch dead. Current head rows already selected; Markov recurrence still needs each preceding chosenID.'),
      ('F29','F35','ACCEPTED_CONTEXT_KV_PUBLICATION_AND_NEXT_QUERY_GENERATION',
       'Match target accepted/selected rows and positions to next DSpark contextKV publication.',
       'Keep per-layer group/slot maps,duplicate-slot write order,accepted-prefix rollback and allfuture readers. Stableaddress or acceptedcount alone is not equalcontextvalue;contextKVprecompute is already shared across draft decode/Markov. Preserve sourcefullwrite and all-rank protocol; no deletion of future query/context work from head prefix pruning.')]
    clauses=[]
    for i,(a,b,relation,claim,gates)in enumerate(rows,1):
        assert tuple(sorted([a,b]))not in oldpairs
        clauses.append(dict(id=f'REL85-{i:02d}',a=a,b=b,relation=relation,operator_claim=claim,applicable_domain=gates,
            evidence=evidence,conditional_compatibility=True,all_runtime_gates_proved=False,speedup_additive=False,critical_path_gain_us=None))
    groups=[dict(id='HG85-FINAL-HC-HEAD',members=['F03','F23','F26','F31','F35'],
        contract='FixedDSpark raw-HCvalueclosure permits distinguishing final private output from interior live residual. ChooseG0S/G1/G2/G3;preserveBF16merge/post/collapse/norm,postinputresidual/coefficients and fullnormalized/head/probability outputs. Source-derived320MiBcapacity and ring/privatebytebudgets are separate from actualmemory/latency.'),
      dict(id='HG85-VERIFY-CONTEXT-DRAFT',members=['F27','F28','F29','F30','F31','F35'],
        contract='Acceptance/row selector follows target head/verification;threeaux checkpointmeans generate context,while draft raw/normalized hidden fork toconfidence/basehead. Keep seed/query/context row namespaces,KVgroup/slot/acceptedprefix epochs and allobserved fullwrite/RNG/probability outputs. No target-raw-to-aux substitution or free crossiteration cache.')]
    for group in groups:group.update(evidence=evidence,scope_state='SCOPED_OPERATOR_READER_AND_REORGANIZATION_CONSTRAINTS_REVIEWED',
        all_subsets_analyzed=False,legal_joint_fusion_proved=False)
    write(O/'DSpark-tail-composition-clauses-v85.json',dict(clauses=clauses,higher_order_groups=groups,
        new_pair_clauses=8,new_higher_order_groups=2,
        refine_existing_relations=['REL73-21:F26/F35 normalized-row reader frontier remains open;rawHCsourceobserver closed only in fixedDSpark.',
            'REL74-08:F26/F27 distinct final andcheckpointgenerations;threeaux are the live DSpark argument.',
            'REL74-07:F27/F29 context main/norm/KV inputs remain;targetsnapshot elision is independent of context partition.'],
        new_maincandidates=0,overall_complete=False))
    print(dict(new_pairs=8,scoped_pairs=92,higher_order_groups=18,central_pending=503,raw_source_reader_closure=True,native_fusion_proved=False))


if __name__=='__main__':main()
