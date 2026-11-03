"""Reconcile finite adjacent guard scope against saved profiles and bounded source."""
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import hashlib
import json

from finalize_origin_coverage_v9 import N, O, read, write
from finalize_Engram_whole_frontier_v30 import packet, graph


def main():
    assert not (O/'REVIEW-CHECKPOINT-v43.json').exists()
    stamp = datetime.now(timezone.utc).isoformat()
    prev = read(O/'M21-adjacent-guard-crosswalk-v42.json')
    remaining = prev['remaining_specific_guard_crosswalk']
    assert len(remaining) == 36
    source = read(O/'M21-remaining-guard-point-source-v43.json')
    for f in source['files']:
        assert hashlib.sha256(Path(f['path']).read_bytes()).hexdigest() == f['sha256']
    reuse = read(O/'M21-reused-point-source-receipts-v43.json')
    assert len(reuse['receipts']) == 4
    # Notes are keyed to the original exact guard site, not inferred from method coverage.
    cp_notes = {
        175: 'Iterate only this layer four cache-plane prefixes, not every runner metadata entry: the runner map also contains draft native DSA metadata without global_metadata.',
        181: 'Skip None source prefixes; every remaining plane resolves its own global_metadata before _get_layer_metadata chooses role-specific query/cache views.',
        199: 'No long-context role returns no compressed sparse indices. This does not remove SWA/global cache writes or output communication.',
        201: 'A non-index-source CP consumer slices stable shared topk by qr.shape[0], not full pre-CP hidden count. Preserve local-query row interval and source-before-consumer generation.',
        212: 'Pad local attention rows to per_rank with zeros before TP head restore/AllToAll. No-query ranks still participate; caller requires output_rows<=per_rank.',
        219: 'Restore an additional zero suffix to hidden_states extent when DP/FlashComm padding exceeds exchanged rows, then project into fixed caller output. Keep padded layout, including bias/reduction observers.',
    }
    dsa_notes = {
        247: 'If K<topk, append -1; otherwise retain all K, including K>topk. Then unsqueeze head dimension, contiguous, signed INT32. No truncation, zero sentinel or arbitrary cast substitution.',
        300: 'Packed SWA writes kind=win via flat slots truncated to value rows; fallback scatter uses coordinate slots. Actual cache alias/layout equivalence is not inferred from equal shapes.',
        315: 'Main-stream cache publication iff packed backend exists AND (prefill_count>0 OR decode_count!=num_reqs). Unknown/mixed prefill classification cannot qualify as pure decode.',
        316: 'The OR handles both explicit prefill and any not-pure-decode count; it is not equivalent to prefill_count alone.',
        395: 'Prepare query/weights only for long-context index-source roles. project_query consumes qr; project_weights consumes correctly sliced hidden on aux stream; apply query RoPE before compression scratch reuse and join weights event.',
        414: 'Quantize only if prepared exists, quantized_query absent, floating query present and nonempty, and max_cache_seq_len>0. Otherwise retain prepared values and let the correct reader/empty-cache branch decide.',
        422: 'False should-quantize guard returns without publishing a new quantize_done; true path waits main start on aux, computes query+scale and records completion. Consumer must wait that actual event, not a same-shape old event.',
        434: 'Base output path zero-pads to hidden extent then performs full projection; current observed CP override has separate pre-AllToAll and post-exchange suffix guards, already reviewed.',
        452: 'share_quant requires caller share_inputs (query_metadata is cache_metadata), exact quant-method type equality and communication-flag equality. This is existing source policy, not a general mathematical equivalence proof for different quantizer parameters.',
        459: 'Shared branch reuses q_quant/q_scale; separate branch publishes q-quant-ready then computes cache quant on aux. Current draft local-query/global-cache rows differ, so whole-input alias reuse is not inferred.',
        471: 'When separate cache quantization records kv_quant_done, main stream waits it; shared branch has no such event. Both converge to staged Q/KV matmul schedule with existing vector/Cube overlaps.',
        489: 'If cache write is not designated main-stream, publish SWA on aux after KV norm/RoPE. Current main.wait_stream(aux) after Q_b joins all aux effects.',
        493: 'If cache write is designated main-stream, publish after the aux join. Exactly one selected stream writes the cache; never drop this branch or double-write under a fused producer.',
        512: 'C1 uses BF16 compressor/norm and source query RoPE; other source branch uses FP32 hidden cast plus two projections and C2 pool/state. Preserve rounding, pool/ring state, independent index/long slots and old-latent read before long in-place RoPE.',
        524: 'Wait COMPRESSOR metadata group when group id is present, before projected pool/state uses it. No group id does not prove same bank generation or permission to bypass unrelated waits.',
        535: 'If either C2 source cos OR sin is None, fetch both full-table values for c2_source_positions and select this layer, then slice to positions count. Skip-ring does not imply cos=1/sin=0 or dead callback.',
        553: 'Long cache packed branch writes kind=cmp via flat slots; fallback scatters long_slots after indexer old-latent consumption and in-place long RoPE. Do not merge with SWA kind=win without physical-owner proof.',
        569: 'No long-context role returns None after its pure hidden row slice. Existing SWA path still runs. For other roles use correct base/CP local row view.',
        572: 'Non-index-source base branch returns stable shared topk active rows, without re-running QLI. CP override instead takes qr rows. Reuse is within the correct source/consumer forward generation.',
        578: 'Candidate filtering plus packed indexer backend appends matching source folded cache. Nonpacked candidate filtering still reaches native QLI with block-ID candidates; false here does not disable the role flag globally.',
        582: 'Index-source branch requires prepared_indexer; non-index-source/short-context early returns do not. A fusion must not evaluate query/weight fields before their guarded producer.',
        583: 'Wait quantize_done if actually recorded; otherwise select_projected receives the floating/optional quantized fields and follows its own early-empty or fallback-quantization branches.',
        599: 'candidate_lengths remains None or active-row view of its own persistent bank. Do not synthesize counts from a different row space or assume optional plane absent from profile silence.',
        601: 'topk_lengths remains None or its own active-row view. Empty-cache source sets lengths0; packed attention may read lengths, independently of index IDs/QLI metadata.',
        605: 'If selected width is zero, clear stable topk active rows to -1. Distinguish zero width, zero query rows and zero cache length: select_projected with indices_output returns that fixed-width bank in its zero-cache branch.',
        607: 'Otherwise nonpacked path copy_(selected) publishes stable topk; packed path expects its backend to write the supplied destination. No redundant-copy claim without actual return-alias/full-write/version proof.',
        609: 'Candidate-source copies candidate_out into shared block-ID bank. These are block IDs with candidate_block_size, not token positions transformed by prepare_indexer_indices. Non-source preserves the existing candidate generation.',
    }
    rows = []
    for r in remaining:
        line = int(r['id'].split(':')[1])
        cp = r['id'].startswith('c0087fcd72ee7f73')
        note = (cp_notes if cp else dsa_notes)[line]
        refs = ['M21-remaining-guard-point-source-v43.json', 'M21-reused-point-source-receipts-v43.json']
        if line in [300,512,524,535,553]:
            refs += ['F33-consumer-and-numerical-contract-v27.json','M21-C2-metadata-fill-and-existing-fusion-contract-v39.json']
        if cp or line == 434:
            refs += ['M19-current-forward-patterns-and-dispatch-v35.json','M19-embedding-HC-CP-row-and-consumer-contract-v35.json']
        if line in [569,572,578,582,583,599,601,605,607,609]:
            refs += ['M21-QLI-plan-direct-consumer-source-v40.json']
        if line in [452,459,471,489,493]:
            refs += ['profile-raised-source-contract-v16.json','F33-profile-order-contract-v27.json']
        rows.append(dict(r, line=line, semantics=note, evidence=refs,
                         state='EXACT_GUARD_DATAFLOW_AND_CONDITIONAL_DOMAINS_RECONCILED',
                         all_actual_variant_flags_bound=False, actual_alias_and_epoch_proved=False))
    all_rows = deepcopy(prev['guard_dataflow_reconciled'])+rows
    assert len(all_rows) == len({r['id'] for r in all_rows}) == 63
    write(O/'M21-adjacent-guard-crosswalk-v43.json', dict(
        generated_utc=stamp, original_S13_static_sites=63, newly_reconciled_guard_sites=36,
        prior_reconciled_guard_sites=27, guard_dataflow_reconciled=all_rows,
        remaining_specific_guard_crosswalk=[],
        scope='Finite 63 static guard dataflows and alternatives reconciled; this is not live all-flag/alias/parameter-bank/native reader proof or whole M07-M11/M19/M21 closure.',
        actual_all_runtime_variants_observed=False, whole_module_closed=False, overall_complete=False))

    facts = [
        dict(grade='DOCUMENTED_PLUS_MEASURED', statement='Reused current native v35 direct CP input/output,19 no-query global-cache patterns and v27 distinct stream overlaps. New finite role/optional-plane/stream-publish guards do not replace those execution counts or create another functional resource bucket.'),
        dict(grade='DOCUMENTED', statement='Global metadata resolver visits only four role-specific cache planes, avoiding unrelated native draft entries. CP non-index-source returns indices for local qr rows; base returns its local _indexer_hidden_states rows. Matching full hidden batch size is not the query row contract.'),
        dict(grade='DOCUMENTED', statement='Index query preparation retains qr projection, hidden-state weight projection, RoPE and event join. Query quantization uses a five-term guard; source cache update waits C2 metadata if required, then indexer selection waits actual quantize_done.'),
        dict(grade='DOCUMENTED', statement='Legacy select_projected early no-cache branch fills the provided indices bank -1, zeroes topk_lengths and candidate-source lengths, and creates -1 candidate blocks. Its nonempty fallback already passes indices_output and lengths_output into prepare_indexer_indices; direct output is baseline, not a new unconditional optimization.'),
        dict(grade='DOCUMENTED', statement='Shared state reset returns None without clearing. Source layers must overwrite the active rows before reuse within the forward. Stable addresses are not content-generation equality across iterations or different accepted/rejected histories.'),
        dict(grade='DERIVED_FROM_PINNED_SOURCE', statement='Normal same-config construction allocates topk bank [max_tokens,config.index_topk] INT32 and wires it into every layer shared state; topology.index_topk=int(config.index_topk). A leading row slice is contiguous at full width. If unchanged, compressed_indices has K==topk, so pad_sparse_indices does not pad, unsqueeze is a view, and contiguous/int may preserve storage. Do not count three hypothetical device kernels. This does not attest arbitrary post-construction replacement/capture banks.'),
        dict(grade='DOCUMENTED_PLUS_CONDITIONAL', statement='Existing quantization sharing is guarded by metadata identity plus quant-method type/communication flags. Partial local-query/global-cache reuse would require row-equivalent quantization with identical scale/parameter/communication semantics and an exact row map; source method type equality alone is not proof. Current positive Q_A/KV primitives are unquantized F.linear; no actual quantize-kernel savings are counted.'),
        dict(grade='DOCUMENTED', statement='Packed prefill/mixed SWA cache write is on main after stream join; other selected cases write on aux before join. Norm/RoPE/output/cache events must preserve exactly one publication and previously observed overlaps. A reduction in launches does not prove lower request latency.'),
    ]
    write(O/'M21-neighbor-guard-and-index-publication-contract-v43.json', dict(
        generated_utc=stamp, facts=facts,
        reused=['F33-consumer-and-numerical-contract-v27.json','F33-profile-order-contract-v27.json',
                'M19-embedding-HC-CP-row-and-consumer-contract-v35.json','M21-QLI-plan-direct-consumer-source-v40.json'],
        not_new_gains=['existing shared quant branch','existing direct index postprocess destination',
                       'normal same-width INT32 contiguous TopK formatting views','no-op shared-state reset',
                       'existing stream overlap and branch-specific cache publication'],
        remaining_predicate_or_implementation_gaps=['Exact CVLinear.quantize wrapper and prepare_indexer_indices return-alias contract absent from local snapshot/archive; no remote retry or claim of self-copy elimination.',
            'Actual optional-length/role/weight/bank aliases, live mutation and captured readers remain conditions for altered variants.'],
        implementation=False, native_validation=False, critical_path_gain_us=None, whole_module_closed=False, overall_complete=False))

    # Pure row-wise counterexample and restricted identity, not a model quantizer test.
    def row_quant(row, scale=None):
        bound = max(map(abs,row)) if scale is None else scale
        if not bound:
            return [0]*len(row), Fraction(1)
        return [round(Fraction(127*x,bound)) for x in row], Fraction(bound,127)
    h = [[1,2],[100,0],[-3,9],[0,0]]
    cases = 0
    for indices in [[],[0],[2,0],[1,1,0],[3,2,0,1]]:
        a = [row_quant(h[i]) for i in indices]
        full = [row_quant(r) for r in h]
        b = [full[i] for i in indices]
        assert a == b
        cases += 1
    local_batch = row_quant(h[0],2)
    global_batch = row_quant(h[0],100)
    assert local_batch != global_batch
    pad_cases = 0
    examples = []
    for tokens in [0,1,3]:
        for width in [0,1,2,5]:
            for topk in [1,2,4]:
                x = [[(-1 if c%3==0 else r*10+c) for c in range(width)] for r in range(tokens)]
                result = [[row+[-1]*max(topk-width,0)] for row in x]
                assert all(len(v)==1 and len(v[0])==max(width,topk) for v in result)
                if width >= topk:
                    assert [v[0]for v in result] == x
                pad_cases += 1
                if tokens == 1 and width in [0,5] and topk == 2:
                    examples.append(dict(input=x,topk=topk,result=result))
    write(O/'F06-F12-row-reuse-and-index-format-domain-review-v43.json',dict(
        generated_utc=stamp, grade='DERIVED', row_local_quantizer_examples=cases,
        quant_identity='For a deterministic identical row-local quantizer, including its row scale, Q(S_pi H)=S_pi Q(H). This fails for batch-dependent scales or differing quantizer parameters.',
        batch_scale_counterexample=dict(row=h[0],local_batch_scale=2,global_batch_scale=100,
                                       local_quantized=local_batch[0],global_quantized=global_batch[0]),
        actual_current_quantizer_contract_verified=False,
        index_format_cases=pad_cases, index_format_examples=examples,
        index_format_domain='Preserve -1, shape[T,1,max(K,topk)], INT32 and original order. K>topk is retained, never truncated. Normal fixed same-width INT32 bank has no required padding/conversion kernel.',
        self_copy_elision='Conditional only if producer returns the same destination range, fully writes it, and no storage/version/debug/captured observer needs copy_. Exact helper return implementation is not in local frozen sources.',
        measured_kernel_or_latency_savings=None, implementation=False, native_validation=False, whole_module_closed=False, overall_complete=False))

    variants = [
        dict(id='F06-v43-PARTIAL-ROW-QUANT-REUSE',parent='F06',related=['F32'],
             transformation='On nontrivial equivalent row-local quantizer paths only, compute common query/cache rows once and route matching quantized rows plus scales without forcing full-global readiness before local Q.',
             current_case_gain_claim=False, current_quantizer_and_saved_work_proved=False),
        dict(id='F12-v43-STABLE-INDEX-PUBLISH-COPY',parent='F12',related=['F31'],
             transformation='Elide only a separately proved redundant stable-index publication copy, preserving exact postprocess values, lengths, candidate bank generation and every observer. Current direct-output postprocess is baseline; helper return alias remains unverified.',
             current_case_gain_claim=False, actual_redundant_copy_proved=False),
    ]
    conditions = [
        ('ROWS','Base/CP local indexed-hidden and qr row maps, physical PAD and valid extents remain exact; do not substitute full global hidden rows.'),
        ('SOURCE_GENERATION','Shared index/candidate/length banks are valid only after the matching source layer publication within this forward. reset is no-op; cross-iteration pointer equality is insufficient.'),
        ('ROLE','Short-context, index-source, candidate-source and candidate-filter roles keep their distinct branch domains and folded-cache ownership.'),
        ('QUANT_FUNCTION','Identical row-local quant function/scales/parameters and communication semantics are required; class equality or same shape alone is insufficient.'),
        ('NOOP_BASELINE','Unquantized/identity quantizers and already direct output/format views have no assumed device work to eliminate.'),
        ('READINESS','Partial quant reuse must preserve local Q early readiness, original main/aux events and useful vector/Cube overlap.'),
        ('INDEX_VALUES','Preserve current prepare_indexer_indices position/compression mapping; do not replace block-ID candidates with token positions or skip opaque mapping.'),
        ('FORMAT','Preserve -1 padding, K>topk nontruncation, INT32, leading-row slice/layout and stable shared-reader ABI.'),
        ('LENGTHS','Candidate lengths/topK lengths are separate optional banks; preserve max-cache0 clearing and all active/inactive readers.'),
        ('COPY_ALIAS','Copy removal requires exact same range, full prior write, no conflicting alias and no version/debug/captured observer.'),
        ('C2_WAR','C2 metadata readiness and index old-latent read precede long in-place RoPE; preserve source cos/sin fallback and compression precision.'),
        ('CACHE_STREAM','Packed mixed/prefill main publication versus aux publication selection remains exact; preserve join and exactly one cache writer.'),
        ('CAPTURE','Keep captured bank addresses/strides and input-ready/last-reader/reuse across graph and iteration boundaries; actual body audit remains M22.'),
        ('COST','No savings from source-only copies/formatting or logical bytes without positive actual work and critical-path analysis; overlapping variants are not additive.'),
    ]
    for v in variants:
        v.update(implemented=False,native_validation=False,critical_path_gain_us=None)
    write(O/'M21-neighbor-conditional-variants-v43.json',dict(
        generated_utc=stamp,variants=variants,conditions=[dict(id='M21-v43-'+k,requirement=v,actual_current_legality_proved=False)for k,v in conditions],
        new_model_candidates=[],implementation=False,whole_module_closed=False,overall_complete=False))
    g = graph(['current_forward_source_generation','prepared_query_and_weights','actual_quantize_done','index_select_and_prepare','indices_lengths_candidates_publication','same_forward_reuse_readers','SMLA_query_reads','all_last_readers','next_forward_overwrite'],[
        ('current_forward_source_generation','prepared_query_and_weights','CORRECT_LOCAL_ROWS_AND_ROLE'),
        ('prepared_query_and_weights','actual_quantize_done','GUARDED_QUANT_PRODUCER_OR_EXISTING_FIELDS'),
        ('actual_quantize_done','index_select_and_prepare','WAIT_IF_EVENT_RECORDED'),
        ('index_select_and_prepare','indices_lengths_candidates_publication','PRESERVE_ALL_OUTPUT_PLANES'),
        ('indices_lengths_candidates_publication','same_forward_reuse_readers','SOURCE_BEFORE_CONSUMER_GENERATION'),
        ('same_forward_reuse_readers','SMLA_query_reads','LOCAL_ROW_COUNT_AND_FIXED_FORMAT'),
        ('SMLA_query_reads','all_last_readers','ALIAS_CAPTURE_OTHER_OBSERVERS'),
        ('all_last_readers','next_forward_overwrite','REUSE_FENCE_NOT_POINTER_CSE')])
    write(O/'M21-neighbor-publication-graph-v43.json',dict(generated_utc=stamp,conditional_graph=g,graphs_acyclic=True,
        note='Data/ordering obligations, not measured causal timing or complete actual aliases.',implementation=False,whole_module_closed=False,overall_complete=False))

    sem = deepcopy(read(O/'M21-semantic-branch-inventory-v42.json'))
    evidence = ['M21-adjacent-guard-crosswalk-v43.json','M21-neighbor-guard-and-index-publication-contract-v43.json',
                'F06-F12-row-reuse-and-index-format-domain-review-v43.json','M21-neighbor-conditional-variants-v43.json']
    for row in sem['static_branch_sites']:
        if row['semantic_family']=='M21-S13':
            row.update(audit_state='EXACT_GUARD_DATAFLOW_AND_CONDITIONAL_DOMAINS_RECONCILED',
                       review_status='POINT_SOURCE_GUARD_RECONCILED',semantic_evidence=evidence,
                       runtime_binding='CASE_CONDITIONS_NOT_ALL_LIVE_FLAGS_OR_ALIASES_ATTESTED')
    for b in sem['branches']:
        if b['id']=='M21-S13':
            b.update(state='ALL_63_GUARD_DATAFLOWS_RECONCILED;RUNTIME_BANK_GENERATIONS_IN_S15',
                     gap='Finite adjacent63guard source cases/consumers/protocols are now locally analyzed. Actual captured/all-reader/alias epochs remain S15/M22; exact external quantizer/helper return alias is a condition for optional copy/quant variants, not evidence of current savings.',
                     evidence=list(dict.fromkeys(b['evidence']+evidence)))
        if b['id']in ['M21-S07','M21-S08','M21-S12','M21-S15']:
            b['evidence']=list(dict.fromkeys(b['evidence']+evidence[:2]))
    sem.update(generated_utc=stamp,parent_inventory='M21-semantic-branch-inventory-v42.json')
    assert len(sem['branches'])==18 and len(sem['static_branch_sites'])==268
    write(O/'M21-semantic-branch-inventory-v43.json',sem)
    audit=deepcopy(read(O/'M21-original-requirement-completion-audit-v42.json'))
    audit.update(generated_utc=stamp,parent_audit='M21-original-requirement-completion-audit-v42.json',result='NOT_COMPLETE',
        prioritized_next=['M22 actual main graph/QLI/parameter banks/all readers/16 phase-unbound graph binding using full saved native result, not compact receipts',
                          'S16 same-denominator resource bounds and S18 full-process transport coverage',
                          'Exact dispatch/PCP predicates once authentication restored; other local finite modules may proceed meanwhile'])
    for row in audit['requirements']:
        row['evidence']=list(dict.fromkeys(row['evidence']+evidence+['M21-semantic-branch-inventory-v43.json']))
        row['completion_proved']=False
        if row['requirement']=='all_relevant_branches':
            row['note']='S13 all63 finite guard dataflows reconciled; S01/S14 exact predicate mappings and S15 graph/body/bank bindings remain. Conditional implementation is not required to record opportunity.'
    write(O/'M21-original-requirement-completion-audit-v43.json',audit)

    current=packet('saved-graph-resources166-v14')
    assert current['summary']['graph_executions']==80 and current['summary']['graph_phase_counts']['PHASE_UNBOUND']==16
    old=packet('profile-vector-graph-blocks166-v10-v3')
    assert old['summary']['graph_executions']==11
    write(O/'M22-saved-profile-local-payload-availability-v43.json',dict(
        generated_utc=stamp,current_receipt=str(N/'saved-graph-resources166-v14.json'),current_payload_keys=list(current),
        current_full_result=current['full_result'],current_full_result_sha256=current['result_sha256'],
        current_summary=current['summary'],current_full_80_graph_bodies_in_this_receipt=False,
        old169_receipt=str(N/'profile-vector-graph-blocks166-v10-v3.json'),old169_graphs=11,
        old169_first_graph_samples_do_not_prove_current166_bank_bindings=True,
        other_local_files_not_exhaustively_searched=True,
        next_saved_data_question='Use full pinned graph-blocks-PMU-and-MC2-v14 and exact graph/native relation sidecars: enumerate actual QLI/SMLA/producer waits, graph definition/bank/capture modes and retained16unbound phases without ordinal or nearest-time pairing.',
        authentication_failure_retained='M21-predicates166-v42.json rc255; no new SSH login or retry',
        whole_module_closed=False,overall_complete=False))
    print(json.dumps(dict(new_exact_guards=36,total_S13_exact_guards=63,remaining_S13_guard_crosswalk=0,
        semantic_families=18,static_sites=268,row_quant_math_examples=cases,index_format_cases=pad_cases,
        conditional_variants=len(variants),conditions=len(conditions),normal_index_padding_gains_not_counted=True,
        whole_module_closed=False,overall_complete=False)))


if __name__=='__main__':
    main()
