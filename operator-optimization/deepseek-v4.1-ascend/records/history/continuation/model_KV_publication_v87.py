"""Compare explicit KV publication graphs; distinguish GM value bytes from physical traffic."""
from math import ceil
from finalize_origin_coverage_v9 import O,read,write

def value_case(rows,accepted,width=512):
    assert 0<=accepted<=rows
    b=2*width
    old=3*rows*b
    retained=(rows+2*accepted)*b
    copied=(3*rows+2*accepted)*b
    direct=rows*b
    return dict(R_valid_distinct_rows=rows,A_required_retained_rows=accepted,D=width,BF16_row_bytes=b,
      source_producer_bank_plus_scatter_value_bytes=old,
      retained_bank_overlay_plus_commit_value_bytes=retained,
      copied_bank_overlay_plus_commit_value_bytes=copied,
      fused_direct_producer_to_cache_value_bytes=direct,
      retained_overlay_credit_vs_source_value_bytes=old-retained,
      copied_overlay_extra_vs_source_value_bytes=copied-old,
      retained_overlay_extra_vs_fused_direct_value_bytes=retained-direct,
      copied_overlay_extra_vs_fused_direct_value_bytes=copied-direct,
      baseline_persistent_payload_write_bytes=rows*b,overlay_required_commit_payload_write_bytes=accepted*b,
      recompute_accepted_KV_BF16_dense_ops=2*accepted*5120*width,
      recompute_RMS_ordinary_ops_reference=accepted*(4*width+1),recompute_RMS_rsqrt_reference=accepted,
      recompute_RoPE64_scalar_ops_reference=accepted*3*64,
      actual_traffic_capacity_or_speedup=None)

def main():
    assert not(O/'KV-publication-variants-v87.json').exists()
    evidence=['KV-publication-source-contract-v87.json','KV-publication-native-scope-v87.json',
      'KV-publication-prefix-and-dependency-proof-v87.json','KV-publication-prior-evidence-v87.json']
    conditions=['Exact all current/future KV,compressed,indexer,aux,external/transfer/debug observers and original row/request/position/group/slot generations.',
      'Preserve cache BF16 casting/normalization/RoPE and attention reduction/softmax/sink/order semantics. New storage/layout/kernel paths do not inherit native parity automatically.',
      'Actual accepted-prefix and additional future-reader row demand,including no-draft/prefill/padded/invalid/duplicate-slot cases.',
      'All data/maps/metadata/aliases published before reads,old readers and queued writers retired before reuse;native capture addresses and CP/TP epochs coherent.',
      'Compare one complete changed graph against both source Scatter and conditional direct-producer-to-cache fusion;GM logical bytes,physical HBM/cache traffic and time are distinct.']
    variants=[
      dict(id='KV87-DELAY-WITHOUT-CURRENT-READ-PATH',decision='EXCLUDED_AS_UNCONDITIONAL_REWRITE',
        graph='Move existing cache scatter after acceptance,leave current cache-only attention unchanged.',
        reason='Current verify needs candidate KV before acceptance;stale-value witness and dependency cycle distinguish it.'),
      dict(id='KV87-RETAIN-ROTATED-GM-BANK',decision='CONDITIONAL',
        graph='Retain already-produced rounded rotated KV bank;current attention reads old-cache plus current-bank overlay;commit required A rows after acceptance.',
        resource='Can remove2(R-A)b Scatter/commit reference bytes against source bank+Scatter,while extending bank lifetime. Against direct fused cache store,extra2Ab remains in this fixed GM domain.'),
      dict(id='KV87-COPY-TO-PRIVATE-GM-BANK',decision='CONDITIONAL_WITH_ADDITIONAL_VALUE_TRAFFIC',
        graph='Copy current rotated KV into a separate private bank,read through overlay,then commit A rows.',
        resource='Copy costs2Rb and commit2Ab. Relative to source bank+Scatter this adds2Ab;do not claim rejected-write savings without copy cost.'),
      dict(id='KV87-COW-PAGES-IN-EXISTING-PA-POOL',decision='CONDITIONAL_SEPARATE_REPRESENTATION',
        graph='Current KV writes go to owned private PA pages within existing cache pool;current block tables select them;adopt required pages with logical prefix publication.',
        resource='Existing attention cache argument can be retained. All R current KV stores still occur;old visible prefix copy,tuple-alias coherence,allocation/lifetime/maps add costs. Rejected suffix bytes need not be erased if unobservable;this is not A-only physical write pruning.'),
      dict(id='KV87-RETAIN-UNTIL-ACCEPT-ONCHIP',decision='CONDITIONAL_NEW_STORAGE_AND_SCHEDULE_DOMAIN',
        graph='Keep necessary current KV in resident on-chip storage until dependent acceptance,then publish A.',
        resource='Must price cross-layer retained state and other live work versus actual tier capacity. No assumption that normal separate kernels retain arbitrary on-chip banks across launches;tile/request pipelining may change peak.'),
      dict(id='KV87-RECOMPUTE-AFTER-ACCEPT',decision='CONDITIONAL_COMPUTE_STORAGE_TRADEOFF',
        graph='Current attention consumes transient current KV;after acceptance recompute required A KV from preserved per-layer inputs or another exact replay route,then commit.',
        resource='Retaining normalized5120D inputs costs10times512D KV values in the explicit same-row BF16 bank model. Recompute adds dense projections,norm/RoPE,possibly weight service and new-shape numeric proof;no free discard of all inputs.')]
    for v in variants:v.update(evidence=evidence,conditions=conditions,implementation=False,native_test=False,critical_path_gain_us=None)
    write(O/'KV-publication-variants-v87.json',dict(variants=variants,
      no_new_maincandidate=True,all_runtime_conditions_closed=False,overall_complete=False))
    models=[value_case(24,a)for a in[0,4,12,24]]+[value_case(6,a)for a in[1,2,6]]
    write(O/'KV-publication-value-work-models-v87.json',dict(models=models,
      scope='Fixed unpacked BF16 D512,valid distinct current writes R and required retained subset A,one ordinary producer bank write/R read/store per scatter or copy. Invalid slots/multiple writers/extra readers need their own model. R24 is an illustrative all-valid24row case,not actual valid-slot attestation.',
      common_reads='Current attention reads of selected KV values are held semantically equal across comparisons;actual read multiplicity/cache locality/overlay metadata are not inferred. Norm/preRoPE forks,quantized/padded layouts,tuple copies and other value planes are excluded and retained separately.',
      formulas=dict(source='3Rb',retained_overlay='(R+2A)b',copied_overlay='(3R+2A)b',fused_direct='Rb',
        retained_credit_vs_source='2(R-A)b',retained_extra_vs_direct='2Ab',copied_extra_vs_source='2Ab'),
      baseline_warning='A GM-byte argument can favor retained overlay over source Scatter yet fail against fused direct cache output. No runtime dominance follows:scatter fragmentation/coalescing,cache tiers,reader schedule/overlap and hardware capacities differ.',
      resource_bound='With necessary work/traffic in a stated representation and valid capacity uppers,use max of weighted shared-resource cuts and true per-request/tile dependency bounds. Bytesalone do not imply latency or MFU.',
      recompute_domain='Conventional BF16 GEMM MAC=2ops,5120→512;RMS and64-tail RoPE ordinary scalar references. Native arithmetic modes/tiling/rounding and weight residency remain unbound.',
      actual_native_equivalence_or_HBM_traffic_or_speedup_proved=False,overall_complete=False))
    memory=[]
    for cap in[24,8192]:
        for layers in[1,40]:
            memory.append(dict(N_physical_bank_capacity=cap,retained_layers=layers,D=512,
              unaliased_KV_BF16_bank_reference_bytes=layers*cap*512*2,
              normalized_hidden_BF16_input_bank_reference_bytes=layers*cap*5120*2,
              input_HC4_BF16_bank_reference_bytes=layers*cap*4*5120*2,
              actual_incremental_allocation_or_usable_budget_cost=None))
    page=[]
    for start in[0,1,122,123,127]:
        q=6;count=ceil((start+q)/128)
        page.append(dict(start_offset_in_128row_page=start,scheduled_current_rows=q,
          touched_128row_pages=count,raw_page_payload_bytes=count*128*512*2,
          one_old_partial_page_present=(start>0),full_old_partial_page_copy_rw_reference_bytes=(2*128*512*2 if start>0 else 0),
          visible_prefix_rows_copy_payload_reference_bytes=start*512*2,
          actual_allocator_block_cost_or_incremental_copy_measured=False))
    write(O/'KV-publication-memory-page-models-v87.json',dict(row_bank_models=memory,page_examples=page,
      four_request_six_row_example=dict(current_rows=24,row_bank_bytes=24576,
        independent_touched_private_pages_range=[4,8],raw_128row_page_payload_range_bytes=[524288,1048576],
        worst_four_full_leading_page_copy_rw_reference_bytes=1048576),
      capacity_credit_example=dict(raw_snapshot_requested_bytes=335544320,
        forty_layer_8192x512_BF16_row_bank_reference_bytes=335544320,
        remainder_if_both_assumed_fully_incremental_and_fully_reclaimed_bytes=0,
        actual_free_memory_or_usable_pool_credit_proved=False),
      liveness_domain='Explicit unaliased row banks retained across1or40layers until dependent acceptance,no early commit/recompute/compression. Reference absolute liveness,not universal minimum or actual incremental allocation. Per-request/tile schedules can lower peak;existing graph buffers may already cover some banks.',
      page_domain='Illustrative append-only disjoint logical128row SWA pages and512BF16 latent width,within existing PA pool. Private pages can preserve visible prefix and adopt whole pages with rejected suffix logically masked. Only prefix values demanded by observers need copying;full-page copy is a reference,not necessary universal work.',
      tuple_alias_scope='v25 V41 tuple backing/globalblock-ID allocation couples SWA/state/longKV/index aliases. Raw128KiB page payload is NOT independently allocatable cost. A changed shared map may require coherent contents for every affected plane;do not swap only one SWA view or multiply overlapping allocations as independent bytes.',
      one_shared_budget='Raw-snapshot actual usable credit minus genuinely additional overlay/COW/recompute/WKV/private-C2/metadata/workspace allocations uses one common pool remainder. No double spending. Actual DeltaB and each Dextra remain unknown.',
      actual_token_capacity_or_latency_gain_proved=False,overall_complete=False))
    write(O/'KV-publication-state-plane-conditions-v87.json',dict(
      SWA='Current attention consumes current original KV;retained-bank overlay or PA-COW must preserve exact current/future data generation.',
      compressed_long_KV='Current following layers may share source compressed values. Deferring persistent commit still needs all current readers to see exact compressed data with original position/visibility.',
      index_key_and_scale='Current QLI reads preRoPE-derived key code and scale banks. Multi-plane overlay/commit must preserve matched code-scale generations and source quantization;SWA-only delay is not enough.',
      C2_ring='Priorv24 pool/update fork is known. Delaying state until acceptance is stronger:preserve old ring read-before-write,current pooled outputs and all rollback-observable history/last queued writers. A single accepted count does not prove the needed ring-position set.',
      draft_context='DSpark3aux→selectedcontext→perlayerKV is a different value/epoch namespace and already selects its source rows. No target-SWA-to-draft-context substitution or repeated selection credit.',
      explicit_unproved=['Actual valid slots/duplicates/A and all future observers','Native new storage/layout/attention numerical parity',
        'COW map coherence across V41 tuple aliases/streams/ranks','Cache readiness and last-reader+last-writer lifetimes',
        'Real tier capacities/incremental liveness/critical-path value'],overall_complete=False))
    print(dict(variants=6,byte_work_models=7,row_bank_models=4,page_examples=5,
      retained_overlay_vs_direct_extra='2Ab',actual_physical_speedup=False))

if __name__=='__main__':main()
