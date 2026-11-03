"""Conditional common memory-budget cut: raw snapshot, WKV memo and private C2 state."""
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import O,read,write


def main():
    assert not(O/'tail-shared-memory-budget-v85.json').exists()
    ring=read(O/'ring-storage-capacity-models-v79.json');assert ring['pool_divisor_bytes']==540928
    C=ring['pool_divisor_bytes'];delta=read(O/'DSpark-tail-reader-closure-v85.json')['BF16_full_HC_buffer_requested_bytes']
    assert delta==335544320
    cases=[('raw_credit_only',delta,0,0,0),
        ('private32_only',0,4*393216,0,4),
        ('memo256_only',0,0,256*51296,0),
        ('raw_private32_memo256',delta,4*393216,256*51296,4),
        ('raw_private16_memo256',delta,4*196608,256*51296,4),
        ('raw_paired16_no_cache',delta,0,0,2)]
    models=[]
    for name,credit,private,cache,released in cases:
        net=credit-private-cache
        minimum=released+net//C;maximum=released+(C-1+net)//C
        models.append(dict(case=name,assumed_usable_budget_credit_bytes=credit,extra_private_ring_bytes=private,
            extra_WKV_cache_bytes=cache,other_extra_metadata_workspace_bytes_assumed=0,
            released_old_common_ID_reservations=released,pool_divisor_bytes=C,
            min_net_free_global_ID_change=minimum,max_net_free_global_ID_change=maximum,
            owner_and_cache_counts_are_illustrative=True,actual_device_budget_or_capacity_gain_proved=False))
    byname={r['case']:r for r in models}
    independent_min=sum(byname[n]['min_net_free_global_ID_change']for n in['raw_credit_only','private32_only','memo256_only'])
    independent_max=sum(byname[n]['max_net_free_global_ID_change']for n in['raw_credit_only','private32_only','memo256_only'])
    assert(independent_min,independent_max)==(596,599)
    assert(byname['raw_private32_memo256']['min_net_free_global_ID_change'],byname['raw_private32_memo256']['max_net_free_global_ID_change'])==(597,598)
    inputs=[]
    for name in['ring-storage-capacity-models-v79.json','ring-storage-source-contract-v79.json','ring-state-layout-and-lifetime-proof-v79.json','Engram-WKV-storage-and-work-models-v84.json']:
        path=O/name;blob=path.read_bytes();inputs.append(dict(path=str(path),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()))
    write(O/'tail-shared-memory-budget-v85.json',dict(input_receipts=inputs,models=models,
        formula='LetB=qC+r,0<=r<C. DeltaB is ACTUAL usable pool-budget credit, Dextra is ACTUAL incremental private/cache/hoist/protocol allocation, Srelease is removed common-ID reservation count. DeltaFree=Srelease+floor((r+DeltaB-Dextra)/C). One common budget remainder, not one per candidate.',
        illustrative_raw_credit_only_global_ID_range=[620,621],
        independent_range_if_wrongly_combining_separate_unknown_remainders=[independent_min,independent_max],
        correctly_shared_remainder_joint_range=[597,598],
        premise='Illustrations assume all335544320requestedrawbytesbecomeusablebudget,unchanged540928tupledivisor,privateownerU=S=4,nosentinel/extrametadata,256totalWKVentries128perlayerwith51200Bvalue+96BIDs. These premises are not current allocation/hit/resident-owner observations.',
        scope_limits=[
            'Tensor requested capacity is NOT a measured physical release or usable KV-budget credit; actual DeltaB remains unknown. Allocator pooling/rounding and changed graph peak memory can alter it.',
            'Scheduled max_num_seqs4 is not a bound on all resident state owners. U/S4 only illustrates a valid stated case, not the frozen workload occupancy.',
            'Dextra must count genuinely additional storage relative to baseline. Two hoisted KV planes absolute liveness is not automatically incremental allocation; old buffers may already exist or be reused.',
            'C2even compactionaloneleavesSWAanchoredtupledivisor unchanged. PrivatebankreleasesonecommonID/requestnot3;pairedstateowner/ABA/rollback/lastreaderANDwriter conditions remain.',
            'Extra cache/privatebank cost and benefit cannot be counted using the same freed bytes twice. Feasible budget also requires retained allocations/reservations fit; no token-capacity or speedup conversion.'
        ],actual_DeltaB_or_new_KV_block_count=None,overall_complete=False))
    evidence=['tail-shared-memory-budget-v85.json','DSpark-tail-reader-closure-v85.json','Engram-WKV-storage-and-work-models-v84.json','ring-storage-capacity-models-v79.json']
    rows=[
        ('F25','F26','RAW_SNAPSHOT_CREDIT_VS_PURE_WKV_CACHE_BUDGET',
            'A source-unused target raw snapshot can change the memory budget available to pure WKV memo/hoist.',
            'Use actual usable DeltaB,notrequestedcapacityasmeasuredfreeVRAM. Memo hit/equivalent-output generation still required. Cache/hoist extra capacity must be deducted before claiming KV-pool gain; each freed byte is spent once.'),
        ('F26','F34','TAIL_MEMORY_CREDIT_AND_C2_PRIVATE_STATE_SHARE_ONE_POOL_BUDGET',
            'Price raw-snapshot representation removal jointly with C2 private/paired state capacity changes.',
            'Keep unchangedSWAanchoreddivisor,onecommonID/requestandallowner/ABA/rollbackretirementgates. UseDeltaFree=Srelease+floor((r+DeltaB-Dextra)/C);rawbufferbytesandringcompactionarenotindependentKVgains. ScheduledB4isnotresidentSbound.'),
        ('F25','F34','WKV_CACHE_AND_STATE_BANK_COMPETE_FOR_LIVE_CAPACITY',
            'Jointly budget versioned WKV cache/hoist and C2 private state rather than treating each fit independently.',
            'Sum genuinely additional live storage/protocol metadata under one peak/budget. Pure WKV content equality does not imply ring-state or current hidden equality. Exact cache hit/owner occupancy/allocator and communication costs remain separate evidence.')]
    clauses=[]
    old=read(O/'candidate-relation-clauses-v84.json')['clauses'];new=read(O/'DSpark-tail-composition-clauses-v85.json')['clauses']
    known={tuple(sorted([c['a'],c['b']]))for c in old+new}
    for i,(a,b,relation,claim,gates)in enumerate(rows,9):
        assert tuple(sorted([a,b]))not in known
        clauses.append(dict(id=f'REL85-{i:02d}',a=a,b=b,relation=relation,operator_claim=claim,applicable_domain=gates,
            evidence=evidence,conditional_compatibility=True,all_runtime_gates_proved=False,speedup_additive=False,critical_path_gain_us=None))
    group=dict(id='HG85-SHARED-MEMORY-BUDGET',members=['F25','F26','F34','F31'],
        contract='Actualusablememorycreditminusgenuinelyincrementalcache/private/hoist/protocolstorageunderonecommonbudgetremainder. Preserve all live banks/epochs/owners and source numerical equality;requestedcapacity/hitrate/allocatorknowledgeare separate.',
        evidence=evidence,scope_state='CONDITIONAL_COMMON_BUDGET_CAPACITY_CUT_REVIEWED',all_subsets_analyzed=False,legal_joint_fusion_proved=False)
    write(O/'tail-shared-memory-relations-v85.json',dict(clauses=clauses,higher_order_groups=[group],new_pair_clauses=3,new_higher_order_groups=1,overall_complete=False))
    print(dict(conditional_budget_models=len(models),single_raw_credit_range=[620,621],joint_range=[597,598],new_memory_pairs=3,total_pairs=95,central_pending=500,actual_capacity_gain=False))


if __name__=='__main__':main()
