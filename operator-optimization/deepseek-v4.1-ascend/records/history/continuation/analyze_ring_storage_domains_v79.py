"""C2 operator state-storage repartition; offline and conditional,not implementation.

Separates descriptor allocation,shared block-ID reservations,and live state.
Only necessary frozen operator/storage interfaces are inspected.
"""
from pathlib import Path
from collections import Counter
import ast,hashlib,json,math,zipfile
from finalize_origin_coverage_v9 import N,O,W,read,write


def main():
    assert not(O/'ring-storage-source-contract-v79.json').exists()
    sroot=W/'frozen-source-snapshot';archive=W/'frozen-sources.zip'
    receipt=read(W/'source-collection-receipt.json')
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['archive_sha256']
    rels=['vllm-ascend/vllm_ascend/models/deepseek_v41/cache_config.py',
          'vllm-ascend/vllm_ascend/core/kv_cache_interface.py',
          'vllm-ascend/vllm_ascend/models/deepseek_v41/compressor.py',
          'vllm-ascend/vllm_ascend/models/deepseek_v41/model.py',
          'vllm/vllm/v1/core/kv_cache_coordinator.py',
          'vllm/vllm/v1/core/single_type_kv_cache_manager.py']
    pins=[];texts={}
    with zipfile.ZipFile(archive)as z:
        for rel in rels:
            p=sroot/rel;b=p.read_bytes();assert b==z.read(rel)
            h=hashlib.sha256(b).hexdigest();pins.append(dict(path=str(p),sha256=h,original_archive_member=rel,
                binding='Byte-identical member of hash-verified original frozen-source archive;not loadedliveobject/nativebinary attestation.'))
            texts[Path(rel).name]=b.decode()
    cfgrow=read(O/'shared-operator-numerical-and-library-contract-v57.json')['model_text_config'];cfgp=Path(cfgrow['path'])
    assert hashlib.sha256(cfgp.read_bytes()).hexdigest()==cfgrow['sha256']
    pins.append(dict(path=str(cfgp),sha256=cfgrow['sha256'],binding='Reused exact savedmodelJSON;parsedconfiguration isnotliveallocator/stateowner snapshot.'))
    cfg=read(cfgp)['text_config'];sources=cfg['kv_source_layer_ids'];ratios=cfg['compress_ratios']
    assert sources==[2,8,14,20]and[ratios[i]for i in sources]==[2,2,2,1]
    width=cfg['head_dim'];assert width==512 and cfg['index_head_dim']==128 and cfg['sliding_window']==128
    cc=texts['cache_config.py'];co=texts['kv_cache_coordinator.py'];mgr=texts['single_type_kv_cache_manager.py'];core=texts['kv_cache_interface.py']
    assert 'return sum(page_sizes)'in cc and 'available_memory // sum(page_sizes)'in cc
    assert 'block_pool=self.block_pool'in co and'self.block_pool = BlockPool('in co
    tree=ast.parse(mgr);circ=next(c for c in tree.body if isinstance(c,ast.ClassDef)and c.name=='CircularBufferManager')
    cs=ast.get_source_segment(mgr,circ)
    assert 'self.block_pool.get_new_blocks(1)'in cs and 'return 0 if self.req_to_blocks.get(request_id) else 1'in cs
    assert 'def remove_skipped_blocks'in cs and 'def cache_blocks'in cs
    assert 'STATE_RING_ROWS = 32'in cc and 'scale_dim: int = 0'in core
    regs=[r for r in read(O/'operator-shape-ceiling-ledger-v55.json')['observed_regimes']if r['operator']in ['_pool_kernel','_cache_update_kernel']]
    direct=[r for r in regs if r['origin']=='DIRECT'];graph=[r for r in regs if r['origin']=='GRAPH']
    assert len(direct)==6 and sum(r['executions']for r in direct)==144
    for r in direct:assert '17306,32,1024'in r['signature']['inputShapes_text']+r['signature']['outputShapes_text']
    assert sum(r['executions']for r in graph)==480
    write(O/'ring-storage-source-contract-v79.json',dict(
        source_files=pins,original_archive=dict(path=str(archive),sha256=receipt['archive_sha256'],receipt=str(W/'source-collection-receipt.json')),
        owner_topology=dict(kv_source_layers=sources,source_ratios=[ratios[i]for i in sources],C2_ring_layers=[i for i in sources if ratios[i]==2],
            number_of_ring_objects=3,source_layers_with_ratio2_in_backbone=sum(x==2 for x in ratios[:cfg['num_hidden_layers']]),
            all_ratio2_consumers_are_not_independent_ring_objects=True),
        global_pool_unit=dict(normal_source_contract='One global block-ID pool is passed to all cache-group managers. V41 has one descriptor/backing per physicaltuple;each globalID corresponds to a page position in every descriptor.',
            ring_group='All three CircularBufferSpec objects are grouped together. CircularBufferManager claims one shared globalID per resident request,not oneID perlayer.',
            scheduled_B_is_not_resident_owner_bound=True,saved_max_num_seqs=4,actual_resident_owner_upper_bound=None,
            excludes='KVPP/offload/other private-pool substitutions,actualruntimegroup identity/overrides/native ABI are not attested by these normal-source clauses.'),
        observed_state_views=[dict(regime_id=r['regime_id'],operator=r['operator'],executions=r['executions'],signature=r['signature'])for r in direct],
        direct_pool_calls=72,direct_update_calls=72,direct_state_logical_blocks=17306,graph_pool_update_records=480,
        graph_state_N_or_actual_pointer_attested=False,
        source_claim='Detaching state removes a reservation in the common global-ID pool;it doesnot release three independently allocated logical ringarrays. max_num_seqs4doesnotboundallpending/resumable stateowners.',
        new_remote_source_read=False,new_source_files_obtained=0,no_model_Torch_NPU_capture_service_or_process_work=True,overall_complete=False))

    # Declared unpacked A3/p128 spec domain,consistent with observed BF16/INT8 planes.
    pages=[]
    for layer,ratio in zip(sources,[2,2,2,1]):
        storage=128//ratio;kv=storage*512*2;index=storage*(128+2);swa=128*512*2
        ring=32*1024*4 if ratio==2 else 0
        pages.append(dict(source_layer=layer,ratio=ratio,storage_rows=storage,long_KV_bytes=kv,index_INT8_plus_FP16_scale_bytes=index,
            SWA_alias_bytes=swa,ring_alias_bytes=ring,capacity=max(kv+index,swa,ring),capacity_after_even_compaction=max(kv+index,swa,ring//2),
            capacity_after_state_detachment=max(kv+index,swa)))
    divisor=sum(x['capacity']for x in pages)
    assert [x['capacity']for x in pages]==[131072]*3+[147712]and divisor==540928
    assert all(x['capacity']==x['capacity_after_even_compaction']==x['capacity_after_state_detachment']for x in pages)
    models=[]
    for U in [1,2,4,8,16]:
        for slots in [32,16]:
            per_owner=3*slots*1024*4
            for sentinel in [0,1]:
                payload=(U+sentinel)*per_owner
                minloss=payload//divisor;maxloss=math.ceil(payload/divisor)
                for S in sorted({0,1,U//2,U}):
                    models.append(dict(bank_owner_capacity_U=U,current_resident_owners_S=S,ring_rows=slots,reserved_invalid_slot=sentinel,
                        state_payload_bytes=payload,metadata_alignment_or_transport_bytes_not_included=True,
                        normal_pool_divisor_bytes=divisor,min_global_ID_capacity_loss=minloss,max_global_ID_capacity_loss=maxloss,
                        min_net_free_global_ID_change=S-maxloss,max_net_free_global_ID_change=S-minloss,
                        prerequisites='BudgetB>=payload,no override,unchanged othergroups and ownercountS;unknownbudget remainder ranges0..divisor-1. FixedUbank reservespayload even ifS<U. Metadata/additionalalignment canincreasecapacityloss.'))
    old_logical=17306*32*1024*4
    write(O/'ring-storage-capacity-models-v79.json',dict(
        domain='UnpackedA3BF16long/SWA,INT8index+oneFP16scale/row,block128,no foldedcache,source layers2/8/14/20. NormalV41fourtuples/nooverride. Not a liveallocator snapshot or packed950domain.',
        tuple_pages=pages,pool_divisor_bytes=divisor,ring32_bytes_per_owner_all_three_layers=393216,ring16_bytes_per_owner_all_three_layers=196608,
        standalone_even_compaction_pool_divisor_change=0,standalone_even_compaction_num_blocks_change=0,
        direct_typed_logical_ring_view_bytes_per_object=old_logical,three_logical_ring_views_are_additive_independent_allocations=False,
        formula=dict(old='N0=floor(B/C);ringgroup reservesS commonIDs,sofree0=N0−other_used−S.',
            new='D=(U+z)*L*R+metadata;N1=floor((B−D)/C);privatebankremovesringgroupreservations,sofree1=N1−other_used.',
            difference='Δfree=S−(N0−N1)<=S;D>=0,Cunchanged,nooverride. ScheduledB4 isnotresidentS norrequiredU.',
            rounding='ForD=qC+t,0<=t<C,loss=q orq+1dependingbudgetremainder(if t0loss=q). Metadata/rounding canerase smalladvantages.',
            meaning='NetfreeglobalIDcapacity,notthroughput,notindividualKVtokens,andnotGiB VRAMreturned. Fixedtotalbudgetpoolbytesplusnewbank remainwithinroundingofbudget.'),
        private_bank_models=models,
        paired_shared_page_model=dict(owners_per_ID=2,conditional_compact_ring_bytes_per_owner_per_tuple=65536,first_three_tuple_page_bytes=131072,
            necessary_common_IDs='atleastceil(S/2) iftwoowners/ID andnootherreplicas;minachievableonlywithlegalpacking/relocation/lifetimes.',
            maximum_released_IDs='S−ceil(S/2)=floor(S/2),beforemetadataworkspacebudgetrounding.',
            no_compulsory_extra_full_private_bank=True,actual_fragmentation_or_relocation_cost_known=False,
            issue='Onehalf ownercannotfree/reassignwholeIDwhileotherhalflive. Currentexclusiveone-requestmanager andpglobal-onlymetadata neednew shared-owner/halfslot/refcount ABI.'),
        migration=dict(private32_live_owner_value_read_plus_write_bytes='2*S*393216',private16_or_pack16_live_even_value_read_plus_write_bytes='2*S*196608',
            cold_new_request_initialization_not_automatically_zero_or_free=True,actual_migration_required_or_cost_measured=False),
        no_projection_GEMM_FLOP_reduction=True,state_allocation_size_is_not_per_call_traffic=True,
        theoretical_state='SYMBOLIC_CEILING',actual_native_memory_bandwidth_or_operator_speedup=None,overall_complete=False))
    # Compact layout preserves the old32row EVEN observational quotient.
    slots=[]
    for half in [0,1]:
        for residue in range(0,32,2):
            slots.append(dict(owner_half=half,old_even_row=residue,new_physical_row=half*16+residue//2))
    assert len({x['new_physical_row']for x in slots})==32 and min(x['new_physical_row']for x in slots)==0 and max(x['new_physical_row']for x in slots)==31
    for t in range(0,100000,2):
        assert (t%32)//2==(t//2)%16
    # Value-only stable-address reuse has an ABA counterexample.
    trace=[dict(step=0,owner='requestA',slot=1,epoch=7,event='enqueue old reader/update;not completed'),
           dict(step=1,owner='requestB',slot=1,epoch=8,event='prematurely reassign sameaddress;writeBstate'),
           dict(step=2,owner='requestA',slot=1,epoch=7,event='old queued write landsinBstate;corruption')]
    write(O/'ring-state-layout-and-lifetime-proof-v79.json',dict(
        compact_slot_table=slots,integer_identity='Forvalidnonnegativeevenpositiont,(tmod32)/2=(t//2)mod16. Differenthalf namespaces mapto disjoint16rowintervals.',
        arithmetic_checks=50000,finite_checks_are_not_native_or_formal_solver_tests=True,
        observational_gate='Onlythealreadydeclared retained-even/history/rollback domain fromF34v24/v62;odd/fullbyte/transfer/debugobservers preventthisquotientwithoutfurtherproof. Do notinfer16fromratioalone.',
        source_kernel_compatibility='Currentkernelrequires32rows/globalp with0invalid. Compactprivate16needsnewABI. Private32canpreservelogical32rowvalues butstillneedsowner indirection;reuse0invalidbyreservingoneextraownerpageornew explicitvaliditycontract.',
        ABA_counterexample=trace,
        required_lifetime_contract=['Keylayer/rank/request identity,privateowner slot,half ifpacked,andgeneration;addposition/history/acceptedprefix/parameterstateasrequiredbyoperator.',
            'Publishsourcevalues+metadata+epoch beforeeachreader oncorrectstream/rank;stableaddress/shape isnotvaluegeneration.',
            'Retire onlyafterlastreaderANDlastqueuedwriter;staleepochdroppingalonecannotpreserve anoldreader stillneedingAvalues. Publishnewowner afterdrainorusedoublebuffer.',
            'PackedIDremainsreserveduntilallhalflifetimesdrain;requestfreealonecannotreturnwholeIDwhileotherownerlive.',
            'Preservepoololdread-before-update WAR;F18earlynorm/F19index/longcachepublication andlastreaders areseparatecompositions.'],
        variants=[dict(id='F34-INPLACE-EVEN',domain='Existing32rowtuplelayout;skiponlyunobservedoddwrites.',pool_capacity_gain=False),
            dict(id='F34-INPLACE-16',domain='Existingtuple/SWAanchor;newcompactkernelbutunchangedsharedpoolcapacity.',pool_capacity_gain=False),
            dict(id='F34-PRIVATE-32',domain='Detachall32rowlogicalstateperresidentowner into separatelybudgetedstablebank;preserveoddvalues.',pool_capacity_gain='PARAMETRIC_RESERVATION_VS_BANK_COST'),
            dict(id='F34-PRIVATE-16',domain='Detachandcompactretained-evenstate;newbank/metadata/nativekernel/capturecontracts.',pool_capacity_gain='PARAMETRIC_RESERVATION_VS_BANK_COST'),
            dict(id='F34-PAIR-16-IN-SHARED-ID',domain='Packtwoeven-stateowners ineach32rowpageofallthreeC2tuples;newhalfownershiplifetimeABI.',pool_capacity_gain='AT_MOST_FLOOR_S_OVER_2_BEFORE_METADATA_AND_FRAGMENTATION')],
        alternatives_not_additive=True,actual_native_or_performance_validation=False,whole_candidate_closed=False,overall_complete=False))
    print(dict(source_files=7,source_ring_objects=3,source_ratio2_consumers=18,shared_pool_divisor=divisor,
        shared_reservations_per_resident_request=1,direct_state_N=17306,logical_view_bytes=old_logical,bank_model_rows=len(models),
        packing_slots=32,variants=5,all_actual_pointer_owner_bounds_known=False,status='BOUNDED_OPERATOR_ANALYSIS_READY'))


if __name__=='__main__':main()
