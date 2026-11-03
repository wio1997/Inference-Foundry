"""C2 common-ID capacity versus encoded Engram cache, analysis only."""
from pathlib import Path
from collections import OrderedDict
import hashlib,json,math
from finalize_origin_coverage_v9 import N,O,read,write

def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def lru(trace,K):
    cache=OrderedDict();misses=0
    for key in trace:
        if key not in cache:
            misses+=1
            if K:
                if len(cache)==K:cache.popitem(last=False)
                cache[key]=None
        elif K:cache.move_to_end(key)
    return misses

def main():
    assert not(O/'REVIEW-CHECKPOINT-v121.json').exists()
    names=['ring-storage-capacity-models-v79.json','ring-storage-source-contract-v79.json','ring-state-layout-and-lifetime-proof-v79.json',
        'tail-shared-memory-budget-v85.json','Engram-hash-UVA-and-gate-costs-v62.json','Engram-WKV-storage-and-work-models-v84.json',
        'receiver-workspace-shared-live-budget-v97.json','engram-ma<REDACTED_TOKEN>.json',
        'native-pad-consequences-v120.json']
    source=N/'state-engram-operator-source-v62/vllm_ascend/ops/triton/engram_lookup.py'
    inputs=[O/n for n in names]+[source,O/'candidate-pair-navigation-v120.json',O/'candidate-relation-clauses-v120.json']
    write(O/'row-cache-budget-inputs-v121.json',dict(inputs=[pin(p)for p in inputs],
        operator_questions=['CanC2statecompactionfundexactencodedEngramrowcacheunderthisexistingpool?',
            'HowmuchofonecommonbudgetcanrowcacheandWKVmemo jointlyconsume?',
            'Whichcachekey/epoch/retirement andlookup-demandconditionspreventdoublecredit?'],
        no_new_remote_source_or_device_work=True,overall_complete=False))
    ring=read(O/names[0]);lookup=read(O/names[4])['lookup_model'];memo=read(O/names[5])
    C=ring['pool_divisor_bytes'];assert C==540928
    R16=ring['ring16_bytes_per_owner_all_three_layers'];R32=ring['ring32_bytes_per_owner_all_three_layers'];assert(R16,R32)==(196608,393216)
    assert ring['standalone_even_compaction_pool_divisor_change']==0 and ring['standalone_even_compaction_num_blocks_change']==0
    formats={}
    for name,item in lookup['formats'].items():
        payload=item['code_bytes']+item.get('unique_scale_bytes',item.get('scale_bytes',0))
        formats[name]=dict(encoded_payload_bytes=payload,aligned_payload_bytes=((payload+31)//32)*32,
            illustrative_metadata_per_slot=64,illustrative_slot_bytes=((payload+31)//32)*32+64)
    assert {k:v['encoded_payload_bytes']for k,v in formats.items()}=={'BF16':512,'INT8_GROUP32':288,'MXFP8_GROUP32':264}
    assert memo['result_bytes_per_row']==51200 and memo['full_ID_key_bytes_per_row']==96
    W=memo['result_bytes_per_row']+memo['full_ID_key_bytes_per_row'];assert W==51296
    write(O/'row-cache-representation-contract-v121.json',dict(formats=formats,
        current_runtime_format_constexpr_attested=False,
        content='Cache exact encoded codes andallmatching group scales;leaveexistingFP32dequant→BF16round behavior unchanged. DecodedBF16cache is a different representation/footprint.',
        key='Stabletable/layer/headidentity,fullrowID,table/scale contentepoch,format/layout/shard/ownerconfiguration. Same numericID oraddress is insufficient acrosslayers/updates;C2request/pageownerIDsarenotembeddingtable/rowIDs.',
        metadata='64B/slot and32Balignment are an explicitlyselected illustrativelayout,notexistingABIorprovedminimum. Chargehash/index/state/pinning/version/event/fill/allocatoroverhead inD0whenknown.',
        demand='A counts owned row accessesaftercompleteconsumerplan;unownedfallback loads/currentzeros,otherlayers andcollectiveparticipation remainseparate. Format/tableownershipnotinferredfromshape.',
        input_profile_scope='Reusev62formatsandselectedhost-UVAoperatorrows. No newcacheimplementation,hitrate,allocation ororiginalfieldbinding.',
        source=pin(source),overall_complete=False))
    ringcases=[dict(case='current_or_even_view_only',S=4,U=4,released_common_IDs=0,ring_extra_bytes=0),
        dict(case='new_paired_owner_ABI',S=4,U=4,released_common_IDs=2,ring_extra_bytes=0),
        dict(case='private_even16_four_owners',S=4,U=4,released_common_IDs=4,ring_extra_bytes=4*R16),
        dict(case='private_full32_four_owners',S=4,U=4,released_common_IDs=4,ring_extra_bytes=4*R32)]
    models=[];capacity=[]
    for rc in ringcases:
        for fmt,f in formats.items():
            q=f['illustrative_slot_bytes'];L=rc['released_common_IDs']
            for J in [0,16,256]:
                fixed=rc['ring_extra_bytes']+J*W
                for K in [0,64,4096,16384]:
                    D=fixed+K*q;div,rem=divmod(D,C)
                    losses=(div,div+(rem>0))
                    models.append(dict(**rc,format=fmt,row_cache_slots=K,WKV_memo_entries=J,
                        assumed_additional_metadata_workspace_allocator_bytes=0,actual_raw_budget_credit_assumed=0,
                        selected_row_cache_slot_bytes=q,WKV_entry_result_and_ID_key_bytes=W,
                        extra_row_cache_bytes=K*q,extra_WKV_bytes=J*W,combined_extra_bytes=D,
                        net_free_global_ID_delta_min=L-losses[1],net_free_global_ID_delta_max=L-losses[0],
                        one_common_budget_remainder=True,actual_release_or_capacity_gain_proved=False))
                allowance=L*C-fixed
                capacity.append(dict(**rc,format=fmt,WKV_memo_entries=J,
                    selected_slot_bytes=q,additional_D0_assumed=0,
                    row_capacity_with_no_old_free_ID_loss_at_remainder0=max(-1,allowance//q),
                    row_capacity_with_no_old_free_ID_loss_at_remainderCminus1=max(-1,(allowance+C-1)//q),
                    minus_one_means_even_K0_infeasible=True,
                    unknown_real_D0_can_reduce_or_remove_this_illustrative_capacity=True))
    assert len(models)==144 and len(capacity)==36
    example=next(r for r in models if r['case']=='private_even16_four_owners'and r['format']=='INT8_GROUP32'and r['row_cache_slots']==4096 and r['WKV_memo_entries']==0)
    assert example['combined_extra_bytes']==2228224 and(example['net_free_global_ID_delta_min'],example['net_free_global_ID_delta_max'])==(-1,0)
    write(O/'row-cache-common-budget-models-v121.json',dict(models=models,no_old_free_ID_loss_capacity_references=capacity,
        budget_scope='OneNPUdevice/allocatorarenaanditsactualC2owner/table-shardconfiguration. Do notsumHBMofotherchips/ranks. Ifagrouprequirescommonblockcount,takeitsrankminimum;independentDPpoolsmustbepricedseparately.',
        C=C,private_even_ring_bytes_per_owner_three_layers=R16,private_full_ring_bytes_per_owner_three_layers=R32,
        formula='LetB=N*C+r,0<=r<C. LgenuinelyreleasedoldcommonIDs,D=D_ring+K*q+J*w+D0,rawusablecredit=DeltaB. Delta_free=L+floor((B+DeltaB-D)/C)-floor(B/C). Useonecommonr,notseparateremainderspercandidate.',
        no_raw_credit_case='DeltaB=0,D=qD*C+t. Delta_free=L-qD-I(r<t). CurrentevenviewcompactiondoesnotchangeC/NorfreeHBM;L0unlessnewlegalprivate/shared-ownerprotocolactuallyreleasesreservations.',
        no_loss_capacity='FornochangeinoldfreeglobalIDsandDeltaB0:K*q+J*w+D_ring+D0<=L*C+r. Thisboundisforselectedstorage/allocatorlayout,notallimplementationsoractualhardwarecapacity.',
        example_private_even_ring_and4096_INT8_slots=example,
        budget_premises=['Normalv79unpackedcache/nooverride,unchangedpooldivisorandotherreservations. Slotmetadata64/D00andS=U=4areillustrative,notC8residentownercounts.',
            'Do notsumthree2.268GBlogicalringviewsasindependentallocations. EvenrowviewcompactionbyitselfchangesneitherpoolCnorallocatedN.',
            'AglobalIDisnotordinaryfreeVRAM. Rowcacheownbankmustbechargedagainstarepartitionedpoolbudgetoractualindependentheadroom;currentpooladdresses/countscannotremainunchangedforfree.',
            'Rawcreditfromv85tail-snapshotproposalbelongsF26andrequiresitsownreaders/retirementproof;itdoesnotcomefromF34viewcompaction. Itiszeroinscenarioshere.',
            'ExtraWKVmemo/rawhoist/MC2workspace/cacheprotocolstorageandmigrationpeakmustsharethesameco-residentbudget. Chargeactualmax_tsum_liveD(t);newpersistentrowcacheandprivateC2co-reside.'],
        theory_state='CONDITIONAL_CEILING',actual_DeltaB_r_K_allocation_or_hardware_speedup_proved=False,overall_complete=False))
    traces=[dict(name='interleaved',trace=['a','b','c','a','b','c']),dict(name='grouped',trace=['a','a','b','b','c','c'])]
    traffic=[]
    for t in traces:
        for K in [0,1,2,3]:
            A=len(t['trace']);M=lru(t['trace'],K);I=M if K else 0
            traffic.append(dict(name=t['name'],trace=t['trace'],A=A,U=len(set(t['trace'])),K=K,cold_LRU_misses=M,hits=A-M,
                INT8_logical_host_payload=288*M,INT8_device_fill_writes=288*I,
                INT8_direct_forward_cache_reads=288*(A-M),INT8_fill_then_read_cache_reads=(288*A if K else None),
                fill_then_read_applicable=K>0,
                target_workload_measurement=False))
    assert lru(traces[0]['trace'],2)==6 and lru(traces[1]['trace'],2)==3
    write(O/'row-cache-miss-traffic-and-memo-demand-v121.json',dict(synthetic_trace_models=traffic,
        invariant='Cachecapacityaloneisnothitrate. SameA6/U3/K2has6or3missesdependingorder. Keysincludecontentepochs;changingepochcreatesnewkey.',
        traffic='Forselectedfullrowaccesspayloadp,Aownedpost-demandaccesses,Mmisses,Iinsertions:hostlogicalrequestsp*M,devicefillwrites p*I. Direct-forwardmissesfeeddequantwithoutrecacheread;fill-then-readaddsdevicecachep*Areads. Tags/probes/queues/evictions/outputIOareadditional.',
        no_wire_claim='LogicalhostrequestsarenotphysicalUVAtransactions;linecoalescing/hardwarecaches/ownership/formatandcurrentcodepathsremainunknown. DevicecacheIOisnotautomaticallyHBM.',
        WKV_memo_composition='Firstselectalllookupconsumers. AfullWKVmemohitcanremoveitslookupssolelyifallotherreaders/TPparticipantsagreeandnewready/collectiveprotocolpermits. Rowcacheisthenpricedonlyontheunionofremainingrowdemands. Do notcreditbothcachesforoneeliminatedhostread.',
        active_mask_composition='v118inactivegatevalue-onlystillneedsfeaturesunlessseparatebypasscertificateandallreadersallowotherwise. Falsemaskisnotautomaticrowcachehit/free lookup.',
        existing_hash_and_communication='v114hash/historyDEAD/commitrollbackandDP/TPglobalbanks/zeros/collectiveparticipationretainexistinggates;cachedrowdoesnotautomaticallydeleteanyofthese.',
        symbolic_service_bound='DerivenecessaryQ_sforthechosendemandandrepresentation,thenT>=max_s(Q_s/U_s)forvalidtypedcapacityupperbounds. Logicalp*Mreferencesarenotuniversallowerboundsforallfusion/summary/memoalgorithms;nohardwareU/hitrate/timecomparisonestablished.',overall_complete=False))
    write(O/'row-cache-retirement-and-numerical-contract-v121.json',dict(
        mixed_epoch_witness=dict(old_codes=2,old_scale=3,old_value=6,new_codes=4,new_scale=5,new_value=20,mixed_values=[10,12],
            conclusion='Codeandscaleloadsseparatedbyevictioncanproduceaneitherepochvalue;preserveonecommittedcontentgenerationandpinuntilallreadersretire.'),
        ring_reuse='NoC2oldreader,queuedwriter,init/zero,physicalpadwrite,capture/replayorrequestIDreusestilladdressestheretiredstorage. Asemanticdeadroworunchangedpointerisnotanallocatorrelease.',
        lifetime='Cacheinsertcodes+scalesbeforepublishingVALIDwithproperstream/deviceordering;readerspinentryandgeneration;evictioncannotreuseuntildequant/allfanout/asyncconsumersfinish. Tableupdatesinvalidatethecorrectgeneration.',
        migration='ExistingC2bit-preservinglayout/owner/pageconditionsremain. Chargeold+newbankoverlapandmigrationworkspace/read-writecontent;steady-statebudgetcannotfundearlylookupbeforethecreditexists.',
        numerical='Encodedcopiespreservebitsbutnewhit/misscodepathmustretainexistingcast,scale dtype,FP32multiply,BF16round,ownershipmask,paddingandspecialvalues. Noquantizationqualityorspeedupsubstitutionclaimed.',
        no_target_implementation=True,new_device_work=False,new_native_promotions=0,overall_complete=False))
    print(json.dumps(dict(status='ANALYZED',new_relation=('F24','F34'),budget_models=144,capacity_reference_models=36,
        illustrative_extra_bytes=example['combined_extra_bytes'],illustrative_free_ID_delta=[-1,0],
        no_actual_hitrate_or_memory_release=True,new_device_work=False)))

if __name__=='__main__':main()
