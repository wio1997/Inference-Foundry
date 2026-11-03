"""One TP carrier cut: recorded rank namespaces and complete logical cost model."""
from pathlib import Path
from collections import Counter
from fractions import Fraction
import json, re, sqlite3
from finalize_origin_coverage_v9 import N, O, read, write
from analyze_evidence_batches_v155 import pin


def main():
    assert not(O/'carrier-resource-summary-v156.json').exists()
    provenance=read(O/'rank-device-archived-provenance-v156.json');hardware=read(O/'archived-hardware-pool-contract-v155.json')
    logs={}
    pattern=re.compile(r'\(Worker pid=(\d+)\).*world_size=(\d+) rank=(\d+) local_rank=(\d+) ')
    for log in provenance['logs']:
        for entry in log.get('selected',[]):
            match=pattern.search(entry['text'])
            if match:
                pid,world,rank,local=map(int,match.groups());assert rank not in logs;logs[rank]=dict(container_pid=pid,world_size=world,logged_local_rank=local,source_log_line=entry['line'])
    assert set(logs)==set(range(16))
    cache=N/'original-profile-cache-v53';device_info_by_trace={Path(x['relative']).parts[2]:x for x in hardware['device_info_files']}
    rows=[];sources=[];tp_groups=set();other_groups=set()
    for database in sorted(cache.rglob('*.db')):
        trace=database.parent.parent.name;match=re.fullmatch(r'dp(\d+)_pp(\d+)_tp(\d+)_dcp(\d+)_ep(\d+)_rank(\d+)_(\d+)_(\d+)_ascend_pt',trace);assert match
        dp,pp,tp,dcp,ep,trace_rank,pid,stamp=map(int,match.groups());connection=sqlite3.connect(database.as_uri()+'?mode=ro',uri=True)
        mapping=connection.execute('select rankId,deviceId from RANK_DEVICE_MAP').fetchall();npu=connection.execute('select id,name from NPU_INFO').fetchall();parallel=connection.execute("select value from META_DATA where name='parallel_group_info'").fetchall();connection.close()
        assert len(mapping)==len(npu)==len(parallel)==1
        rank,device=mapping[0];assert rank==ep and logs[rank]['container_pid']==pid and logs[rank]['world_size']==16 and logs[rank]['logged_local_rank']==tp
        info=device_info_by_trace[trace];assert info['runtime_reported_device_id']==device==npu[0][0]and int(info['archive_pid'])==pid
        profiler_path=cache/'runtime'/'official-traces'/trace/('profiler_info_%d.json'%rank);profiler=read(profiler_path);assert profiler['rank_id']==rank
        groups=json.loads(parallel[0][0]);group_records=[]
        for key,group in groups.items():
            members=group['global_ranks'];assert members[group['group_rank']]==rank
            if group['group_name']=='tp':tp_groups.add(tuple(members));assert len(members)==8
            else:other_groups.add((group['group_name'],tuple(members)))
            group_records.append(dict(communicator_key=key,**group))
        rows.append(dict(trace=trace,database=pin(database),profiler_info=pin(profiler_path),device_info=info['source'],global_EP_rank=rank,trace_DP_label=dp,trace_TP_label=tp,trace_rank_label=trace_rank,container_worker_pid=pid,logged_local_rank=logs[rank]['logged_local_rank'],log_source_line=logs[rank]['source_log_line'],exported_rank_device_id=device,captured_DeviceInfo_id=info['runtime_reported_device_id'],NPU_INFO_name_is_generic_label=npu[0][1],groups=group_records,physical_chip_or_board_binding_proved=False))
        sources.extend([database,profiler_path,Path(info['source']['path'])])
    assert len(rows)==32 and Counter(x['global_EP_rank']for x in rows)==Counter({i:2 for i in range(16)})
    assert tp_groups=={tuple(range(8)),tuple(range(8,16))}
    namespace=dict(version=156,rows=rows,profile_instances=32,worker_processes=16,TP_global_rank_groups=[list(x)for x in sorted(tp_groups)],other_recorded_group_memberships=[dict(name=name,global_ranks=list(members))for name,members in sorted(other_groups)],rank_to_exported_device_metadata_bridge_bound=True,exported_device_metadata_to_physical_chip_bridge_bound=False,local_rank_is_not_global_rank_or_physical_chip=True,container_visibility_is_identity_ordered_0_to15=True,container_visibility_and_same_number_do_not_certify_logic_to_physical_mapping=True,source='RANK_DEVICE_MAP/NPU_INFO/parallel_group_info + profiler_info + captured info.json + archivedWorkerlog;all explicit keys,not nearest-time.',new_native_operands_or_communication_task_bindings=0,overall_complete=False)
    write(O/'rank-device-namespace-bindings-v156.json',namespace)
    prior=read(O/'post-attention-projected-head-carrier-v153.json');geometries=prior['geometry'];peer_weight=4096*1024*2;delta_per_M=(4096-1024)*2
    models=[]
    for geometry in geometries:
        M=geometry['M'];P=geometry['P'];old=geometry['old_remote_only_value_bytes'];new=geometry['new_remote_only_reference_value_bytes'];weight=(P-1)*peer_weight
        assert old==(P-1)*M*8192 and new==(P-1)*M*2048 and weight==geometry['extra_weight_group_content_for_explicit_replication']
        for reuse in[1,2,4,8,16]:
            original=reuse*old;alternative=reuse*new+weight;saved=original-alternative;ratio=Fraction(alternative,original)
            models.append(dict(M=M,Q=geometry['Q'],T=geometry['T'],G=geometry['G'],P=P,same_layer_same_weight_version_calls=reuse,old_activation_offrank_delivery_bytes_total=original,new_activation_offrank_delivery_bytes_total=reuse*new,one_time_weight_delivery_bytes_per_rank=weight,new_activation_plus_weight_delivery_bytes_total=alternative,net_delivery_bytes_saved_total=saved,new_to_old_delivery_ratio=dict(numerator=ratio.numerator,denominator=ratio.denominator,value=float(ratio)),positive_logical_delivery_saving=saved>0,extra_distinct_BF16_weight_content_per_rank_per_resident_layer=weight,actual_wire_HBM_or_latency_proved=False))
    # Counterexamples and boundaries for this cost model, rather than another
    # large Cartesian sweep. Strict positivity is distinct from break-even.
    sequences=[]
    for values in[[800],[816],[834],[800,800],[800,816],[834,834],[1,1364],[1,1365],[1365],[1366],[1,1,1363],[1,1,1364]]:
        signed=7*(delta_per_M*sum(values)-peer_weight)
        sequences.append(dict(owner_M_sequence=values,sum_M=sum(values),net_offrank_delivery_bytes_saved=signed,positive_saving=signed>0,strict_break_even_integer_sum_M=1366,one_distribution_per_same_layer_same_weight_generation=True))
    weight_contract=dict(version=156,prior_operator_contract=pin(O/'post-attention-projected-head-carrier-v153.json'),candidate_ids=['F14','F15','F16'],module_ids=['M11','M23'],cut='headwiseattentionmerge/inverseRoPE -> recipient-specificWO_A -> TPheadAllToAll,comparedwithcurrentAllToAll -> recipientWO_A',model='Explicitdirectsource-recipientlogicaldeliveryreference;one8MiBBF16weightgrouppercurrentrank,distributeall8groupweightstoeachtokenowneronceperreuseepoch,thenretain.',formulas=dict(P=8,peer_weight_BF16_bytes=peer_weight,old_peer_activation_bytes='8192*M',new_peer_activation_bytes='2048*M',saved_peer_activation_bytes='6144*M',one_time_extra_weight_delivery_bytes_per_rank='7*8388608',net_saved_bytes_over_calls='7*(6144*sum(M_i)-8388608)',strict_positive_condition='sum(M_i)>4096/3;integer sum(M_i)>=1366',uniform_reuse_ratio='new/old=1/4+1024/(M*R)',uniform_minimum_positive_calls='floor(4096/(3*M))+1'),models=models,variable_M_sequence_boundaries=sequences,scope_guards=['SameM/Gcarrierdomainandallsender/receiver/headmaps,pad/bias/nonfinite/roundingandfullv153contracts.','Eachreuseepochreferstoonespecificlayerandidenticalallrecipientweightversions/layout/precision;differentlayers/generationscannotpoolR.','Weightdistributionisoneexplicitconstruction,notnecessaryforallimplementations;checkpointloading,existingreplication,streamedweights,sharedmapping,multicastandotherrepresentationsneedtheirowncompletecostgraphs.','Extra56MiB/rank/layerisdistinctBF16contentintheexplicitreplicationmodel,notactualnewallocation;onchip/allocator/workspace/lifetime/nativefitstillunbound.','Deliverybytesaresemanticper-source/per-recipientpacketcontents,nothardwarewiretraffic,information-theoreticminimum,HBMreadsorservice-timebound.','SameclassicalWO_Aworkisnotidenticaltiling/throughput;all8recipientweights/batching/extraresourcecontentionpricedbeforepredictingspeedup.'],current_M_single_call_weight_redistribution_can_outweigh_activation_delivery_saving=True,current_M_two_same_weight_calls_win_only_this_delivery_model=True,no_whole_candidate_rejection_or_approval=True,new_pair_clauses=0,new_native_legalization=0,overall_complete=False)
    write(O/'projected-carrier-weight-amortization-v156.json',weight_contract)
    boards=read(O/'archived-chip-board-mapping-v155.json')['first_mapping'];board_by_chip={x['physical_id']:x['npu_board']for x in boards}
    placement=[]
    for split in[0,2,4,6,8]:
        half=(8-split)//2;A=[2*b for b in range(split)]+[chip for b in range(split,split+half)for chip in[2*b,2*b+1]];B=sorted(set(range(16))-set(A));assert len(A)==len(B)==8
        groups=[]
        for chips in[A,B]:
            inside=sum(i!=j and board_by_chip[i]==board_by_chip[j]for i in chips for j in chips);cross=sum(board_by_chip[i]!=board_by_chip[j]for i in chips for j in chips)
            assert inside==8-split and cross==48+split
            groups.append(dict(hypothetical_rank_to_physical_chip=chips,boards_used=len({board_by_chip[x]for x in chips}),nonself_directed_pairs=56,same_board_directed_pairs=inside,cross_board_directed_pairs=cross,per_peer_activation_old_bytes='8192*M',per_peer_activation_new_bytes='2048*M',per_peer_one_time_weight_delivery_bytes=peer_weight,cross_board_total_net_saved_over_epoch='cross_board_directed_pairs*(6144*sum(M_i)-8388608)',ratio_placement_invariant_if_uniform_carriers=True))
        placement.append(dict(split_boards=split,groups=groups,is_actual_placement=False))
    write(O/'carrier-board-placement-sensitivity-v156.json',dict(version=156,conditional_domain='TwoTP8groupsoccupy16distinctphysicalchipsbijectivelyon8two-chipboards;thisbijectionisnotattestedforthearchivedruntime.',proof='Foronegroupletu=two-chipboards,s=splitboards.Then2u+s=8and0<=s<=8,sevenotherpeerpairsperrankgive56directednonselfdeliveries.Intrabboardpairs=2u=8-s,so crossboardpairs=48+swiths even. Complementgrouphasthesameu,s. Ratio(1/4+1024/(M*R))cancelsanycommonpositivedeliverymaskfactor.',placements=placement,classification_not_all_rank_permutation_enumeration=True,actual_rank_to_chip_bridge_proved=False,direct_delivery_not_native_network_wire_model=True,not_a_universal_physical_communication_lower_bound=True,no_cross_board_clock_topology_or_bandwidth_bound=True,overall_complete=False))
    summary=dict(version=156,status='PROGRESS',previous_goal_turn='PROGRESS:v155rawavailabilityandworkflowcompressionsealed;no newpairdispatch',metadata_rank_device_profile_instances=32,metadata_worker_processes=16,metadata_TP_groups=2,exported_device_to_physical_bridge_closed=False,operator_cut='TPheadAllToAll/WO_Aprojectionplacement',existing_geometry_rows=len(geometries),logical_weight_amortization_models=len(models),variable_M_boundary_examples=len(sequences),board_occupancy_classes=len(placement),no_pair_or_connected_subset_enumeration=True,uniform_single_call_new_delivery_ratio_range=[min(m['new_to_old_delivery_ratio']['value']for m in models if m['same_layer_same_weight_version_calls']==1),max(m['new_to_old_delivery_ratio']['value']for m in models if m['same_layer_same_weight_version_calls']==1)],uniform_two_call_new_delivery_ratio_range=[min(m['new_to_old_delivery_ratio']['value']for m in models if m['same_layer_same_weight_version_calls']==2),max(m['new_to_old_delivery_ratio']['value']for m in models if m['same_layer_same_weight_version_calls']==2)],strict_break_even_sum_M=1366,explicit_extra_BF16_weight_content_per_rank_per_layer=58720256,proof_domain='Declaredlogicaldeliveryconstructiononly;notactualallocation/wire/HBM/nativeparityorlatency.',new_pair_clauses=0,new_independence_proofs=0,new_native_operand_or_phase_promotions=0,new_whole_module_closures=0,new_device_work=False,model_candidates=35,framework_candidates=7,module_inventory=26,semantic_rows=358,inventory_exhaustiveness_proved=False,physical_MFU_or_lowest_established=False,overall_complete=False)
    write(O/'carrier-resource-summary-v156.json',summary)
    dependencies=[O/name for name in['rank-device-archived-provenance-v156.json','device-id-primary-concept-v156.html','device-id-primary-receipt-v156.json','archived-hardware-pool-contract-v155.json','archived-chip-board-mapping-v155.json','post-attention-projected-head-carrier-v153.json','post-attention-source-and-carrier-contract-v153.json','raw-field-availability-and-reuse-v155.json','coverage-compression-protocol-v155.json','raw-evidence-batch-postcheck-result-v155p.json','REVIEW-CHECKPOINT-v155.json','checkpoint-manifest-v155.json']]
    write(O/'carrier-resource-inputs-v156.json',dict(inputs=[pin(p)for p in sorted(set(sources+dependencies))],previous_operator_proof_not_reexecuted=True,overall_complete=False))
    print(json.dumps(summary))


if __name__=='__main__':main()
