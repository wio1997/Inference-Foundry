"""Batch field availability and graph navigation; no Cartesian candidate work."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib, json, re, struct
from finalize_origin_coverage_v9 import N, O, read, write


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
    assert not(O/'raw-evidence-batch-summary-v155.json').exists()
    inv=read(O/'raw-profile-inventory-v155.json');cache=read(O/'raw-field-cache-index-v155.json');formats=read(O/'raw-cann-field-format-index-v155.json')
    # Matching declared fields, not guesses about reserved bytes.
    hs=struct.Struct('<HHIII4Q6IQ2Id3Q6I112B');ts=struct.Struct('<HHIIIQQI55I');assert hs.size==ts.size==256
    hc=[];hc_counts=Counter();tensor=[];tensor_counts=Counter()
    for row in formats['raw_hccl_files']:
        b=Path(row['local_path']).read_bytes();assert hashlib.sha256(b).hexdigest()==row['sha256']and len(b)%256==0
        counts=Counter();headers=Counter()
        for r in hs.iter_unpack(b):
            assert r[0]==0x5a5a
            counts['records']+=1;counts['source_address_nonzero']+=r[19]!=0;counts['destination_address_nonzero']+=r[20]!=0;counts['positive_reported_size']+=r[21]>0;headers[(r[1],r[2],r[4])]+=1
        hc_counts.update(counts);hc.append(dict(source=pin(row['local_path']),relative=row['relative'],counts=dict(counts),headers=[dict(level=a,struct_type=b,data_len=c,count=v)for(a,b,c),v in headers.items()]))
    selected=[r for r in cache['files']if re.match(r'^(unaging|aging)\.additional\.tensor_info\.slice_\d+',Path(r['relative']).name)]
    assert len(selected)==80
    for row in selected:
        b=Path(row['local_path']).read_bytes();assert hashlib.sha256(b).hexdigest()==row['payload_sha256']and len(b)%256==0 and row['payload_is_full_file']
        counts=Counter();headers=Counter()
        for r in ts.iter_unpack(b):
            assert r[0]==0x5a5a and 0<=r[7]<=5
            counts['records']+=1;counts['declared_tensor_slots']+=r[7];headers[(r[1],r[2],r[4])]+=1
            for slot in range(r[7]):
                direction=r[8+11*slot];counts['input_slots'if direction==0 else'output_slots'if direction==1 else'other_direction_slots']+=1
        tensor_counts.update(counts);tensor.append(dict(source=pin(row['local_path']),relative=row['relative'],counts=dict(counts),headers=[dict(level=a,struct_type=b,data_len=c,count=v)for(a,b,c),v in headers.items()]))
    census=dict(version=155,matching_capture_CANN='9.2.0',record_endianness='little;archiveDeviceInfo.ctrl_cpu_endian_little=1',HCCL_struct='<HHIII4Q6IQ2Id3Q6I112B',tensor_struct='<HHIIIQQI55I',record_bytes=256,hccl_files=hc,hccl_counts=dict(hc_counts),tensor_files=tensor,tensor_counts=dict(tensor_counts),tensor_declared_fields=['input_or_output','format','dtype','8shape_slots'],tensor_schema_has_declared_address_stride_or_workspace_fields=False,reserved_bytes_not_reinterpreted_as_operands=True,zero_HCCL_addresses_are_unavailable_metadata_not_real_null_buffers=True,reported_size_not_physical_wire_or_last_read=True,no_native_TASK_operand_binding_promoted=True,new_device_work=False,overall_complete=False)
    write(O/'raw-cann-field-census-v155.json',census)
    info=[]
    for row in cache['files']:
        if not re.fullmatch(r'info\.json\.\d+',Path(row['relative']).name):continue
        x=read(Path(row['local_path']));assert len(x['DeviceInfo'])==1
        dev=x['DeviceInfo'][0];info.append(dict(source=pin(row['local_path']),relative=row['relative'],runtime_reported_device_id=dev['id'],archive_pid=x['pid'],cannVersion=x['cannVersion'],aic_count=dev['ai_core_num'],aiv_count=dev['aiv_num'],aic_reported_frequency=dev['aic_frequency'],aiv_reported_frequency=dev['aiv_frequency'],ctrl_cpu_endian_little=dev['ctrl_cpu_endian_little']))
    assert len(info)==32 and all(x['cannVersion']=='9.2.0'and x['aic_count']==24 and x['aiv_count']==48 for x in info)
    boards=read(O/'archived-chip-board-mapping-v155.json')
    hardware=dict(version=155,device_info_files=info,reported_device_id_counts=dict(Counter(x['runtime_reported_device_id']for x in info)),archived_board_mapping=dict(source=pin(O/'archived-chip-board-mapping-v155.json'),samples=boards['sample_count'],all_sample_mappings_equal=boards['all_sample_mappings_equal'],physical_chips=16,boards=8,chips_per_board=2,HBM_reported_MiB_per_chip=65536),host_decoder_is_matching=False,matching_decoder_files_read_from_stopped_capture_image=True,reported_clock_is_certified_fmax=False,runtime_device_to_physical_chip_bridge_proved=False,rank_to_physical_board_bridge_proved=False,exact_executed_math_mode_or_capacity_upper_proved=False,physical_MFU_or_lowest_module_established=False,new_device_work=False,overall_complete=False)
    write(O/'archived-hardware-pool-contract-v155.json',hardware)
    torch=read(O/'raw-torch-field-census-v155.json');assert torch['all_complete']and not any(str(i)in torch['tag_occurrences']for i in[5,6,7])
    memcpy_pattern=r'^(aging|unaging)\.compact\.memcpy_info\.slice_\d+'
    missing_memcpy=[z['relative']for z in inv['files']if re.match(memcpy_pattern,Path(z['relative']).name)]
    missing_torchmem=[z['relative']for z in inv['files']if Path(z['relative']).name=='torch.memory_usage']
    assert not missing_memcpy and not missing_torchmem
    availability=dict(version=155,scope='Bounded availability of named archived streams and declared matching schema fields;not absence of every possible runtime evidence source.',routes=[dict(id='TORCH_RAW_INPUT_ARGS',status='ABSENT_IN_COMPLETE_SAVED_STREAM',files=32,records=torch['records'],bytes=torch['source_bytes'],absent_tags=[5,6,7],schema_may_support_fields_but_this_capture_did_not_record_them=True,evidence='raw-torch-field-census-v155.json',repeat_same_scan_for_each_pair=False),dict(id='CANN_TENSOR_OPERANDS',status='DECLARED_METADATA_IS_FORMAT_DTYPE_SHAPE',files=80,records=tensor_counts['records'],declared_slots=tensor_counts['declared_tensor_slots'],no_declared_address_workspace_or_stride_fields=True,reserved_bytes_not_operand_proof=True,evidence='raw-cann-field-census-v155.json',repeat_same_scan_for_each_pair=False),dict(id='HCCL_SRC_DST',status='ALL_RECORDED_ADDRESS_FIELDS_ZERO_UNAVAILABLE',files=64,records=hc_counts['records'],nonzero_src=hc_counts['source_address_nonzero'],nonzero_dst=hc_counts['destination_address_nonzero'],source_bank_or_task_LAST_READ_proved=False,evidence='raw-cann-field-census-v155.json',repeat_same_scan_for_each_pair=False),dict(id='RUNTIME_MEMCPY_INFO',status='NO_MATCHING_FILE_IN_COMPLETE_PHYSICAL_INVENTORY',filename_pattern=memcpy_pattern,match_count=0,scope_is_inventory_not_runtime_nonexistence=True,memcpy_size_direction_schema_does_not_declare_src_dst=True,repeat_same_filename_search_for_each_pair=False),dict(id='TORCH_ALLOCATOR_STREAM',status='NO_TORCH_MEMORY_USAGE_FILE_IN_PHYSICAL_INVENTORY',exact_filename='torch.memory_usage',match_count=0,other_memory_sources_not_exhausted=True,repeat_same_filename_search_for_each_pair=False)],reuse_rule='Reuse this availability result only for these immutable source hashes and matching schemas;new capture/file/schema or a contradictory explicit field reopens the route.',missing_operand_inference='UNKNOWN;not independent/not zero work/not no fusion',raw_field_scan_was_shared_batch=True,native9_1352_closed=False,PMU849_phase16_assignment3136_closed=False,new_pair_clauses=0,new_independence_proofs=0,new_native_promotions=0,overall_complete=False)
    write(O/'raw-field-availability-and-reuse-v155.json',availability)
    # One record per producer cut. These are source-graph indexes, never actual
    # coexecution/identity, and no connected-subset or candidate-pair enumeration.
    edges=read(O/'operator-graph-edge-coverage-v124.json')['rows'];groups=defaultdict(list)
    for index,row in enumerate(edges):groups[(row['graph'],row['template'],row['edge']['producer'])].append(index)
    cuts=[]
    for key,indices in groups.items():
        cuts.append(dict(graph=key[0],template=key[1],producer=key[2],source_edge_rows=indices,consumers=list(dict.fromkeys(edges[i]['edge']['consumer']for i in indices)),candidate_associations=sorted({x for i in indices for field in['producer_candidate_associations','consumer_candidate_associations']for x in edges[i].get(field,[])}),same_invocation_and_all_branch_binding='UNKNOWN',native_materialization_value_alias_or_epoch_binding='UNKNOWN',grouping_is_navigation_not_semantic_equivalence=True,dispatch_as_individual_pair_task=False,proves_independence=False))
    write(O/'source-cut-navigation-v155.json',dict(version=155,source=pin(O/'operator-graph-edge-coverage-v124.json'),edge_rows=len(edges),producer_cuts=len(cuts),cuts=cuts,every_source_edge_has_exactly_one_navigation_owner=True,graph_inventory_exhaustiveness_proved=False,positive_declared_edges_are_not_all_same_invocation_DAG_edges=True,no_new_connected_subset_enumeration=True,no_candidate_pair_dispatch=True,unmapped_or_blank_scopes_remain_unknown=True,new_semantic_proofs=0,overall_complete=False))
    queue=dict(version=155,primary_unit='A conclusion-changing operator cut/effect question with exact premise guards, shared evidence and delta checks.',not_candidate_cartesian_pairs=True,work=[dict(id='Q155-GRAPH-CUTS',state='READY_FOR_TARGETED_DELTA_BINDING',action='Use actual profile TASK/API/flow bindings and existing source contracts to select concrete materialization/comm/state cuts. Bind value/epoch/observer/branch premises and cost once;do not start a fourth candidate star.',evidence=['source-cut-navigation-v155.json','raw-field-availability-and-reuse-v155.json'],remaining='Partial graph and actual operands/readers/variants remain unknown;source-edge grouping does not prove coexecution or relevance.'),dict(id='Q155-PHYSICAL-POOL',state='OPEN_EXPLICIT_BINDING_GAP',action='Resolve runtime rank/device to physical chip/board and actual executed math mode/capacity upper from already-saved provenance if available;reuse archived metadata,do not substitute observed clocks for fmax.',evidence=['archived-hardware-pool-contract-v155.json'],remaining='No certified fmax,complete math-mode/weight/binary bridge or physical MFU bound.'),dict(id='Q155-NATIVE-OPERANDS',state='OPEN_SHARED_GAP_NO_SAME_STREAM_RESCAN',action='Retain address/length/workspace/alias/reader questions as one shared obligation. Search a materially different saved artifact or a precise operator-source invariant only when it changes a named fusion/ceiling conclusion.',evidence=['raw-field-availability-and-reuse-v155.json'],remaining='Original9regimes1352executions and broader runtime operands remain unresolved;no new NPU grant.')],legacy_383_rows_are_coverage_index_not_383_tasks=True,existing_212_clauses_are_scoped_reuse_not_all_variant_proofs=True,module_inventory=26,semantic_rows=358,inventory_exhaustiveness_proved=False,whole_modules_closed=0,overall_complete=False)
    write(O/'coverage-work-queue-v155.json',queue)
    protocol=read(O/'coverage-compression-protocol-v155.json')
    for batch in protocol['active_batches']:
        if batch['id']=='B155-RAW-FIELDS':batch.update(status='SCOPED_CENSUS_COMPLETE_OPERAND_GAPS_OPEN',evidence_ready=['raw-field-availability-and-reuse-v155.json','raw-torch-field-census-v155.json','raw-cann-field-census-v155.json'],repeat_unchanged_stream_scans=False)
        elif batch['id']=='B155-HARDWARE-POOL':batch.update(status='ARCHIVED_METADATA_BOUND_PHYSICAL_BRIDGE_OPEN',evidence_ready=['archived-hardware-pool-contract-v155.json'],repeat_unchanged_metadata_checks=False)
        else:batch.update(status='READY_FOR_TARGETED_DELTA_BINDING',evidence_ready=['source-cut-navigation-v155.json','coverage-work-queue-v155.json'],new_structural_grouping_is_not_semantic_proof=True)
    protocol['work_queue']='coverage-work-queue-v155.json';write(O/'coverage-compression-protocol-v155.json',protocol)
    pointer=read(N/'CURRENT-ANALYSIS-WORKFLOW.json');pointer['protocol']=pin(O/'coverage-compression-protocol-v155.json');pointer['work_queue']=pin(O/'coverage-work-queue-v155.json');pointer['primary_queue']=[x['id']for x in queue['work']];write(N/'CURRENT-ANALYSIS-WORKFLOW.json',pointer)
    summary=dict(version=155,status='PROGRESS',workflow_compression_effective=True,scope_preserved=True,torch_files=32,torch_records=torch['records'],torch_source_bytes=torch['source_bytes'],absent_torch_input_tags=[5,6,7],hccl_files=64,hccl_records=hc_counts['records'],hccl_nonzero_addresses=0,tensor_files=80,tensor_records=tensor_counts['records'],device_info_files=32,archived_chip_board_samples=197,source_edges_indexed_once=len(edges),source_producer_navigation_cuts=len(cuts),no_repeated_pair_field_search=True,legacy_counts_not_progress_percentage=True,new_scoped_pair_clauses=0,new_independence_proofs=0,new_native_operand_or_phase_promotions=0,new_device_work=False,model_candidates=35,framework_candidates=7,module_inventory=26,semantic_rows=358,inventory_exhaustiveness_proved=False,whole_modules_closed=0,physical_MFU_or_lowest_established=False,overall_complete=False)
    write(O/'raw-evidence-batch-summary-v155.json',summary)
    paths=[O/name for name in['raw-profile-inventory-v155.json','raw-field-cache-index-v155.json','raw-record-schema-source-index-v155.json','raw-cann-schema-source-index-v155.json','raw-cann-field-format-index-v155.json','raw-decoder-availability-v155.json','archived-chip-board-mapping-v155.json','raw-evidence-local-corrections-v155.json','raw-torch-field-census-v155.json','post-draft-terminal-postcheck-result-v154p.json','checkpoint-manifest-v154.json','candidate-relation-coverage-v154.json','candidate-pair-navigation-v154.json','operator-graph-edge-coverage-v124.json']]
    paths+=[Path(z['local_path'])for z in selected]+[Path(z['local_path'])for z in formats['raw_hccl_files']]+[Path(z['source']['path'])for z in info]
    for index,key in [('raw-record-schema-source-index-v155.json','files'),('raw-cann-schema-source-index-v155.json','files'),('raw-cann-field-format-index-v155.json','schemas')]:paths+=[Path(x['local_path'])for x in read(O/index)[key]]
    paths+=[O/('raw-torch-field-census-batch%02d-v155.json'%i)for i in range(8)]
    write(O/'raw-evidence-batch-inputs-v155.json',dict(inputs=[pin(p)for p in sorted(set(paths))],reuse_old_proof_hashes_without_reexecuting_unchanged_finite_cases=True,overall_complete=False))
    print(json.dumps(summary))


if __name__=='__main__':main()
