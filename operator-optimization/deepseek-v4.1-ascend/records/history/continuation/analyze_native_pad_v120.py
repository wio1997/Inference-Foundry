"""Audit the one approved Pad capture and decode only rendered own-graph metadata."""
from pathlib import Path
import hashlib,json,re,struct
from finalize_origin_coverage_v9 import N,O,read,write
from decode_memset_metadata_v111 import decode_record,inline_flowtable,write_spans

def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def main():
    assert not(O/'REVIEW-CHECKPOINT-v120.json').exists()
    folder=N/'native-pad-probe-results-v120b';own=read(folder/'probe-result.json');dump=read(folder/'own-native-pad-graph.json')
    failed=read(N/'native-pad-probe166-v120.json')['result'];remote=read(N/'native-pad-probe166-v120b.json')['result']
    assert failed['capture_attempts']==0 and failed['status']=='GUARD_STOPPED'
    assert remote['capture_attempts']==1 and remote['attempt']['returncode']==0 and remote['status']=='ONE_PROBE_ATTEMPT_FINISHED'
    assert own['status']=='CAPTURED_METADATA_QUERIED_NO_REPLAY'and len(own['tasks'])==len(dump)==6
    assert remote['container_guard']==remote['container_after']
    for field in ['resource_guard','resource_after']:
        assert all('No running processes found in NPU '+str(i)in remote[field]['stdout']for i in range(8))
    assert not(N/'ssh166-pad-v120.sock').exists()
    assert own['plan']['own_application_input_output_bytes']==44
    assert [t['seq_id']for t in own['tasks']]==list(range(6))
    assert [t['get_params_returncode']for t in own['tasks']]==[0,0,0,0,0,107000]
    indexed=read(O/'pad-provider-discovered-index-v111.json')['files']+read(O/'pad-provider-padmeta-index-v111.json')['files']
    mpath=Path(next(r['local_path']for r in indexed if r['path'].endswith('MemSet_1a6864193b99ef93ef38616f04a712ab_high_performance.json')))
    ppath=Path(next(r['local_path']for r in indexed if r['path'].endswith('PadV3_f4660b375c589d51c1f0e17818182091_high_performance.json')))
    meta=read(mpath);pad=read(ppath)
    items=[]
    for t,g in zip(own['tasks'],dump):
        words=[int(x,16)for x in re.findall(r'0x[0-9a-fA-F]+',g.get('args',{}).get('Kernel Args',''))]
        items.append(dict(seq=t['seq_id'],name=g['name'],type=t['type'],get_params_returncode=t['get_params_returncode'],
            numBlocks=t.get('numBlocks'),argsSize=t.get('argsSize'),opInfoSize=t.get('opInfoSize'),isHostArgs=t.get('isHostArgs'),rendered_words=words,
            args_address=t.get('args_address')))
    ms=next(r for r in items if r['name']==meta['kernelName']);ps=next(r for r in items if r['name']==pad['kernelName'])
    assert(ms['numBlocks'],ps['numBlocks'])==(48,1)
    words=ms['rendered_words'];assert len(words)==19 and ms['argsSize']==152
    raw=struct.pack('<19Q',*words);nptr=len(meta['parameters']);assert nptr==3
    expected=nptr*8+meta['opParaSize'];assert expected==128
    try:inline_flowtable(words,ms['args_address'],ms['argsSize'],meta)
    except ValueError as e:old_error=str(e)
    else:raise AssertionError('Do not claim the old whole-packet hypothesis validated')
    offsets={w-ms['args_address']for w in words[:nptr]if ms['args_address']+24<=w and w+96<=ms['args_address']+ms['argsSize']and(w-ms['args_address'])%8==0}
    assert offsets=={24};off=offsets.pop();record=decode_record(raw[off:off+96]);spans=write_spans(record,False)
    assert record['need_core_num']==1 and record['core_num']==48 and record['last_elems']==8 and record['last_tail_bursts']==1
    assert spans['union']==[[0,32]]and write_spans(record,True)['union']==[[0,32]]
    assert ms['rendered_words'][0]==ps['rendered_words'][5]
    assert ms['rendered_words'][0]not in[own['bool_input_address'],own['bool_output_address']]
    assert items[0]['rendered_words'][0]==own['bool_input_address']and items[4]['rendered_words'][1]==own['bool_output_address']
    assert pad['workspace']==dict(num=1,size=[1536],type=[0])and pad['parameters'][4]==dict(dtype='float32',init_value=0.0)
    sources=read(O/'pad-provider-source-contract-v111.json')['source'];sourcepaths=[Path(r['path'])for r in sources]
    inputs=[O/'native-pad-authorization-v119.json',O/'native-pad-probe-plan-v111.json',O/'native-pad-probe-provider-controls-v111.json',
        N/'run_native_pad_probe166_v120.py',N/'run_native_pad_probe166_v120b.py',N/'diagnostic-native-pad-probe-v111/inspect_pad_graph.py',
        N/'native-pad-probe166-v120.json',N/'native-pad-probe166-v120b.json',N/'native-pad-submission-v120.json',N/'native-pad-submission-v120b.json',
        O/'native-pad-probe-result-index-v120.json',O/'native-pad-probe-result-index-v120b.json',folder/'probe-result.json',folder/'own-native-pad-graph.json',
        mpath,ppath,N/'decode_memset_metadata_v111.py']+sourcepaths
    write(O/'native-pad-inputs-v120.json',dict(inputs=[pin(p)for p in inputs],new_device_work=True,scope='OneapprovedownPadcapture,metadataonly,noModel/noReplay/noGPUderef.',overall_complete=False))
    write(O/'native-pad-observations-v120.json',dict(status=own['status'],capture_attempts=1,guard_only_stops=1,
        guard_correction='InitialwrapperfilteredchipnumberinsteadphysicalcardandstoppedbeforeCDLL/Torch/NPU. CorrectedwrapperselectsPhyIDs0/1withfreshidleandmapguards;sameapprovedprobeSHA.',
        application_IO_bytes=44,elapsed_process_wall_seconds=remote['elapsed_seconds'],container_PID_unchanged=True,all8cards_no_process_before_after=True,
        streams=len({t['stream_handle']for t in own['tasks']}),tasks=items,
        kernel_GetParams_successes=5,default_GetParams_107000=1,
        selected_MemSet_kernel=ms['name'],selected_Pad_kernel=ps['name'],
        selected_provider_disk_controls_checked_before_library_device=True,
        debug_task_ts_dur_not_execution_time=True,no_tensor_data_D2H_or_opaque_GPU_argument_deref=True,
        graph_replays=0,model_loads=0,SSH_closed=True,overall_complete=False))
    write(O/'native-pad-packet-and-store-contract-v120.json',dict(
        argsSize=152,rendered_uint64_words=19,metadata_opParaSize=meta['opParaSize'],metadata_pointer_slots=3,
        old_whole_packet_prediction_bytes=expected,old_strict_helper_rejection=old_error,
        new_extraction='Uniquealready-renderedprefixpointerwithinreportedpacketbounds:argword1==args_address+24. Decodeonly96source-declaredbytesatthatobservedoffset. This doesnotvalidateoldtotal-packetpacking.',
        flowtable_offset=off,decoded_source_record=record,source_requested_store_spans=spans,
        remaining_bytes_after_old_predicted_end=raw[expected:].hex(),remaining24bytes_semantics='UNKNOWN;currentlyzero,notcalledpaddingorfields.',
        MemSet_destination_word=ms['rendered_words'][0],same_word_in_Pad_index=5,
        workspace_role='OwnMemSetfirstworkspaceinputmatchesPadworkspaceposition(afterintercoreprefix),pinnedmetadataoneFLOAT32zeroedworkspaceandTIKsync_workspace;distinctfrombothapplicationBOOLbuffers.',
        advertised_workspace_metadata_bytes=1536,actual_allocator_extent_or_HBM_bytes_measured=False,
        source_compatible_requested_zero_store_bytes=32,
        critical_distinction='MemSetlaunch48 !=need_core_num1. Source1writer*8FLOAT32elements=32B;1536isadvertisedworkspacecapacity,notprovedallocationorwrittenextent. Captureisnotreplayandstorecontentwasnotreadback.',
        actual_model_MemSet_destinations_lengths_or_BOOL_fields_recovered=False,overall_complete=False))
    write(O/'native-pad-consequences-v120.json',dict(
        authorized_diagnostic='GRANTED_AND_CONSUMED;oneactualcapture,noequivalentretrywithoutnewpurpose/approval.',
        capability='NativeMemSetKERNELGetParamssucceeds,andcontrolleddumpcontainsboundedinlineflowtableforonelegacyprovider. Priorv109MEMCPYresultwasadifferentroute.',
        invalid_inferences=['argsSize152=>152Bfill','96Brecord=>96Bfill','workspace1536=>1536Bwritten/allocated','launch48=>48writers','onediagnostic=>all1272historicalMemSetsdecoded','debugdur9.5=>op9.5us'],
        operator_scope='WholePad/initializerfusionremainsconditionalonpreservingBOOLprefix/Truepadding,allreaders,error/alias/captureandbarriers. Do notdeleteonlyoldMemSetorretire1536Bfromthisone32Bsource-span.',
        further_evidence='Originaltaskprovider/argumentpacket/generationandallreadersmustbebound;noexistingmodelcapture/replayoradditionalNPUprobeauthorizedbythiscompletedone.',
        pending_original_regimes=9,pending_original_executions=1352,new_native_promotions=0,new_pairs=0,new_groups=0,
        new_device_work=True,physical_time_floor_MFU_or_speedup_proved=False,overall_complete=False))
    print(json.dumps(dict(status='ANALYZED',one_actual_capture=True,tasks=6,native_MemSet_GetParams=0,
        MemSet_args_bytes=152,flowtable_offset=24,source_requested_zero_bytes=32,workspace_metadata_bytes=1536,
        original_model_gaps_closed=0,SSH_closed=True)))

if __name__=='__main__':main()
