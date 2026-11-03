"""Installed native PadV3/MemSet protocol evidence and bounded metadata requirements."""
from pathlib import Path
from collections import Counter
import ast,hashlib,json,struct
from finalize_origin_coverage_v9 import N,O,read,write
from decode_memset_metadata_v111 import FORMAT,FIELDS,decode_record,write_spans,inline_flowtable

def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def sections(p,names):
    p=Path(p);text=p.read_text();tree=ast.parse(text);parts=[]
    for name in names:
        n=next(n for n in ast.walk(tree)if isinstance(n,(ast.ClassDef,ast.FunctionDef))and n.name==name)
        parts.append(dict(name=name,first_line=n.lineno,last_line=n.end_lineno,source=text.splitlines()[n.lineno-1:n.end_lineno]))
    return dict(**pin(p),parts=parts)

def main():
    assert not(O/'REVIEW-CHECKPOINT-v111.json').exists()
    indices=['pad-provider-source-index-v111.json','pad-provider-discovered-index-v111.json','pad-provider-padmeta-index-v111.json','pad-provider-binary-index-v111.json']
    rows=[r for name in indices for r in read(O/name)['files']if r['status']=='READ']
    for r in rows:assert pin(r['local_path'])['sha256']==r['sha256']
    dynamic=next(Path(r['local_path'])for r in rows if r['path'].endswith('/ops_legacy/dynamic/mem_set.py'))
    pad=next(Path(r['local_path'])for r in rows if r['path'].endswith('/ops_legacy/dynamic/pad_v3.py'))
    src=[sections(dynamic,['Constant','get_tiling_data','data_move_out','init_value_set','compute_value_set_ele_size','addr_clean','build','mem_set']),
        sections(pad,['Constant','tiling_args','init_src_dst_gm','pad_v3_d_compute_tiling'])]
    original=read(O/'native-pad-probe-plan-v111.json')['original_call_source'];original_path=Path(original['path'])
    assert pin(original_path)['sha256']==original['sha256']and 'F.pad(is_padding, (0, sp_pad), value=True)'in original_path.read_text()
    src.append(sections(original_path,['_ascend_sp_padding_mask_impl']))
    write(O/'pad-provider-source-contract-v111.json',dict(source=src,
        facts=['Selected legacyPadV3 creates sync_workspace INT64[core_nums*4] with workspace/atomic flags exceptAscend610/BS9SX1A;output atomic flag isFalse on other reviewedSoCs.',
            'OutsideMODE4 on thoseSoCs,PadV3 fills output,then block_barrier(sync_workspace),then do_pad. MODE4 skips that outer fill/barrier;only one sync_workspace reference is a barrier call in the file.',
            'Selected default dynamicMemSet accepts separate workspace address arguments andflowtable records;it is not the byte/D32 API that lowered toMEMCPY in v109.',
            'Dynamic default zero path reads24 INT32 slots perworkspace,with mixedI32/I64 fields;addresses andtiling values are required to derive actualstore spans.'],
        current_installed_sources_not_historical_loaded_objects=True,no_new_device_work=True,overall_complete=False))
    metas=[];padmetas=[];binary_rows={r['path']:r for r in read(O/'pad-provider-binary-index-v111.json')['files']}
    for r in rows:
        p=Path(r['local_path'])
        if p.suffix!='.json' or '/kernel/config/'in r['path']:continue
        d=read(p)
        if p.name.startswith('MemSet_'):
            v=d['compileInfo']['vars'];assert v['workspace_num']in range(1,17)
            metas.append(dict(**pin(p),kernelName=d['kernelName'],workspace_num=v['workspace_num'],core_num=v['core_num'],
                parameters=len(d['parameters']),opParaSize=d['opParaSize'],dfx_size=d['compileInfo']['dfx_size'],
                per_workspace_source_record_bytes=96,binary_declared_sha256=d['sha256']))
        elif p.name.startswith('PadV3_'):
            v=d['compileInfo']['vars'];w=d['workspace'];assert w==dict(num=1,size=[1536],type=[0])
            padmetas.append(dict(**pin(p),kernelName=d['kernelName'],dtype=d['supportInfo']['inputs'][0]['dtype'],
                coreType=d['coreType'],compiled_core_num=v['core_num'],workspace=w,parameters=d['parameters'],opParaSize=d['opParaSize'],
                workspace_initializer_parameter_index=4,source_sync_bytes=48*4*8))
    assert sorted(m['workspace_num']for m in metas)==list(range(1,17))and len(padmetas)==8
    controls=read(O/'native-pad-probe-provider-controls-v111.json');assert controls['all_binary_checksums_match']
    write(O/'pad-provider-metadata-contract-v111.json',dict(MemSet_variants=sorted(metas,key=lambda r:r['workspace_num']),PadV3_variants=padmetas,
        binary_controls=controls,source_only_conditional_role='In this pinned A3legacyPadV3 family,one1536B zero-initialized barrier workspace is a concrete candidate for the prependedMemSet destination. Original call addresses/kernel hashes are not recovered.',
        no_opParaSize_to_tensor_extent_inference=True,no_historical_binary_selection_or_mode_attestation=True,overall_complete=False))
    owners=read(O/'pad-owner-invocations-v72.json')['invocations'];regimes=read(O/'pad-helper-regime-refinement-v72.json')['regimes'];dbs={};native=[]
    for r in regimes:
        blocks=Counter();names=Counter()
        for db,tid in r['native_TASK_keys']:
            if db not in dbs:
                d=read(N/f'original-profile-cache-v53/analysis/native-execution-identity-v14/db-{db:02d}.json')
                dbs[db]=({t['task_record_rowid']:t for t in d['TASK_records']},{x['globalTaskId']:x for x in d['COMPUTE_definitions']})
            ts,defs=dbs[db];de=defs[ts[tid]['globalTaskId']];blocks[(de['blockNum'],de['mixBlockNum'])]+=1;names[de['name_text']]+=1
        assert blocks=={(48,0):r['executions']}
        native.append(dict(regime_id=r['regime_id'],executions=r['executions'],blocks=[dict(blockNum=k[0],mixBlockNum=k[1],count=v)for k,v in blocks.items()],names=dict(names),binary_hash_or_workspace_address_bound=False))
    cohorts=Counter((r['field'],r['context'],r['PadV3_signature']['inputShapes_text'],r['PadV3_signature']['inputDataTypes_text'],r['PadV3_signature']['outputShapes_text'])for r in owners)
    write(O/'pad-provider-native-reconciliation-v111.json',dict(regimes=native,executions=sum(r['executions']for r in native),
        contexts=dict(Counter(r['context']for r in owners)),typed_shape_cohorts=[dict(field=k[0],context=k[1],inputs=k[2],types=k[3],output=k[4],executions=v)for k,v in cohorts.items()],
        reuse_of_v72_not_new_owner_attribution=True,all_recorded_blocks48_not_active_writer_count_proof=True,
        current_source_family_not_same_call_operand_evidence=True,pending_regimes_promoted=0,overall_complete=False))
    layout=[];offset=0
    for i,(name,fmt)in enumerate(zip(FIELDS,list('iiii')+list('qqq')+list('ii')+list('qqq')+list('iiiiii'))):
        layout.append(dict(name=name,byte_offset=offset,bytes=struct.calcsize('<'+fmt),storage=fmt));offset+=struct.calcsize('<'+fmt)
    assert offset==96 and struct.calcsize(FORMAT)==96
    def record(K,F,L,gap=0,full=16320):
        return dict(select_key=0,need_core_num=K,full_elems=full,full_bursts=full//8,front_elems=F,
            front_loops=F//full,front_part=F//full*full,front_tail_bursts=(F%full*4+31)//32,front_tail_repeats=(F%full+63)//64,
            last_elems=L,last_loops=L//full,last_part=L//full*full,last_tail_bursts=(L%full*4+31)//32,last_tail_repeats=(L%full+63)//64,
            core_num=48,gap_bytes=gap,init_dtype=0,init_value_raw=0)
    examples=[]
    for name,r in [('one_writer_1536',record(1,384,384)),('48writers_same1536',record(48,8,8)),
        ('padded_tail_100',record(1,32,32,28)),('multiple_full_chunks',record(3,16328,16336)),('empty',record(1,0,0))]:
        raw=struct.pack(FORMAT,*[r[k]for k in FIELDS]);assert decode_record(raw)==r
        examples.append(dict(name=name,record=r,raw_hex=raw.hex(),padded_tail=write_spans(r,True),aligned_tail=write_spans(r,False)))
    one=next(m for m in metas if m['workspace_num']==1);metadata=read(Path(one['path']));base=0x10000000;data=struct.pack(FORMAT,*[examples[0]['record'][k]for k in FIELDS])
    packet=struct.pack('<3Q',0x20000000,0,base+24)+data+bytes(8);words=list(struct.unpack('<16Q',packet));extracted=inline_flowtable(words,base,128,metadata)
    assert extracted['records']==[examples[0]['record']]
    write(O/'MemSet-default-zero-metadata-model-v111.json',dict(scope='Pinned legacy dynamic non-long defaultFLOAT32zero source only;CPU supplied metadata,not a native call',
        layout=layout,record_bytes=96,examples=examples,inline_packet_reference=dict(args_address=base,args_size=128,words=words,extracted=extracted),
        packet_packing_and_input_order_are_unvalidated_until_controlled_native_dump=True,
        source_span_bytes_not_semantic_extent_or_HBM_or_capacity=True,original_profile_has_no_such_tiling_values=True,overall_complete=False))
    write(O/'pad-barrier-fusion-conditions-v111.json',dict(
        variants=[dict(id='PAD111-MODE4',finding='In the selected source,MODE4 skips the only sync_workspace barrier reference. A reset-removal candidate exists if the actual loaded mode/ABI/privateworkspace/allreaders andlater resets/retirement are proved.',
            no_unchanged_kernel_drop_without_actual_guard_or_readers=True),
            dict(id='PAD111-NON4',finding='The same source fills output then globally barriers before copying input. Barrier workspace zeroing must be ready before any reader;simplymoving per-coreclears into the kernel is not automatically legal.',
                alternatives='One-pass disjoint fullwriter alreadyv72 can remove the fill/barrier protocol;or a newepoch protocol must prove initialization/wrap/activeparticipants/concurrentreaders. Neither is implemented.'),
            dict(id='PAD111-WRITERCOUNT',finding='48 launched MemSet blocks do not establish48 usefulwriters or1536Bactualwrites;syntheticK1/K48 records can submit the same1536Bspan.')],
        costs=dict(selected_barrier_workspace_source_bytes=1536,
            frozen_non4_reference='I input bytes,O output bytes:Pad readI+copywriteI+fillwriteO;separate resetW gives2I+O+W logical references.',
            direct_explicit_output_reference='I+O withdisjoint one-passoutputwriters;extra mapping/state/readers costs still charged.',
            existing_v72_credit='Refine one chosen baseline;do not add the same Pad/cast/interface credits twice.',
            no_numeric_physical_ceiling_MFU_or_speedup=True),
        original7MemSet_and2BOOL_fields_still_pending=True,new_device_work=False,overall_complete=False))
    paths=[N/'pad-provider-source166-v111.json',N/'pad-provider-source166-v111b.json',N/'pad-provider-source166-v111c.json',N/'pad-provider-source166-v111d.json',
        N/'pad-provider-inventory166-v111.json',O/'pad-owner-invocations-v72.json',O/'pad-helper-regime-refinement-v72.json',O/'native-pad-probe-provider-controls-v111.json',O/'native-pad-probe-plan-v111.json',
        N/'diagnostic-native-pad-probe-v111/inspect_pad_graph.py',N/'decode_memset_metadata_v111.py',original_path]+[O/s for s in indices]+[Path(r['local_path'])for r in rows]
    paths += [N/f'original-profile-cache-v53/analysis/native-execution-identity-v14/db-{db:02d}.json'for db in dbs]
    paths=list(dict.fromkeys(paths));write(O/'pad-provider-inputs-v111.json',dict(inputs=[pin(p)for p in paths],new_device_work=False,
        one_bounded_container_Python_filename_inventory_only=True,remaining_operations_archive_copy=True,overall_complete=False))
    print(json.dumps(dict(status='SOURCE_NATIVE_PROTOCOL_REVIEWED',MemSet_variants=16,PadV3_variants=8,binaries_verified=3,
        existing_native_executions=1272,typed_cohorts=len(cohorts),metadata_examples=5,probe='PREPARED_NOT_EXECUTED',original_gaps_closed=0)))

if __name__=='__main__':main()
