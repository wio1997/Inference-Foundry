"""Match cached native instructions to the own-Pad scalar control/operand plan."""
from pathlib import Path
from collections import Counter
import hashlib,json,struct
from finalize_origin_coverage_v9 import N,O,read,write
from decode_memset_metadata_v111 import FIELDS,FORMAT,decode_record,write_spans
from native_memset_scalar_model_v128 import elf_sections,instructions,scalar_trace


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
    assert not (O/'REVIEW-CHECKPOINT-v128.json').exists()
    old=read(O/'pad-provider-binary-index-v111.json')['files'];dis=read(O/'native-static-disassembly-index-v128.json')
    parsed=[];all_code={}
    for row in dis['objects']:
        src=next(x for x in old if x['path']==row['path']);blob=Path(src['local_path']).read_bytes()
        assert len(blob)==row['bytes']and hashlib.sha256(blob).hexdigest()==row['sha256']
        sections=elf_sections(blob);code=instructions(Path(row['local_disassembly']).read_text());text=sections['.text']
        assert sorted(code)==list(range(text['address'],text['address']+text['bytes'],4))
        for pc,ins in code.items():assert struct.unpack_from('<I',text['data'],pc-text['address'])[0]==ins['word']
        arg=sections['__CCE_KernelArgSize'];assert arg['bytes']==4
        parsed.append(dict(path=src['local_path'],sha256=row['sha256'],disassembly=row['local_disassembly'],
            text_sha256=text['sha256'],text_bytes=text['bytes'],instruction_words_checked=len(code),
            CCE_KernelArgSize_section_uint32=struct.unpack('<I',arg['data'])[0],
            sections=[{k:v for k,v in s.items()if k!='data'}for s in sections.values()],
            opcode_inventory=dict(Counter(i['op']for i in code.values()))))
        all_code[Path(row['path']).name]=(code,sections)
    pad=[x for x in parsed if Path(x['path']).name.startswith('PadV3_')]
    assert pad[0]['text_sha256']==pad[1]['text_sha256']and pad[0]['sha256']!=pad[1]['sha256']
    write(O/'native-store-ELF-controls-v128.json',dict(objects=parsed,
        both_int8_Pad_contiguity_variants_have_identical_text=True,
        equal_text_does_not_establish_equal_metadata_or_runtime_input_plan=True,
        actual_model_loaded_binary_bound=False,actual_loaded_instruction_bytes_attested=False,new_device_work=False,overall_complete=False))
    code,sections=next(v for k,v in all_code.items()if k.startswith('MemSet_'))
    own=read(O/'native-pad-observations-v120.json');task=next(x for x in own['tasks']if x['name'].startswith('MemSet_'))
    packet=b''.join(x.to_bytes(8,'little')for x in task['rendered_words']);base=task['args_address'];flow=task['rendered_words'][1]
    assert flow-base==24 and len(packet)==152
    record=packet[24:120];fields=decode_record(record);assert fields==read(O/'native-pad-packet-and-store-contract-v120.json')['decoded_source_record']
    traces=[scalar_trace(code,record,task['rendered_words'][0],flow,b)for b in range(task['numBlocks'])]
    output=[dict(block=t['block'],**e)for t in traces for e in t['events']if e['kind']=='OUTPUT_STORE_SITE']
    assert len(output)==1 and output[0]['block']==0 and output[0]['pc']==0x478
    assert output[0]['destination_relative']==0 and output[0]['raw_control']==(32<<16)|1
    source=write_spans(fields,True);assert source['union']==[[0,32]]
    write(O/'native-store-own-scalar-traces-v128.json',dict(
        scope='Explicit own-v120 byte record plus pinned native scalar-control semantics;not runtime execution or full ISA simulation.',
        compiled_MemSet_argument_size_field=16,full_saved_argument_packet_bytes=152,
        native_entry_pointer_load=dict(pc=0x40,op='LDP.b64',first_pointer_offset=0,second_pointer_offset=8),
        own_flowtable_offset=24,source_record_bytes=96,decoded_fields=fields,traces=traces,output_store_sites=output,
        source_requested_spans=source,all48_enter_outer_core_count_guard=True,only_block0_reaches_output_store_site=True,
        input_transfer_sites_per_block=3,
        raw_MOVX8_encoding_not_generally_decoded=True,vector_values_cache_pipeline_DMA_or_physical_HBM_not_simulated=True,
        disk_object_symbol_matches_saved_own_kernel_name=True,actual_loaded_instruction_bytes_attested=False,
        no_argument_address_dereferenced=True,new_device_work=False,original_pending_operands_recovered=0,overall_complete=False))
    synthetic=[]
    for K in range(49):
        for gap in [0,4,28]:
            f=dict(fields,need_core_num=K,front_elems=8,last_elems=8-gap//4,gap_bytes=gap,
                front_loops=0,last_loops=0,front_part=0,last_part=0,front_tail_bursts=1,last_tail_bursts=1)
            raw=struct.pack(FORMAT,*(f[k]for k in FIELDS));src=write_spans(f,True);outs=[]
            for b in range(48):
                t=scalar_trace(code,raw,0x12000000,0x13000000,b)
                outs.extend(dict(block=b,**e)for e in t['events']if e['kind']=='OUTPUT_STORE_SITE')
            assert len(outs)==len(src['spans'])==K
            for e,s in zip(outs,src['spans']):
                assert e['block']==s['core']and e['destination_relative']==s['begin']
                expected=s['end']-s['begin']
                encoded=((expected if 'ALIGN'in e['op']else expected//32)<<16)|1
                assert e['raw_control']==encoded
            synthetic.append(dict(need_core_num=K,gap_bytes=gap,blocks_checked=48,
                store_sites=len(outs),source_requested_bytes=src['submitted_store_value_bytes'],
                source_plan_and_raw_descriptor_arithmetic_match=True,native_tiling_legality_proved=False))
    write(O/'native-store-synthetic-control-validation-v128.json',dict(
        cases=synthetic,records_checked=len(synthetic),block_traces_checked=len(synthetic)*48,
        scope='Synthetic zero-FLOAT32 loop-free records;K0..48,gap0/4/28. Not proof tiler emits these records or full ISA behavior.',
        actual_native_execution=False,new_device_work=False,overall_complete=False))
    write(O/'native-store-fusion-consequences-v128.json',dict(
        findings=[
            dict(id='NATIVE128-ARGUMENT-SIZES',finding='MemSet ELF __CCE_KernelArgSize contains16;entry LDP reads two64bit address operands. Own snapshot packet152 and inline record offset24 remain separate concepts. Pad field56 and packet320 are also different sizes.',
                implication='A host packet size,compiler argument-size field,source record size,workspace declaration and source store span cannot substitute for one another.'),
            dict(id='NATIVE128-WRITER-GUARDS',finding='For saved own-v120 zero record,the pinned scalar control reaches one output store site at0x478 onblock0. All48 pass outercore_num48 guard and visit3inputtransfer instruction sites before useful-writer selection.',
                implication='48launchedblocks and generic MemSet name cannot justify48usefulwriters or bandwidth-only ceiling. Static instruction visits are not measured latency or HBM bytes.'),
            dict(id='NATIVE128-CONFIG',finding='Own sole output site usesworkspacebase andraw descriptor0x200001;source-matched request is32B. 147synthetic records/7056blocktraces matchsource plan anddescriptor arithmetic underexplicit model premises.',
                implication='Strengthens own-provider control/operand plan only. No physicalstore measurement,no original1272parameter transfer andno isolatedhelperdeletion proof.'),
            dict(id='NATIVE128-TEXT-IDENTITY',finding='Two cached INT8 Pad contiguity variants haveidentical12748Btext,butdifferentfullobjecthashes andmetadata.',
                implication='Kernel text/name equality alone cannot selectattrs/flowtable/storage/reader obligations.')],
        native_scalar_model_limitations=['Saved own graph was captured,not replayed. No new hardware operation occurred.',
            'Model supplies the96Brecord toscalar UB view atknowninputtransfer sites;MOVX8 DMA encoding is not generallydecoded.',
            'Vector/cache/pipeline/coherence/latency/physicaltransaction effects andrelocation/load coverage are outside this scalar model.',
            'Unknowninstructions/registers/memory failclosed. Supported cases areloop-free defaultFLOAT32zero only.',
            'Original loadedbinary,actualfieldvalues,destinationallocation/alias/workspace/allreaders andcapturedBOOL remainunbound.'],
        physical_ceiling_or_MFU_or_speedup_proved=False,original_pending_promotions=0,new_device_work=False,overall_complete=False))
    inputs=[O/'host-adapter-postcheck-result-v127p.json',O/'native-static-disassembly-index-v128.json',
        N/'native-static-preflight166-v128.json',N/'native-static-tool-discovery166-v128.json',N/'native-static-disassembly166-v128.json',
        O/'pad-provider-binary-index-v111.json',O/'native-pad-observations-v120.json',O/'native-pad-packet-and-store-contract-v120.json',
        N/'decode_memset_metadata_v111.py',N/'native-static-session-close-v128.json',N/'native-static-session-terminal-v128.json']
    inputs.extend(Path(x['local_path'])for x in old)
    inputs.extend(Path(x['local_disassembly'])for x in dis['objects'])
    write(O/'native-store-inputs-v128.json',dict(inputs=[pin(p)for p in inputs],questions=[
        'Does the pinned native entry consume the proposed address operands, and do distinct argument-size concepts agree only where actually supported?',
        'For the already-saved own zero record,which native scalar guards and output-store sites are reachable across48blocks?',
        'Does native scalar/descriptor arithmetic agree with the independent source store model without inferring execution,HBM traffic,old operands or legal fusion?'],
        no_model_framework_survey=True,no_NPU_runtime_execution=True,overall_complete=False))
    print(json.dumps(dict(status='NATIVE_SCALAR_PLAN_RECONCILED',objects=3,instruction_words_checked=sum(r['instruction_words_checked']for r in parsed),
        own_blocks=48,own_output_store_sites=len(output),synthetic_records=147,synthetic_block_traces=7056,new_device_work=False)))


if __name__=='__main__':main()
