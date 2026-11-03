"""Verify native disk words, saved-input scalar certificates and unchanged gaps."""
from copy import deepcopy
from pathlib import Path
import argparse,hashlib,json,struct
from finalize_origin_coverage_v9 import N,O,T,L,read,write
from native_memset_scalar_model_v128 import elf_sections,instructions,scalar_trace


def coreid_independent(code,path):
    """Conservative taint: no COREID-derived scalar reaches guards/addresses.

    This checks only scalar dataflow on the supported paths. It does not model
    hardware topology or vector/cache/pipeline effects.
    """
    tags={}
    def tagged(a):return tags.get(a.strip('[]'),False)
    for pc in path:
        ins=code[pc];op=ins['op'];a=ins['args']
        if op=='MOV':tags[a[0]]=a[1]=='COREID'or tagged(a[1])
        elif op=='MOVX8':tags[a[0]]=False
        elif op=='MOVK':pass  # Conservatively retain any old tag.
        elif op in ['NEG.s64','SIGNEXT.s32','ZEROEXT.u32']:tags[a[0]]=tagged(a[1])
        elif op in ['SHL.b64','SHR.u64','SHR.s64']:tags[a[0]]=tagged(a[0])or tagged(a[1])
        elif op=='INSERT':tags[a[0]]=tagged(a[0])or any(map(tagged,a[1:]))
        elif op in ['ADD.s64','SUB.s64','MUL.s64','AND.b64','OR.b64','MADD.s64']:
            tags[a[0]]=any(map(tagged,a[1:]))or(op=='MADD.s64'and tagged(a[0]))
        elif op.startswith(('LD.','LDP.')):
            pair=op.startswith('LDP.');i=2 if pair else 1
            assert not any(map(tagged,a[i:])),('COREID-dependent load',pc)
            tags[a[0]]=False
            if pair:tags[a[1]]=False
        elif op.startswith(('CMP.','JUMP','MOV_UB_TO_OUT','MOV_OUT_TO_UB')):
            assert not any(map(tagged,a)),('COREID-dependent guard/transfer',pc)
    return True


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
    def ck(name,value):
        assert value,name
        checks.append(name)
    def pinned(r):
        b=Path(r['path']).read_bytes();return len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
    for r in read(O/'native-store-inputs-v128.json')['inputs']:ck('input:'+r['path'],pinned(r))
    receipt=read(N/'native-static-disassembly166-v128.json');env=receipt['envelope'];result=json.loads(env['stdout'])
    ck('static_controller_terminal',receipt['returncode']==env['returncode']==0 and env['container_pid']==2032500 and not result['new_device_work']and not result['CANN_or_Torch_runtime_loaded']and not result['original_objects_modified'])
    index=read(O/'native-static-disassembly-index-v128.json');controls=read(O/'native-store-ELF-controls-v128.json')
    ck('vendor_decoder',index['tool']==result['tool']and index['tool']['version_returncode']==0 and 'hiipu64'in index['tool']['version_stdout'])
    allcode={};words=0
    for row in index['objects']:
        original=next(x for x in read(O/'pad-provider-binary-index-v111.json')['files']if x['path']==row['path'])
        raw=Path(original['local_path']).read_bytes();sections=elf_sections(raw);text=sections['.text']
        asm=Path(row['local_disassembly']).read_text();code=instructions(asm);allcode[Path(row['path']).name]=code
        ck('remote_to_local:'+row['path'],next(x for x in result['objects']if x['path']==row['path'])['stdout']==asm and row['returncode']==0 and not row['stderr'])
        ck('ELF_text_coverage:'+row['path'],sorted(code)==list(range(0,text['bytes'],4)))
        for pc,ins in code.items():ck('native_word:'+Path(row['path']).name+':'+str(pc),struct.unpack_from('<I',text['data'],pc)[0]==ins['word'])
        words+=len(code)
        saved=next(x for x in controls['objects']if x['path']==original['local_path'])
        ck('section_control:'+row['path'],saved['text_sha256']==text['sha256']and saved['instruction_words_checked']==len(code)and saved['CCE_KernelArgSize_section_uint32']==struct.unpack('<I',sections['__CCE_KernelArgSize']['data'])[0])
    ck('8081native_words',words==8081)
    pad=[x for x in controls['objects']if Path(x['path']).name.startswith('PadV3_')]
    ck('equal_text_not_equal_object',pad[0]['text_bytes']==pad[1]['text_bytes']==12748 and pad[0]['text_sha256']==pad[1]['text_sha256']and pad[0]['sha256']!=pad[1]['sha256'])
    code=next(v for k,v in allcode.items()if k.startswith('MemSet_'));own=read(O/'native-store-own-scalar-traces-v128.json')
    task=next(x for x in read(O/'native-pad-observations-v120.json')['tasks']if x['name'].startswith('MemSet_'))
    packet=b''.join(x.to_bytes(8,'little')for x in task['rendered_words']);record=packet[24:120]
    ck('distinct_sizes',own['compiled_MemSet_argument_size_field']==16 and own['full_saved_argument_packet_bytes']==152 and own['source_record_bytes']==96 and own['own_flowtable_offset']==24)
    ck('entry_two_addresses',code[0x38]['op']=='MOV'and code[0x38]['args']==['X1','PARA_BASE']and code[0x40]['op']=='LDP.b64'and code[0x40]['args']==['X0','X2','[X1]','#0'])
    ck('actual_inline_address_in_saved_bytes',task['rendered_words'][1]==task['args_address']+24)
    ck('core_count_load_guard',code[0xac]['op']=='LD.b32'and code[0xac]['args']==['X5','[X5]','#80','#0']and code[0xbc]['op']=='CMP.LE.s64'and code[0xc0]['args']==['#6788'])
    ck('need_core_load',code[0x118]['op']=='LDP.b32'and code[0x118]['args']==['X5','X8','[X4]','#4'])
    ck('record_binding',struct.unpack_from('<i',record,4)[0]==1 and struct.unpack_from('<i',record,80)[0]==48 and struct.unpack_from('<ii',record,88)==(0,0))
    for b,t in enumerate(own['traces']):
        current=json.loads(json.dumps(scalar_trace(code,record,task['rendered_words'][0],task['rendered_words'][1],b)))
        ck('own_rederive:'+str(b),current==t)
        ck('own_coreid_independence:'+str(b),coreid_independent(code,t['path']))
        ev=[e for e in t['events']if e['kind']=='INPUT_TRANSFER_SITE']
        ck('own3transfer_sites:'+str(b),[e['pc']for e in ev]==[0x74,0xc4,0x14c]and ev[0]['source']==ev[1]['source']==task['rendered_words'][1])
        stores=[e for e in t['events']if e['kind']=='OUTPUT_STORE_SITE']
        ck('own_writer_predicate:'+str(b),len(stores)==(1 if b==0 else 0))
        ck('own_static_visits_not_cycles:'+str(b),t['instruction_sites_visited']==(180 if b==0 else 140)and not t['hardware_execution_or_timing_proved'])
    store=own['output_store_sites'][0]
    ck('own_only_store',len(own['output_store_sites'])==1 and store['pc']==0x478 and store['destination']==task['rendered_words'][0]and store['raw_control']==0x200001 and store['extra']==[0])
    synthetic=read(O/'native-store-synthetic-control-validation-v128.json');ck('147records7056blocks',synthetic['records_checked']==147 and synthetic['block_traces_checked']==7056 and len(synthetic['cases'])==147)
    for row in synthetic['cases']:
        K,gap=row['need_core_num'],row['gap_bytes']
        ck('independent_loopfree_formula:'+str((K,gap)),row['store_sites']==K and row['source_requested_bytes']==(32*K-gap if K else 0)and not row['native_tiling_legality_proved'])
    bad=deepcopy(code);bad[0]['op']='UNSUPPORTED_OPCODE';rejected=False
    try:scalar_trace(bad,record,0x12000000,0x13000000,0)
    except AssertionError:rejected=True
    ck('unknown_instruction_rejected',rejected)
    wrong=bytearray(record);struct.pack_into('<i',wrong,88,1);rejected=False
    try:scalar_trace(code,bytes(wrong),0x12000000,0x13000000,0)
    except AssertionError:rejected=True
    ck('unsupported_dtype_rejected',rejected)
    ck('no_native_attestation_overclaim',not own['actual_loaded_instruction_bytes_attested']and not controls['actual_loaded_instruction_bytes_attested']and not controls['actual_model_loaded_binary_bound']and own['original_pending_operands_recovered']==0 and not own['new_device_work'])
    terminal=read(N/'native-static-session-terminal-v128.json');ck('SSH_authoritatively_gone',not terminal['control_socket_exists']and not terminal['owned_SSH_master_processes']and terminal['all_static_read_subprocesses_terminal'])
    if args.preflight:manifest=read(O/'checkpoint-manifest-v127.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v128.json');ck('open_progress',cp['status']=='PROGRESS'and not cp['overall_complete']and cp['whole_modules_closed']==0)
        cov=read(O/'candidate-relation-coverage-v128.json');ck('same135_26_460',(cov['scoped_conditional_pairs'],cov['higher_order_groups'],cov['remaining_pair_relations'])==(135,26,460))
        m=read(O/'module-phase-branch-matrix-v128.json');ck('same358_26',len(m['modules'])==26 and sum(len(x['branches'])for x in m['modules'])==358)
        old=read(O/'operator-analysis-coverage-ledger-v127.json');new=read(O/'operator-analysis-coverage-ledger-v128.json')
        ck('native_uncertainty_unchanged',all(old[k]==new[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']))
        for r in read(O/'candidate-record-index-v128.json')['model_candidates']:ck('candidate:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('task_loop_active',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING')
        manifest=read(O/'checkpoint-manifest-v128.json')
    for r in manifest['artifacts']:ck('manifest:'+r['path'],pinned(r))
    out=dict(status='PASS',checks=len(checks),preflight=args.preflight,native_disk_instruction_words=8081,
        saved_own_block_certificates=48,synthetic_records=147,synthetic_block_traces=7056,
        native_execution_or_loaded_instruction_attestation=False,original_pending_promotions=0,new_device_work=False,overall_complete=False)
    if args.preflight:write(O/'native-store-preflight-v128.json',out)
    print(json.dumps(out))


if __name__=='__main__':main()
