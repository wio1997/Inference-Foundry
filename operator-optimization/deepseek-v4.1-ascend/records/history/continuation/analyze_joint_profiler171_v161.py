"""Decode the bounded real probe using the pinned raw TLV schema,without runtime imports."""
import struct,json
from pathlib import Path
from collections import Counter
from finalize_origin_coverage_v9 import N,O,read,write
from analyze_evidence_batches_v155 import pin

BASE=N/'joint-hostargs-probe-cache-v161/bundle'
sample=read(BASE/'probe-result.json')
def tlvs(raw):
    p=0;rows=[]
    while p<len(raw):
        assert p+6<=len(raw);tag,n=struct.unpack_from('<HI',raw,p);p+=6
        assert p+n<=len(raw);rows.append((tag,raw[p:p+n]));p+=n
    assert p==len(raw);return rows
fixed=struct.Struct('<3q4QB?')
def integer(s):return int(s,16) if s.startswith('0x') else int(s)
def vector(s):return [int(x) for x in s.split(',')] if s else []
def tensors(value):
    out=[]
    for item in value.split(')'):
        if not item:continue
        fields=item.split(';');assert len(fields)==8
        out.append({'TensorImpl':integer(fields[0]),'storage_base':integer(fields[1]) if fields[1] else None,'dtype':fields[2],'element_bytes':int(fields[3]),'shape':vector(fields[4]),'stride':vector(fields[5]),'device_type':int(fields[6]),'device_index':int(fields[7]),'emitted_field_count':8})
    return out
phases={}
for label in ['control','observed']:
    files=list((BASE/'profiles'/label).glob('*/FRAMEWORK/torch.op_range'));assert len(files)==1
    events=[];counts=Counter()
    for tag,body in tlvs(files[0].read_bytes()):
        assert tag==1 and len(body)>=fixed.size
        header=fixed.unpack_from(body,0);pairs=tlvs(body[fixed.size:]);fields={k:v.decode() for k,v in pairs};assert len(fields)==len(pairs)
        counts.update(fields.keys())
        event={'name':fields.get(2),'sequence':header[2],'pid':header[3],'tid':header[4],'tags':sorted(fields),'dtype_labels':fields.get(3),'shape_text':fields.get(4)}
        if 5 in fields:event['input_tensors']=tensors(fields[5])
        if 7 in fields:event['scalar_type_labels']=fields[7]
        events.append(event)
    info=read(files[0].parent.parent/'profiler_info.json')
    phases[label]={'raw_op_range':pin(files[0]),'events':events,'field_tag_counts':{str(k):v for k,v in counts.items()},'profiler_configuration':info.get('config',info)}
control=phases['control'];observed=phases['observed']
assert all(k not in control['field_tag_counts'] for k in ['5','6','7'])
assert observed['field_tag_counts'].get('5',0)>0
pad=[e for e in observed['events'] if e['name']=='aten::constant_pad_nd'];assert len(pad)==1
inputs=pad[0]['input_tensors'];chosen=[r for r in inputs if r['TensorImpl']==sample['input']['TensorImpl']];assert len(chosen)==1
tensor=chosen[0];expected=sample['input'];owner=sample['input_owner']
assert tensor['storage_base']==expected['storage_base']==owner['storage_base']
assert tensor['storage_base']!=expected['tensor_data_ptr'] and expected['tensor_data_ptr']-tensor['storage_base']==1
assert tensor['shape']==[20] and tensor['stride']==[1] and tensor['element_bytes']==1
assert expected['storage_offset']==1 and expected['storage_nbytes']==21

mem=[];memory_files=list((BASE/'profiles'/'observed').glob('*/FRAMEWORK/torch.memory_usage'))
assert len(memory_files)==1
fmt=struct.Struct('<7q2b3B2Q');assert fmt.size==77
for tag,b in tlvs(memory_files[0].read_bytes()):
    assert len(b)==fmt.size
    values=fmt.unpack(b);names=['ptr','time_ns','alloc_size','total_allocated','total_reserved','total_active','stream_ptr','device_type','device_index','component_type','data_type','allocator_type','thread_id','process_id']
    mem.append(dict(zip(names,values),outer_tag=tag))
tracefiles=list((BASE/'trace').glob('*.jsonl'));assert len(tracefiles)==1
trace=[json.loads(line) for line in tracefiles[0].read_text().splitlines()]
native=[r for r in trace if r['type']=='host_args'];footers=[r for r in trace if r['type']=='summary'];assert len(native)==len(footers)==1
packet=native[0];footer=footers[0];blob=bytes.fromhex(packet['args_hex']);assert len(blob)==packet['args_size']==152 and packet['placeholders']==[[8,24]]
assert packet['launch_rc']==0 and footer['written']==1 and all(footer[k]==0 for k in ['name_failed','reentrant_skips','quota_skips','time_skips','bad_packet','byte_skips'])
first=int.from_bytes(blob[:8],'little');matches=[r for r in mem if r['ptr']==first]
containing=[dict(r,raw_word0_offset=first-r['ptr']) for r in mem if r['alloc_size']>0 and r['device_type']==20 and r['ptr']<=first<r['ptr']+r['alloc_size']]
public_bases={sample[k]['storage_base'] for k in ['input_owner','input','control_output','observed_output']}
result={'status':'REAL_OWN_CASE_FIELDS_AND_HOST_INTERPOSITION_OBSERVED','scope':'Current171 one69B eagerPad pair;not original model/capturedBOOL/wholegraph','phases':phases,'selected_pad_input_tensor':tensor,'supplemental_public_input_metadata':expected,
 'observed_storage_base_matches_owner_not_offset_view_data_ptr':True,'actual_extra_scalar_or_offset_or_extent_fields_appear_in_builtin_record':False,
 'builtin_fields_runtime_observed_in_this_case':True,'compiled_producer_all_paths_or_all_modules_attested':False,
 'memory_records':mem,'native_host_packet':packet,'native_summary':footer,'native_kernel_name':bytes.fromhex(packet['kernel_name_hex']).decode(),
 'raw_packet_word0':first,'raw_word0_matches_public_tensor_base':first in public_bases,'allocator_records_matching_raw_word0':matches,'recorded_allocation_ranges_containing_raw_word0':containing,
 'word0_typed_destination_and_write_length_decoded':False,'private_workspace_or_public_output_role_not_inferred_from_name_or_pointer_mismatch':True,
 'native_task_id_and_stream_id':{'status':'NOT_QUERIED','reason':'QUERY_IDS=0;rc=-1/task_id0 are logger sentinels,not failed getters or real task0'},
 'allocator_stream_token_and_native_stream_handle_not_automatically_same_namespace':True,
 'time_domain_note':'Raw profiler syscnt and CLOCK_REALTIME host API timestamps were not joined by nearest time.',
 'recorded_packet_bytes':len(blob),'host_interposition_success_for_own_case':True,'internal_CANN_error_state_all_path_equivalence_proved':False,
 'actual_parameter_roles_or_original_buffer_independence_promoted':0,'original_model_operands_recovered':0,'full_model_capture_ready':False,'whole_modules_closed':0,'overall_complete':False}
write(O/'joint-profiler-capability-observation-v161.json',result)
print(json.dumps({'status':result['status'],'control_tags':control['field_tag_counts'],'observed_tags':observed['field_tag_counts'],'pad_input':tensor,'scalar_type_labels':pad[0].get('scalar_type_labels'),'memory_records':mem,'packet_bytes':len(blob),'raw_word0':first,'public_buffer_base_match':first in public_bases,'matching_memory_records':matches,'original_model_operands_recovered':0,'whole_modules_closed':0},ensure_ascii=False))
