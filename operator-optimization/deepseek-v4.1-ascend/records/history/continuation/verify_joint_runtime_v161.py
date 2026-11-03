"""Independent bounded checks of acquisition bytes, scope and state transition.

No runtime imports, native calls, historical operator-suite reruns or profile replay.
"""
import sys,json,struct,hashlib,math
from pathlib import Path
from datetime import datetime,timezone
from finalize_origin_coverage_v9 import N,O,T as TASK,L as LOOP,read,write
from analyze_evidence_batches_v155 import pin
PRE='--preflight' in sys.argv
gates=[]
def check(ok,msg):
    assert ok,msg
    gates.append(msg)
def valid(p):
    b=Path(p['path']).read_bytes()
    assert len(b)==p['bytes'] and hashlib.sha256(b).hexdigest()==p['sha256'],p['path']
manifest=O/('checkpoint-manifest-v159.json' if PRE else 'checkpoint-manifest-v161.json')
sealed=read(manifest)
for row in sealed['artifacts']:valid(row)
check(True,'Predecessor/current authority pins match before/after the shared-state transition')
inputs=read(O/'joint-runtime-inputs-v161.json')['inputs']
for row in inputs:valid(row)
check(True,'All newly collected immutable inputs match their byte and SHA256 pins')

dups=read(O/'six-duplicate-parameter-values-v160.json')
expected={f'layers.{i}.engram.{tail}' for i in [1,14] for tail in ['k_weight','q_weight','wkv.weight']}
check({r['name'] for r in dups['rows']}==expected and len(dups['rows'])==6,'All and only six preidentified duplicate parameter pairs were compared')
for r in dups['rows']:
    assert len(r['copies'])==2 and r['direct_bytewise_equal'] and r['first_different_block_start_if_any'] is None
    assert r['copies'][0]['sha256']==r['copies'][1]['sha256']
    assert r['payload_bytes_each']==math.prod(r['shape'])*2
    assert not r['actual_runtime_loader_selection_attested']
check(dups['bytes_read']==sum(r['payload_bytes_each']*2 for r in dups['rows'])==1258618880 and dups['stable_file_stats_before_after'],
 'Current duplicate content receipts reconcile exactly; runtime/history not inferred')

loaded=read(O/'loaded-provider-bindings-v160.json');backing=read(O/'mapped-provider-backing-objects-v160.json')
records=loaded['loaded_provider_mapping_records']
check(len(records)==40 and len({r['pid'] for r in records})==20 and loaded['examined_owned_processes']==21,
 'Provider mapping census stays within the original own-container process episode')
for obj in backing['objects']:
    mappings=[r for r in records if r['path']==obj['mapped_path']]
    assert len(obj['candidates'])==1 and len(mappings)==20
    candidate=obj['candidates'][0]
    for r in mappings:
        assert int(r['mapped_device'].split(':')[0],16)==int(candidate['device'].split(':')[0],16)
        assert int(r['mapped_device'].split(':')[1],16)==int(candidate['device'].split(':')[1],16)
        assert r['mapped_inode']==candidate['inode'] and r['sha256']==candidate['sha256']
    assert not obj['runtime_call_selection_or_private_COW_text_attested']
check(len(backing['objects'])==2,'Overlay backing files match kernel mapping device/inode and full ELF hashes')
header=Path(loaded['header']['local_path']).read_bytes()
check(len(header)==loaded['header']['bytes'] and hashlib.sha256(header).hexdigest()==loaded['header']['sha256'],
 'Exact CANN9.2 header bytes pinned, without repeating an SDK survey')
check(all(len(v)==1 for v in loaded['declarations'].values()) and len(loaded['declarations'])==4,
 'Four relevant API declarations retained with exact signatures')
abi=read(O/'provider-sdk-abi-layout-v160.json')
check(abi['build_rc']==abi['execution_rc']==0 and abi['values']=={'pointer_bytes':8,'size_t_bytes':8,'return_bytes':4,'placeholder_bytes':8,'placeholder_align':4,'addr_offset':0,'data_offset':4},
 'CPU-only compile/type/layout checks agree with the real SDK')
check(not abi['new_NPU_context'] and not abi['CANN_runtime_linked_or_called'] and len(abi['dynamic_dependencies'])==1 and 'libc.so.6' in abi['dynamic_dependencies'][0],
 'ABI executable linked only libc; no CANN/context evidence invented')
cpu=read(N/'host-adapter-CPU166-v127c.json');cpu=json.loads(cpu['envelope']['stdout'])
check(len(cpu['cases'])==23 and all(r['returncode']==0 for r in cpu['cases']) and not cpu['CANN_runtime_loaded_in_tests'],
 'Final successful v127c CPU receipt reused; initial failed build is not treated as success')

auth=read(O/'native-hostargs-authorization-v161.json');prep=read(O/'native-hostargs-probe-preparation-v161.json')
check(auth['user_reply']=='允许本次探针并恢复服务' and auth['at_most_one_native_attempt'],
 'Specific authorization is one probe plus service restore')
for filename,key in [('native_hostargs_pad171_v161.py','probe_source_sha256'),('native_hostargs_controller171_v161.py','controller_source_sha256')]:
    assert pin(N/filename)['sha256']==auth[key]==prep[key]
check(True,'Executed probe and controller match the authorized source hashes')
idx=read(O/'joint-hostargs-probe-file-index-v161.json')
for r in idx['files']:valid(dict(path=r['local_path'],bytes=r['bytes'],sha256=r['sha256']))
check(len(idx['files'])==25 and sum(r['bytes'] for r in idx['files'])==idx['bytes']==108031 and idx['bytes']<8388608,
 '25 bounded diagnostic files match raw hashes and output guard')
base=N/'joint-hostargs-probe-cache-v161/bundle'
sample=read(base/'probe-result.json')
check(sample['status']=='EAGER_CONTROL_OBSERVED_VALUES_MATCH','Both actual eager Pad outputs passed the expected-value comparison')

# Deliberately separate raw parsing from the production decoder.
def split(raw):
    at=0;out=[]
    while at<len(raw):
        assert len(raw)-at>=6
        kind=int.from_bytes(raw[at:at+2],'little');size=int.from_bytes(raw[at+2:at+6],'little');at+=6
        assert at+size<=len(raw);out.append((kind,raw[at:at+size]));at+=size
    assert at==len(raw)
    return out
ph={}
for label in ['control','observed']:
    p=next((base/'profiles'/label).glob('*/FRAMEWORK/torch.op_range'))
    rows=[]
    for tag,raw in split(p.read_bytes()):
        assert tag==1 and len(raw)>=58
        fields={k:v.decode() for k,v in split(raw[58:])};rows.append(fields)
    ph[label]=rows
check(not any(k in r for r in ph['control'] for k in [5,6,7]) and sum(5 in r for r in ph['observed'])==6,
 'Raw control lacks detail tags; observed memory-enabled profile emits six tensor-detail events')
pad=next(r for r in ph['observed'] if r[2]=='aten::constant_pad_nd')
f=pad[5].rstrip(')').split(';');view=sample['input'];owner=sample['input_owner']
check(len(f)==8 and int(f[0],0)==view['TensorImpl'] and int(f[1],0)==view['storage_base']==owner['storage_base'] and view['tensor_data_ptr']==view['storage_base']+1,
 'Raw 8-field tensor serializes storage base, not the offset-one view data pointer')
check(pad[7]=='Bool' and 6 not in pad and view['storage_offset']==1 and view['storage_nbytes']==21,
 'Scalar remains a type label; actual view offset/extent are supplemental facts')
obs=read(O/'joint-profiler-capability-observation-v161.json')
mempath=next((base/'profiles'/'observed').glob('*/FRAMEWORK/torch.memory_usage'))
mem=[struct.unpack('<7q2b3B2Q',raw) for tag,raw in split(mempath.read_bytes())]
check(len(mem)==6 and any(r[2]==2560 and r[7]==20 for r in mem),'Six raw allocator records include a 2560-byte NPU allocation')
trace=next((base/'trace').glob('*.jsonl'));lines=[json.loads(x) for x in trace.read_text().splitlines()]
native=[r for r in lines if r['type']=='host_args'];foot=[r for r in lines if r['type']=='summary']
check(len(native)==len(foot)==1 and native[0]['launch_rc']==0 and foot[0]['written']==1,
 'Exactly one successful real HostArgs packet was recorded')
packet=bytes.fromhex(native[0]['args_hex']);word0=int.from_bytes(packet[:8],'little')
check(len(packet)==152 and native[0]['args_size']==152 and native[0]['placeholders']==[[8,24]],
 '152-byte packet and placeholder pair verified without treating a zero relocation slot as NULL')
ranges=[r for r in mem if r[2]>0 and r[7]==20 and r[0]<=word0<r[0]+r[2]]
check(len(ranges)==1 and ranges[0][2]==2560 and word0-ranges[0][0]==1024,
 'Raw first pointer lies inside the recorded allocation at offset1024; role/lifetimes still unknown')
check(obs['native_task_id_and_stream_id']['status']=='NOT_QUERIED' and native[0]['task_id_rc']==-1 and native[0]['stream_id_rc']==-1,
 'Unqueried task/stream IDs are logger sentinels, not a native task0 bridge')
reuse=read(O/'joint-native-existing-proof-reuse-v161.json')
fmt='<4i3q2i3q6i';names=list(reuse['decoded_record'])
decoded=dict(zip(names,struct.unpack(fmt,packet[24:120])))
check(decoded==reuse['decoded_record']==read(O/'native-store-own-scalar-traces-v128.json')['decoded_fields'] and not reuse['loaded_device_instruction_bytes_attested'],
 'Exact inline-record equality permits only explicitly conditional reuse of the existing scalar proof')
completion=read(O/'native-hostargs-window-completion-v161.json');terminal=completion['controller_terminal']
check(terminal['native_attempts']==1 and terminal['native_exit_code']==0 and terminal['restore']=='OWN_ORIGINAL_SERVICE_HEALTH_200' and completion['health_http']==200 and completion['main_state']=='running' and completion['stage_state']=='exited' and completion['all_current_NPU_processes_owned_by_main'],
 'One probe terminal; original own main service restored healthy and stage exited')
check(not terminal['full_model_capture_ready'] and not obs['full_model_capture_ready'] and obs['original_model_operands_recovered']==0,
 'No full-model, historical1352, bypass/error-path or all-reader promotion')
summary=read(O/'joint-runtime-summary-v161.json')
check(summary['module_inventory']==26 and summary['semantic_rows']==358 and not summary['inventory_exhaustiveness_proved'] and not summary['physical_MFU_or_lowest_established'] and not summary['overall_complete'],
 'Full scope, whole-module completion and physical ceilings remain open')
if not PRE:
    cp=read(O/'REVIEW-CHECKPOINT-v161.json');ob=read(Path(cp['coverage_obligations']))
    check({r['id'] for r in ob['obligations']}=={f'G{i:02d}' for i in range(15)} and cp['whole_modules_closed']==0,
     'All15 broad coverage obligations preserved')
    cases=read(TASK/'cases.json')['cases']
    expected=read(O/'case-registrations-v161.json')['cases']
    check(all(c['id'] in cases and cases[c['id']]['attributes']==c['attributes'] for c in expected) and 'c8-input32k-output1k-prefix90-166' in cases and 'c8-input32k-output1k-prefix90' in cases,
     '171 full/smoke/probe cases registered without overwriting169/166 cases')
result={'status':'PASS','checkpoint':161,'preflight':PRE,'gates':gates,'gate_count':len(gates),
 'sealed_manifest':pin(manifest),'sealed_manifest_files':len(sealed['artifacts']),
 'input_pins_verified':len(inputs),'verification_scope':'New immutable evidence integrity, independent raw byte decoding, exact guards and scoped authority transition only.',
 'old_unchanged_operator_suites_reexecuted':False,'new_NPU_work_in_verifier':False,
 'overall_complete':False,'verifier':pin(Path(__file__)),'observed_utc':datetime.now(timezone.utc).isoformat()}
write(O/('joint-runtime-preflight-v161.json' if PRE else 'joint-runtime-postcheck-result-v161p.json'),result)
if not PRE:write(N/'CURRENT-POSTCHECK.json',result)
print(json.dumps({'status':'PASS','preflight':PRE,'gates':len(gates),'pins':len(sealed['artifacts']),'overall_complete':False}))
