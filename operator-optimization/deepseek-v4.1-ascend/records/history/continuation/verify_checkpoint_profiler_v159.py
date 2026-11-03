"""Independent metadata/source checks; no weight payload or old operator suites."""
import json,hashlib,math,re,sys
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
from finalize_origin_coverage_v9 import N,O,T as TASK_DIR,L as LOOP_DIR,read,write
from analyze_evidence_batches_v155 import pin

PRE='--preflight' in sys.argv
gates=[]
def check(ok,label):
    assert ok,label
    gates.append(label)
def valid_pin(row):
    p=Path(row['path']);b=p.read_bytes()
    assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'],str(p)
def strict(p):
    def pairs(items):
        d={}
        for k,v in items:assert k not in d,(str(p),k);d[k]=v
        return d
    return json.loads(Path(p).read_text(),object_pairs_hook=pairs)

manifest=O/('checkpoint-manifest-v158.json' if PRE else 'checkpoint-manifest-v159.json')
sealed=read(manifest)
for row in sealed['artifacts']:valid_pin(row)
check(True,'Sealed authority byte pins match before/after state transition')
inputs=read(O/'checkpoint-profiler-inputs-v159.json')['inputs']
for row in inputs:valid_pin(row)
check(True,'Every newly acquired source/cache input pin matches')

inventory=read(O/'checkpoint-all-header-index-v159.json')
index=strict(inventory['index_source']['local_path'])['weight_map']
headers={};occ=defaultdict(list)
sizes={'I8':1,'F32':4,'BF16':2,'F8_E8M0':1}
for row in inventory['headers']:
    raw=Path(row['header_local_path']).read_bytes()
    assert len(raw)==row['header_bytes'] and hashlib.sha256(raw).hexdigest()==row['header_sha256']
    body=strict(row['header_local_path']);headers[row['shard']]=body
    for name,t in body.items():
        if name=='__metadata__':continue
        a,b=t['data_offsets'];assert 0<=a<=b<=row['source_file_bytes']-8-len(raw)
        assert all(isinstance(x,int) and x>=0 for x in t['shape'])
        assert b-a==math.prod(t['shape'])*sizes[t['dtype']]
        occ[name].append((row['shard'],t))
check(len(headers)==142 and len(index)==143324,'All142 shard headers and143324 index declarations independently reconciled')
check(set(index)==set(occ) and all(k in headers[v] for k,v in index.items()),'No missing indexed tensor; index declarations kept distinct from physical occurrences')
duplicates={k:v for k,v in occ.items() if len(v)>1}
check(set(duplicates)=={f'layers.{i}.engram.{tail}' for i in (1,14) for tail in ('k_weight','q_weight','wkv.weight')},'Exactly six duplicate Engram names preserved across files')
check(sum(map(len,occ.values()))==143330 and all(len(v)==2 for v in duplicates.values()),'Physical143330 occurrences are not143324 unique declarations')
check(Counter(headers[shard][name]['dtype'] for name,shard in index.items())==Counter(inventory['indexed_dtype_label_counts']),'Canonical dtype labels recomputed from declared shards')
check(not inventory['indexed_entries_missing'] and not inventory['index_declared_route_is_not_runtime_loader_selection_proof'] is False and inventory['weight_payload_bytes_read']==0,'Header census does not attest runtime load precedence or payload values')
for row in inventory['duplicate_parameter_names']:
    assert not row['payload_value_equality_proved'] and not row['actual_runtime_loader_winner_proved']

weights=read(O/'output-weight-checkpoint-coverage-v159.json')
expected={f'{prefix}.{i}.attn.{op}.weight' for prefix,count in [('layers',40),('mtp',3)] for i in range(count) for op in ('wo_a','wo_b')}
records=weights['tensor_records'];check({r['name'] for r in records}==expected and len(records)==86,'All80 target and6 MTP output-projection weight headers covered')
qd=strict(N/'checkpoint-operator-cache-v159/metadata/target/quant_model_description.json')
qd=qd.get('quant_model_description',qd)
for row in records:
    t=headers[index[row['name']]][row['name']]
    assert t['dtype']=='BF16' and t['shape']==([8192,4096] if row['kind']=='wo_a' else [5120,8192])
    assert row['quant_description_category']=='FLOAT' and qd[row['name']]=='FLOAT'
check(True,'FLOAT category is distinguished from actual BF16 storage')
check(not weights['current_native_rank_weight_residency_attested'] and not weights['all_weight_values_or_historical_capture_identity_attested'],'Global checkpoint groups are not loaded replicas or historical tensor attestation')

public=read(O/'TorchNPU-profiler-field-sources-v159.json')
check(public['commit']=='1d88b8608fd1e44545b5804026c7a57c44240bb4' and len(public['files'])==8,'Declared installed TorchNPU commit used for eight primary-source files')
for row in public['files']:
    b=Path(row['local_path']).read_bytes();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
    assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==row['git_blob_sha1'] and row['git_tree_blob_matched']
producer=read(O/'TorchNPU-input-field-producer-contract-v159.json');contract=producer['source_contract']
for row in producer['Python_file_byte_matches']:
    a=Path(row['installed_or_cached']['path']).read_bytes();b=Path(row['declared_commit_source']['path']).read_bytes();assert a==b and row['byte_equal']
check(len(producer['Python_file_byte_matches'])==3,'Three installed/cached Python files byte-match declared source')
cpp=(N/'TorchNPU-field-source-v159/torch_npu/csrc/profiler/npu_profiler.cpp').read_text()
expression=re.search(r'bool report_details\s*=\s*(.*?);',cpp,re.S).group(1)
check(all('config.'+k in expression for k in ['report_input_shapes','with_stack','with_modules','profile_memory']) and 'record_op_args' not in expression,'Torch detail expression requires shapes,stack/modules,memory; record_op_args is separate')
check('ActivityType::CPU' in cpp and contract['normal_start_requires_CPU_activity_callback'],'CPU callback prerequisite retained')
for row in producer['gate_witnesses']:
    assert row['source_Torch_detail_gate']==bool(row.get('CPU_activity',True) and row['record_shapes'] and (row['with_stack'] or row['with_modules']) and row['profile_memory'])
cc=(N/'TorchNPU-field-source-v159/torch_npu/csrc/toolkit/profiler/src/data_reporter.cpp').read_text()
check('t.storage().data()' in cc and len(contract['TensorMetadata_fields'])==8,'Serialized tensor pointer is storage base, with eight declared metadata fields')
check(all(x in contract['does_not_serialize'] for x in ['storage_offset','storage_nbytes','allocation generation','tensor version/mutation epoch','actual tensor contents']),'Offsets/extents/generations/contents remain unobserved by this serializer')
check('return std::string("Bool")' in cpp and 'return std::string("Int")' in cpp and 'not scalar values' in contract['TLV7_scalar_payload'],'Scalar payload is type labels, not values')
check(not producer['installed_compiled_producer_and_runtime_bypass_coverage_attested'],'Declared Git/Python matches are not compiled or historical runtime attestation')

ambiguity=read(O/'profiler-field-ambiguities-v159.json')['offset_ambiguity']
a,b=ambiguity['world_disjoint']['A_byte_interval'],ambiguity['world_disjoint']['B_byte_interval']
c,d=ambiguity['world_overlap']['A_byte_interval'],ambiguity['world_overlap']['B_byte_interval']
check(max(a[0],b[0])>=min(a[1],b[1]) and max(c[0],d[0])<min(c[1],d[1]),'Same emitted fields admit disjoint and overlapping reference worlds')
init=read(O/'output-weight-initialization-alternatives-v159.json')
for row in init['initialization_content_models']:
    layers=row['selected_target_layers'];assert row['B_extra_distinct_BF16_weight_content']==layers*56*1024**2 and row['C_extra_distinct_BF16_weight_content']==layers*126*1024**2 and not row['actual_new_allocation_or_fit_proved']
check(init['steady_state_direct_output_value_delivery']['B_to_A']=='2/3' and not init['wire_HBM_latency_or_physical_MFU_proved'],'Startup and steady logical-delivery domains remain separate from allocation/wire/time bounds')
summary=read(O/'checkpoint-profiler-summary-v159.json')
check(summary['module_inventory']==26 and summary['semantic_rows']==358 and not summary['inventory_exhaustiveness_proved'] and not summary['overall_complete'],'Whole original scope stays open; no count/report completion inference')
if not PRE:
    cp=read(O/'REVIEW-CHECKPOINT-v159.json');obs=read(Path(cp['coverage_obligations']))
    check({r['id'] for r in obs['obligations']}=={f'G{i:02d}' for i in range(15)},'All15 broad obligations inherited without deletion')
    check(cp['candidate_ledger'].endswith('v154.json') and cp['module_ledger'].endswith('v154.json') and cp['whole_modules_closed']==0,'Unchanged operator authorities reused and no module closure promoted')
result={'status':'PASS','checkpoint':159,'preflight':PRE,'gates':gates,'gate_count':len(gates),'verification_scope':['All indexed checkpoint metadata and duplicate occurrences','Selected output-weight storage and initialization domains','Exact declared-version profiler producer contract and limits','Inherited full scope and immutable evidence integrity'],'sealed_manifest':pin(manifest),'sealed_manifest_files':len(sealed['artifacts']),'input_pins_verified':len(inputs),'old_unchanged_operator_case_suites_reexecuted':False,'v159_evidence_acquisition_new_device_work':False,'overall_complete':False,'observed_utc':datetime.now(timezone.utc).isoformat(),'verifier':pin(Path(__file__))}
out=O/('checkpoint-profiler-preflight-v159.json' if PRE else 'checkpoint-profiler-postcheck-result-v159p.json');write(out,result)
if not PRE:write(N/'CURRENT-POSTCHECK.json',result)
print(json.dumps({'status':'PASS','preflight':PRE,'gate_count':len(gates),'sealed_files':len(sealed['artifacts']),'indexed_tensors':len(index),'header_occurrences':sum(map(len,occ.values())),'overall_complete':False}))
