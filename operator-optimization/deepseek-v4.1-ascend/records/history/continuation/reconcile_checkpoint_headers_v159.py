"""Local-only whole-index reconciliation,including six duplicate Engram names."""
from pathlib import Path
from collections import Counter, defaultdict
import base64, hashlib, json
from finalize_origin_coverage_v9 import N, O, read, write
from analyze_evidence_batches_v155 import pin


def unique(pairs):
    result={}
    for key,value in pairs:
        assert key not in result,'Repeated JSON object key '+key
        result[key]=value
    return result


def main():
    output=O/'checkpoint-all-header-index-v159.json';assert not output.exists()
    previous=read(O/'checkpoint-operator-metadata-v159.json')['models'][0];index_meta=previous['index'];index=json.loads(Path(index_meta['local_path']).read_text(),object_pairs_hook=unique);weight_map=index['weight_map'];raw=N/'checkpoint-all-header-raw-v159.json';receipt=read(N/'checkpoint-all-header-receipt-v159.json');assert receipt['returncode']==0 and pin(raw)['sha256']==receipt['stdout_sha256'];received=json.loads(raw.read_text());old={x['shard']:x for x in previous['headers']};headers={name:{k:v for k,v in row.items()if k!='selected_tensors'}for name,row in old.items()}
    for source in received['headers']:
        row={k:v for k,v in source.items()if k!='payload'};rel=Path(row['shard']);assert not rel.is_absolute()and'..'not in rel.parts;p=N/'checkpoint-operator-cache-v159/headers/target'/(row['shard']+'.header.json');data=p.read_bytes();assert len(data)==row['header_bytes']and hashlib.sha256(data).hexdigest()==row['header_sha256']and data==base64.b64decode(source['payload']);row['header_local_path']=str(p);headers[row['shard']]=row
    assert set(headers)==set(weight_map.values())
    bodies={};occurrences=defaultdict(list);all_types=Counter();all_bytes=0
    for shard,row in headers.items():
        data=Path(row['header_local_path']).read_bytes();assert len(data)==row['header_bytes']and hashlib.sha256(data).hexdigest()==row['header_sha256'];body=json.loads(data,object_pairs_hook=unique);bodies[shard]=body
        for name,tensor in body.items():
            if name=='__metadata__':continue
            lo,hi=tensor['data_offsets'];assert 0<=lo<=hi<=row['source_file_bytes']-8-row['header_bytes'];all_types[tensor['dtype']]+=1;all_bytes+=hi-lo;occurrences[name].append(dict(shard=shard,shape=tensor['shape'],dtype=tensor['dtype'],data_offsets=[lo,hi],declared_payload_bytes=hi-lo,index_declares_this_shard=weight_map.get(name)==shard))
    missing=[];indexed_types=Counter();indexed_bytes=0;quant=read(Path(next(x['local_path']for x in previous['metadata']if x['relative']=='quant_model_description.json')));categories=Counter()
    for name,shard in weight_map.items():
        if name not in bodies[shard]:missing.append(dict(name=name,shard=shard));continue
        tensor=bodies[shard][name];indexed_types[tensor['dtype']]+=1;indexed_bytes+=tensor['data_offsets'][1]-tensor['data_offsets'][0];category=quant.get(name,'UNSPECIFIED');categories[category if isinstance(category,str)else'<'+type(category).__name__+'>']+=1
    duplicates=[dict(name=name,index_declared_shard=weight_map.get(name),occurrences=rows,payload_value_equality_proved=False,actual_runtime_loader_winner_proved=False)for name,rows in sorted(occurrences.items())if len(rows)>1]
    nonselected=[dict(name=name,**row)for name,rows in sorted(occurrences.items())for row in rows if not row['index_declares_this_shard']]
    cohort=dict(index_sha256=index_meta['sha256'],headers=[dict(shard=s,sha256=headers[s]['header_sha256'],file_bytes=headers[s]['source_file_bytes'],mtime_ns=headers[s]['source_mtime_ns'])for s in sorted(headers)])
    record=dict(version=159,origin='Currentconfiguredcheckpointheadercohortonly;indexdeclarationnotactualruntime-loaderprecedence.',index_source=index_meta,headers=[headers[x]for x in sorted(headers)],header_files=len(headers),headers_reused_without_remote_rescan=len(old),new_headers_read=len(received['headers']),indexed_tensor_count=len(weight_map),indexed_entries_found=len(weight_map)-len(missing),indexed_entries_missing=missing,physical_header_tensor_occurrences=sum(all_types.values()),unique_header_tensor_names=len(occurrences),duplicate_parameter_names=duplicates,non_index_selected_header_occurrences=nonselected,indexed_dtype_label_counts=dict(indexed_types),all_occurrence_dtype_label_counts=dict(all_types),indexed_quant_description_category_counts=dict(categories),total_index_declared_tensor_payload_bytes=indexed_bytes,total_all_header_occurrence_payload_bytes=all_bytes,full_index_header_crosscheck=not missing,JSON_object_keys_unique=True,metadata_cohort_fingerprint=hashlib.sha256(json.dumps(cohort,sort_keys=True,separators=(',',':')).encode()).hexdigest(),index_unchanged_between_header_batches=True,global_atomic_snapshot_or_historical_loaded_values_attested=False,index_declared_route_is_not_runtime_loader_selection_proof=True,weight_payload_bytes_read=0,full_weight_file_hashes_computed=False,new_device_work=False,overall_complete=False)
    write(output,record)
    summary=read(O/'checkpoint-profiler-summary-v159.json');summary.update(all_checkpoint_headers=len(headers),all_indexed_tensor_metadata=len(weight_map),all_header_dtype_label_counts=dict(indexed_types),full_checkpoint_header_inventory_complete=not missing,header_inventory_is_not_actual_load_inventory=True,duplicate_parameter_names_requiring_load_route_binding=len(duplicates));write(O/'checkpoint-profiler-summary-v159.json',summary)
    inputs=read(O/'checkpoint-profiler-inputs-v159.json');pins={x['path']:x for x in inputs['inputs']}
    for p in[output,N/'checkpoint-all-header-receipt-v159.json',O/'checkpoint-header-local-corrections-v159.json',N/'read_all_checkpoint_headers_v159a_failed.py']+[Path(x['header_local_path'])for x in headers.values()]:pins[str(p)]=pin(p)
    inputs['inputs']=[pins[x]for x in sorted(pins)];write(O/'checkpoint-profiler-inputs-v159.json',inputs)
    print(json.dumps(dict(status='PASS_WITH_EXPLICIT_RUNTIME_SELECTION_GAP',headers=len(headers),reused=len(old),new=len(received['headers']),index_entries=len(weight_map),missing=len(missing),dtype_labels=dict(indexed_types),duplicate_names=len(duplicates),nonselected_occurrences=len(nonselected),weight_payload_bytes_read=0,new_device_work=False)))


if __name__=='__main__':main()
