"""Independently audit captured current-runtime short SDMA and native cache layout."""
import collections,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 run=ROOT/'runs/20261017T1777-vllm-shared-native-checkpoint-v1';client=json.loads((ROOT/'runs/20261017T1772-vllm-shared-short-matrix-v1/receipt.json').read_text())['matrix_result'];assert client['client_output_pass'];namespace=client['namespace'];source={};all_init=[];short_submit=[];short_complete=[];short_reads=[];diagnostics={};sdk_summary={};runtime={}
 for node in [30,31,43,45]:
  path=run/f'node-{node}.json';raw=path.read_bytes();source[str(path.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest();outer=json.loads(raw);assert outer['rc']==0;obj=json.loads(outer['stdout']);runtime[str(node)]=obj['container_id'];assert obj['/health']['status']==200 and obj['container_state']['Running']
  events=[]
  for archive in obj['persistent_audit'].values():
   data=''.join(json.dumps(row,sort_keys=True)+'\n' for row in archive['rows']).encode();assert len(data)==archive['bytes'] and hashlib.sha256(data).hexdigest()==archive['sha256'];events.extend(archive['rows'])
  initialized={x['pid']:x for x in events if x['event']=='initialize_complete'};assert len(initialized)==16 and all(x['rc']==0 and x['protocol']=='SDMA' and x['protocol_value']==1 for x in initialized.values());all_init.extend(initialized.values())
  selected_pids=set()
  for event in events:
   ids=event.get('request_ids',[])
   if not any(namespace in rid for rid in ids):continue
   assert node in [43,45] and len(ids)==1
   event=event|{'node':node};rank=int(re.search(r'-d(\d+)-D',ids[0]).group(1));assert (node==43)==(rank<8);event['D_DP']=rank
   if event['event']=='model_state_read_submit':
    assert event['ranges']==event['unique_ranges']==180 and event['payload_bytes_requested']==49479680
    specs={x['class']:x for x in event['state_specs']};attention=specs['AscendMLAAttentionSpec'];indexer=specs['AscendSFAIndexerCacheSpec'];assert len(attention['layers'])==79 and {int(re.search(r'layers\.(\d+)',name).group(1)) for name in attention['layers']}==set(range(79));assert len(indexer['layers'])==22 and any('.78.' in x for x in indexer['layers']);short_submit.append(event)
   elif event['event']=='model_state_read_complete':
    assert event['rc']==0 and event['payload_bytes_completed']==49479680;short_complete.append(event);selected_pids.add(event['pid'])
  for event in events:
   if event['event']=='read_complete' and event['pid'] in selected_pids:
    assert event['rc']==0 and event['payload_bytes_completed']==event['payload_bytes_requested'];short_reads.append(event)
  if node in [43,45]:
   stats={}
   for item in obj['native_SDK_small_file_snapshots']:
    assert item['rc']==0
    if item['container_id']!=obj['container_id']:continue
    for f in item['files']:
     assert f['complete_line_ending'] and hashlib.sha256(f['text'].encode()).hexdigest()==f['sha256']
     pid=int(re.search(r'ptracer_(\d+)\.dat$',f['path']).group(1));assert pid in initialized or pid==7
     if pid not in initialized:continue
     values=[]
     for line in f['text'].splitlines():
      a=line.split()
      if len(a)>8 and a[2]=='TP_HYBM_SDMA_BATCH_GD_TO_LD':values.append([int(x) for x in a[3:7]])
     assert values and all(x[0]==x[1] and x[2]==0 for x in values) and values[-1][3]==0
     stats[str(pid)]={'intervals':len(values),'BEGIN':sum(x[0] for x in values),'GOOD_END':sum(x[1] for x in values),'BAD_END':sum(x[2] for x in values),'last_ON_FLY':values[-1][3]}
   assert len(stats)==16;sdk_summary[str(node)]={'workers':16,'native_SDMA_batch_intervals':sum(x['intervals'] for x in stats.values()),'GOOD_END':sum(x['GOOD_END'] for x in stats.values()),'BAD_END':0,'last_ON_FLY_each':0}
  errors=[x for stream in obj['logs'].values() for x in stream['error_lines']];assert all('smem_trans_entry.cpp:' in x and ('QueryRole' in x or 'OnAddSlices' in x) for x in errors);diagnostics[str(node)]={'native_error_line_count':len(errors),'preserved_error_lines':errors,'interpretation':'Native connection/rank diagnostic rows retained. Every audited short SDMA completion and output succeeded; exact internal retry cause not established.'}
 assert len(all_init)==64 and len(short_submit)==len(short_complete)==32
 for rank in range(16):
  rows=[x for x in short_complete if x['D_DP']==rank];assert len(rows)==2 and len({x['pid'] for x in rows})==2
 assert sum(x['GOOD_END'] for x in sdk_summary.values())==36 # short32 + first long cold/warm4, all explicit actual HTTP; no startup read attribution.
 result={'project_id':'glm53','framework':'vllm','shared_run_id':'20261017T1766-vllm-shared-pool-v4','runtime_CIDs':runtime,'pass':True,'qualification':'CURRENT_SHORT_ALL16_D_DP_NATIVE_SDMA_AND_REQUIRED_NATIVE_LAYOUT_COMPLETE','short_namespace':namespace,'short_backend_HTTP_calls':32,'short_worker_SDMA_completions':32,'short_SDMA_descriptor_bytes':sum(x['payload_bytes_completed'] for x in short_complete),'physical_wire_bytes':None,'SDMA_initialized_workers':64,'native_state_layout':{'target_layers':78,'draft_layers':1,'MLA_K_and_V_allocations':158,'native_indexer_allocations':22,'ranges_per_short_worker':180,'draft_layer78_in_both_specs':True,'scope':'Native vLLM required cache layout. SGLang auxiliary buffer representation is framework-specific.'},'SDK_native_batch_counters_including_first_long_attempt':sdk_summary,'preserved_native_diagnostics':diagnostics,'long_cache_MTP_graph_qualification':'PENDING_NEW_LONG_AND_POSTAUDIT','formal_requests':0,'baseline_accepted':False,'source_evidence_sha256':source}
 (ROOT/'results/vllm-shared-short-native-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['preserved_native_diagnostics','source_evidence_sha256']},indent=2))
if __name__=='__main__':main()
