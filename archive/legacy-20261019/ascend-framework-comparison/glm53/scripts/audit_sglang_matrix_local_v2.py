"""Independent controller audit of captured real native PD evidence; no resource calls."""
import collections,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];run=ROOT/'runs/20261017T1500-sglang-v5-matrix-state-v1';objects={};register={};writes=[];totalinit=0
for n in [34,35,36,41]:
 p=run/f'node-{n}.json';outer=json.loads(p.read_text());assert outer['rc']==0;o=json.loads(outer['stdout']);objects[n]=o;assert o['model']['state']['Running']
 for name,a in o['persistent_audit'].items():
  raw=''.join(json.dumps(r,sort_keys=True,default=str)+'\n' for r in a['rows']).encode();assert len(raw)==a['bytes'] and hashlib.sha256(raw).hexdigest()==a['sha256'],name
  for r in a['rows']:
   if r['event']=='initialize_complete':assert r['protocol']=='SDMA' and r['protocol_value']==1 and r['rc']==0;totalinit+=1
   if r['event']=='model_state_registered':
    m=r['metadata'];assert m['hidden_kv_layers']==78 and m['draft_kv_layers']==1 and m['kv_cache_dtype_str']=='bfloat16' and m['kv_buf_groups']==3 and m['page_size']==128
    assert collections.Counter(m['kv_item_lens'])=={131072:79,16384:79,32768:79};assert sum(m['aux_item_lens'])==22016;register[r['session']]={'node':n,'P_or_D':'P' if n in [34,35] else 'D','global_DP':m['system_dp_rank'],'engine_rank':m['engine_rank'],'gpu_id':m['gpu_id']}
   if r['event']=='write_complete':
    assert r['rc']==0 and r['payload_bytes_completed']==r['payload_bytes_requested'] and r['physical_link_bytes'] is None;writes.append({'node':n,**r})
assert totalinit==64 and len(register)==64 and len(writes)==64
coverage=collections.defaultdict(lambda:collections.Counter());records=[]
for r in writes:
 source=register[r['local_session']];dest=register[r['remote_session']];assert source['P_or_D']=='P' and dest['P_or_D']=='D';assert source['global_DP']==dest['global_DP']%4
 kind=r['context']['kind'];assert kind in ['send_kvcache','send_aux'];size=r['payload_bytes_completed'];assert size==(56950784 if kind=='send_kvcache' else 22016)
 assert r['ranges']==(237 if kind=='send_kvcache' else 11)
 coverage[dest['global_DP']][kind]+=1;records.append({'P_global_DP':source['global_DP'],'D_global_DP':dest['global_DP'],'D_gpu_id':dest['gpu_id'],'D_node':dest['node'],'kind':kind,'descriptor_bytes':size})
assert set(coverage)==set(range(16)) and all(counts=={'send_kvcache':2,'send_aux':2} for counts in coverage.values())
assert sum(r['payload_bytes_completed'] for r in writes)==1823129600
bykind={kind:sum(r['payload_bytes_completed'] for r in writes if r['context']['kind']==kind) for kind in ['send_kvcache','send_aux']}
assert bykind=={'send_kvcache':1822425088,'send_aux':704512}
report={'project_id':'glm53','status':'V5_HOST10_ALL_P4_D16_SHORT_MODEL_AND_NATIVE_SDMA_ALL_STATE_BYTES_PASSED','client_run':'runs/20261017T1490-sglang-pd-matrix-v4','audit_run':str(run.relative_to(ROOT)),'SDK_SDMA_initializations':64,'SDMA_write_completions':64,'all_native_write_rc':[0],'descriptor_bytes_by_state':bykind,'total_descriptor_bytes':1823129600,'physical_link_bytes':None,'D_global_DP_covered':sorted(coverage),'D_workers_covered':32,'P_global_DP_covered':[0,1,2,3],'state_layout':'BF16 K79/V79/indexer79 includes78target+1draft; all3groups+11auxitems. No C8 scale state.','per_request_D_workers':2,'per_D_worker_main_payload_bytes':56950784,'per_D_worker_aux_bytes':22016,'raw_node_sha256':{str(n):hashlib.sha256((run/f'node-{n}.json').read_bytes()).hexdigest() for n in objects},'formal_requests':0,'baseline_accepted':False,'limitations':['Short512inputs/16outputs only; same80Kretrieval/MCknownkeyreadback qualification remainsrequired.','NativeDeepEPNORMAL eager path, no graphperformanceclaim.','SDKdescriptor bytes are not physical link counters.']}
(ROOT/'results/sglang-pd-sdma-matrix-v2.json').write_text(json.dumps(report,indent=2)+'\n')
p=ROOT/'results/sglang-pd-matrix-v4.json';o=json.loads(p.read_text());o['status']='V5_HOST10_ALL_P4_D16_512_TOKEN_NATIVE_PD_MODEL_OUTPUT_AND_SDMA_ALL_STATE_BYTES_PASSED';o['server_audit']='results/sglang-pd-sdma-matrix-v2.json';p.write_text(json.dumps(o,indent=2)+'\n')
print(json.dumps(report,indent=2))
