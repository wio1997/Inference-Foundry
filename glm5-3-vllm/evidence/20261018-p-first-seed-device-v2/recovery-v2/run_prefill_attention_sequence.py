from pathlib import Path
import json,subprocess,sys,time
repo=Path('/Users/wio/work/Inference-Foundry-execution-plan');r=repo/'glm5-3-vllm/evidence/20261018-p-first-seed-device-v2/recovery-v2';root='/data/wio/glm5-3-vllm-execution/20261018-state-tail/prefill-attention-live1-20261018/node32';cid='8df6987f0793978af501a754b5b186a120cf4255169f5eed8835a5defb8b3148'
base=[sys.executable,str(repo/'glm5-3-vllm/tools/collect_node.py'),'--control','/tmp/glm53-active-20261018.sock']
def collect(script,label,out,args=(),timeout=600,node=32):
 dest=r/out;existing=list(dest.glob('*/metrics.json')) if dest.exists() else []
 if existing:
  assert len(existing)==1
  metrics=json.loads(existing[0].read_text());assert not metrics.get('failure')
  return metrics
 argv=base+['--node',str(node),'--script',str(script),'--label',label,'--output',str(dest),'--timeout',str(timeout)]+['--script-arg='+a for a in args]
 q=subprocess.run(argv,cwd=repo,capture_output=True,text=True,timeout=timeout+180)
 (r/(out+'-collector.txt')).write_text(q.stdout+q.stderr);assert q.returncode==0,q.stdout+q.stderr
 metrics=json.loads(next(dest.glob('*/metrics.json')).read_text());return metrics
start=time.monotonic()
while not (r/'prefill-attention-startup-outcome.json').exists():
 assert time.monotonic()-start<1000
 time.sleep(10)
ready=json.loads((r/'prefill-attention-startup-outcome.json').read_text());assert ready['ready'] and ready['cid']==cid
print(json.dumps({'phase':'ready','host_elapsed':ready['elapsed_host_s']}),flush=True)
def run(phase):
 x=collect(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/validate_prefill_padding.py','prefill_attention_'+phase.replace('-','_')+'_v1','prefill-attention-'+phase,('--execute','--phase',phase,'--run-dir',root,'--expected-P-CID',cid,'--probe-kind','attention','--cases',root+'/prefill-padding-cases-v1.json'),timeout=1200 if phase=='c4' else 600)
 assert x['completed'] and x['numeric_all_pass'],(phase,x.get('failure'),[a['case_id'] for a in x['rows'] if not a.get('numeric_pass')])
 print(json.dumps({'phase':phase,'rows':len(x['rows']),'numeric_all':x['numeric_all_pass'],'summary':x['summary'] if phase in ['c1','c4'] else None}),flush=True);return x
functional=run('functional')
paths=collect(r/'read_attention_paths.py','prefill_attention_paths_v1','prefill-attention-paths',timeout=60);assert paths['ranks']==list(range(16)) and not paths['marker_present']
proof=[]
for row in functional['rows']:
 rid=row['P_response']['kv_transfer_params']['remote_request_id'];joined=[x for x in paths['rows'] if rid in x['request_ids']];assert sorted(set(x['tp_rank'] for x in joined))==list(range(16))
 for x in joined:
  assert x['sfa_forward']==78 and sum(x['prolog_rows'].values())==78 and sum(x['o_proj_rows'].values())==78,x
  expected=x['real_tokens'] if x['selected'] else x['graph_tokens'];assert x['prolog_rows']=={str(expected):78} and x['o_proj_rows']=={str(expected):78},x
  assert x['tail_clear']==(78 if x['selected'] else 0),x
  assert x['native_graph_mode']=='PIECEWISE',x
 proof.append({'case':row['case_id'],'mode':row['mode'],'request_id':rid,'ranks':list(range(16)),'work':list({(x['real_tokens'],x['graph_tokens'],x['selected'],tuple(x['prolog_rows'].items()),tuple(x['o_proj_rows'].items()),x['tail_clear']) for x in joined})})
(r/'prefill-attention-path-summary-v1.json').write_text(json.dumps({'all16_joined':True,'native_graph_preserved':True,'actual_native_attention_call_shapes_verified':True,'rows':proof,'scope':'Native Python input shapes/call counters; kernel profile/duration still separate'},indent=2)+'\n')
print(json.dumps({'phase':'all16_attention_shapes','passed':True}),flush=True)
quality=run('quality');cached=run('cached')
proxy=collect(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/start_proxy.py','prefill_attention_official_proxy_v1','prefill-attention-official-proxy',('--run-dir',root,'--expected-P-CID',cid),timeout=100);assert proxy['ready']
print(json.dumps({'phase':'official_proxy','CID':proxy['CID'],'source':proxy['source_sha256']}),flush=True)
run('c4-functional')
paths=collect(r/'read_attention_paths.py','prefill_attention_paths_v2','prefill-attention-paths-v2',timeout=60);assert not paths['marker_present']
cachedproof=[]
for row in cached['rows']:
 rid=row['P_response']['kv_transfer_params']['remote_request_id'];joined=[x for x in paths['rows'] if rid in x['request_ids']];assert sorted(set(x['tp_rank'] for x in joined))==list(range(16))
 cachedproof.append({'case':row['case_id'],'mode':row['mode'],'computed':sorted(set(c for x in joined for c in x['computed_before'])),'real_tokens':sorted(set(x['real_tokens'] for x in joined))})
assert any(any(c>0 for c in x['computed']) for x in cachedproof)
(r/'prefill-attention-cached-summary-v1.json').write_text(json.dumps({'all16_joined':True,'actual_cache_reuse':True,'rows':cachedproof},indent=2)+'\n')
for node in [33,32]:
 x=collect(r/'preflight_prefill_attention.py','prefill_attention_preflight_'+str(node)+'_v1','prefill-attention-preflight-'+str(node),('--disable-P-traces',) if node==32 else (),timeout=60,node=node)
 print(json.dumps({'phase':'timing_preflight','node':node,'remaining':x['remaining_s']}),flush=True)
run('c1');run('c4')
print(json.dumps({'sequence_completed':True,'default_restored':'native, markerabsent','device_profile_pending':True}),flush=True)
