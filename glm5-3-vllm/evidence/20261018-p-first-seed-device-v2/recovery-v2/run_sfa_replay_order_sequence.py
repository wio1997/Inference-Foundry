from pathlib import Path
import json,re,subprocess,sys,time,urllib.request
repo=Path('/Users/wio/work/Inference-Foundry-execution-plan');r=repo/'glm5-3-vllm/evidence/20261018-p-first-seed-device-v2/recovery-v2';root='/data/wio/glm5-3-vllm-execution/20261018-state-tail/sfa-replay-order-live1-20261018/node33';d_cid='e202811aa3535d277385bb4a401ada4cf0c6fcfec5536a3adae2b6724d53210f';p_cid='e812d51531c504f8c7a036671499ad4b8fcf45ca3741a43dd3ef22a846a10db7';p_root=root.replace('node33','node32');base=[sys.executable,str(repo/'glm5-3-vllm/tools/collect_node.py'),'--control','/tmp/glm53-active-20261018.sock']
def collect(script,label,out,args=(),timeout=600,node=33):
 dest=r/out;existing=list(dest.glob('*/metrics.json')) if dest.exists() else []
 if existing:
  assert len(existing)==1;x=json.loads(existing[0].read_text());assert not x.get('failure');return x
 q=subprocess.run(base+['--node',str(node),'--script',str(script),'--label',label,'--output',str(dest),'--timeout',str(timeout)]+['--script-arg='+a for a in args],cwd=repo,capture_output=True,text=True,timeout=timeout+180);(r/(out+'-collector.txt')).write_text(q.stdout+q.stderr);assert q.returncode==0,q.stdout+q.stderr;return json.loads(next(dest.glob('*/metrics.json')).read_text())
t=time.monotonic()
while not (r/'sfa-replay-order-startup-outcome.json').exists():
 assert time.monotonic()-t<1100;time.sleep(10)
ready=json.loads((r/'sfa-replay-order-startup-outcome.json').read_text());assert ready['both_ready'];print(json.dumps({'phase':'both_ready','elapsed':{k:v['elapsed'] for k,v in ready['nodes'].items()}}),flush=True)
def run(phase):
 x=collect(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/validate_sfa_replay_order.py','sfa_replay_order_'+phase.replace('-','_')+'_v1','sfa-replay-order-'+phase,('--execute','--phase',phase,'--run-dir',root,'--expected-D-CID',d_cid,'--cases',root+'/prefill-padding-cases-v1.json'),timeout=1200 if phase=='c4' else 600)
 assert x['completed'] and x['numeric_all_pass'],(phase,x.get('failure'),[a['case_id'] for a in x['rows'] if not a.get('numeric_pass')]);print(json.dumps({'phase':phase,'rows':len(x['rows']),'numeric_all':True,'summary':x['summary'] if phase in ['c1','c4'] else None}),flush=True);return x
functional=run('functional');paths=collect(r/'read_sfa_replay_paths.py','sfa_replay_order_paths_v1','sfa-replay-order-paths',timeout=60);assert paths['ranks']==list(range(16)) and not paths['marker_present'];proof=[]
for row in functional['rows']:
 prefix=row['chunks'][0]['id'];joined=[x for x in paths['rows'] if any(rid==prefix or rid.startswith(prefix+'-') for rid in x['request_ids'])];assert sorted(set(x['tp_rank'] for x in joined))==list(range(16)),(prefix,len(joined),paths['rows'][:1]);counts={rank:0 for rank in range(16)}
 for x in joined:
  assert x['role_eligible'] and len(x['calls'])==1,x
  call=x['calls'][0];assert call['eligible'] and call['static_sfa'] and call['host_update_tasks']==0 and call['captured_entry_present'],call
  assert call['selected']==(row['mode']=='adaptive') and call['requested']==(row['mode']=='adaptive'),call
  counts[x['tp_rank']]+=1
 assert len(set(counts.values()))==1
 proof.append({'case':row['case_id'],'mode':row['mode'],'D_response_prefix':prefix,'all16':True,'target_replays_per_rank':counts[0],'selected_native_static_replay_without_hostwait':row['mode']=='adaptive'})
(r/'sfa-replay-order-path-summary-v1.json').write_text(json.dumps({'all16_joined':True,'same_native_graph_and_static_SFA_no_hosttasks':True,'rows':proof,'scope':'Actual cached-graph/guard calls and unchanged source replay path; not GPU elapsed time.'},indent=2)+'\n');print(json.dumps({'phase':'all16_static_SFA_paths','passed':True}),flush=True)
quality=run('quality')
def prefix_metrics(phase):
 return collect(r/'read_sfa_prefix_metrics.py','sfa_replay_prefix_'+phase+'_v1','sfa-replay-order-prefix-'+phase,timeout=60)
before=prefix_metrics('before');cached=run('cached');after=prefix_metrics('after');assert any(v>before.get(k,0) for k,v in after.items() if 'hits' in k),(before,after)
(r/'sfa-replay-order-cache-summary-v1.json').write_text(json.dumps({'P_prefix_metrics_before':before,'P_prefix_metrics_after':after,'actual_aggregate_cache_hits_increased':True,'whole16_complete_numeric':True,'scope':'Native prefix cache aggregate counters; no perrequest computed-token claim.'},indent=2)+'\n')
proxy=collect(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/start_proxy.py','sfa_replay_order_official_proxy_v1','sfa-replay-order-official-proxy',('--run-dir',p_root,'--expected-P-CID',p_cid),timeout=100,node=32);assert proxy['ready'];print(json.dumps({'phase':'official_proxy','CID':proxy['CID'],'source':proxy['source_sha256']}),flush=True)
run('c4-functional');collect(r/'read_sfa_replay_paths.py','sfa_replay_order_paths_v2','sfa-replay-order-paths-v2',timeout=60)
for node in [32,33]:
 x=collect(r/'preflight_sfa_replay_order.py','sfa_replay_order_preflight_'+str(node)+'_v1','sfa-replay-order-preflight-'+str(node),('--disable-D-trace',) if node==33 else (),timeout=60,node=node);print(json.dumps({'phase':'timing_preflight','node':node,'remaining':x['remaining_s']}),flush=True)
run('c1');run('c4');(r/'sfa-replay-order-sequence-outcome.json').write_text(json.dumps({'completed':True,'native_marker_restored':True,'no_general_quality_acceptance':True},indent=2)+'\n');print(json.dumps({'sequence_completed':True,'default_restored':'native'}),flush=True)
