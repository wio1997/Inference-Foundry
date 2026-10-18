from pathlib import Path
import importlib.util,json,subprocess,sys,time
repo=Path('/Users/wio/work/Inference-Foundry-execution-plan');r=Path(__file__).resolve().parent
root='/data/wio/glm5-3-vllm-execution/20261018-state-tail/rope-cache-layout-live1-20261019/node33';cid='246f4f1090349ae9b8018af114460537df89c7073547190f5d0949326edc2046'
start=time.monotonic()
while not (r/'rope-cache-layout-startup-outcome.json').exists():
 assert time.monotonic()-start<1100;time.sleep(10)
assert json.loads((r/'rope-cache-layout-startup-outcome.json').read_text())['both_ready']
spec=importlib.util.spec_from_file_location('collect_base',r/'run_mtp_moe_live2_functional.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def run(phase,extra=()):
 name='rope-cache-layout-'+phase
 args=('--execute','--phase',phase,'--probe-kind','rope_cache_layout','--run-dir',root,'--expected-D-CID',cid,'--cases',root+'/prefill-padding-cases-v1.json')+extra
 x=m.collect(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/validate_sfa_replay_order.py','rope_cache_layout_'+phase.replace('-','_')+'_v1',name,args,timeout=1000)
 assert x['completed'] and x['numeric_all_pass'],x.get('failure');print(json.dumps({'phase':phase,'rows':len(x['rows']),'numeric_all':True}),flush=True);return x

def check_paths(results,label,name):
 paths=m.collect(r/'read_rope_cache_paths.py',label,name,timeout=90)
 assert paths['ranks']==list(range(16)) and not paths['marker_present']
 proof=[]
 for row in results['rows']:
  prefix=row['chunks'][0]['id'];joined=[z for z in paths['rows'] if z['request_id']==prefix or z['request_id'].startswith(prefix+'-')];assert sorted(set(z['rank'] for z in joined))==list(range(16)),(prefix,len(joined))
  for z in joined:
   requested=row['mode']=='adaptive';assert z['calls']>0 and z['requested_values']==[requested] and not z['fallbacks']
   assert z['all_full_cache_bitwise_equal'] and z['all_persistent_addresses_unchanged'] and z['cache_bytes']==[268435456] and z['native_layouts']==[[128,1]] and z['cached_layouts']==[[64,1]],z
   assert (z['selected']==z['calls']) if requested else z['selected']==0
  proof.append({'case':row['case_id'],'mode':row['mode'],'all16_real_getter_calls':True,'all_selected_if_requested':True,'all_persistent_addresses_unchanged':True,'full_cache_bitwise_equal':True,'calls_by_rank':{z['rank']:z['calls'] for z in joined},'Ns':sorted(set(n for z in joined for n in z['Ns']))})
 (r/(name+'-summary.json')).write_text(json.dumps({'completed':True,'rows':proof,'source_files':paths['files'],'scope':paths['scope']},indent=2)+'\n')
 print(json.dumps({'phase':name,'all16_actual_path':True}),flush=True)

f=run('functional');check_paths(f,'rope_cache_layout_paths_v1','rope-cache-layout-paths')
proxy=m.collect(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/start_proxy.py','rope_cache_layout_proxy_v1','rope-cache-layout-proxy',('--run-dir','/data/wio/glm5-3-vllm-execution/20261018-state-tail/rope-cache-layout-live1-20261019/node32','--expected-P-CID','0c0487ac798214c855ba2bc9886b86c802fdca215fe362dd2384a925b063740b'),node=32,timeout=100);assert proxy['ready'];print(json.dumps({'phase':'official_proxy','CID':proxy['CID']}),flush=True)
c4f=run('c4-functional');check_paths(c4f,'rope_cache_layout_paths_v2','rope-cache-layout-paths-v2')
for node in [32,33]:
 x=m.collect(r/'preflight_rope_cache_layout.py','rope_cache_layout_preflight_'+str(node)+'_v1','rope-cache-layout-preflight-'+str(node),('--disable-D-trace',) if node==33 else (),node=node,timeout=90);print(json.dumps({'phase':'observer_off_preflight','node':node,'remaining_s':x['remaining_s']}),flush=True)
quality=run('quality');assert len(quality['sequence_comparisons'])==26
same=sum(z['all_same'] for z in quality['sequence_comparisons'].values());print(json.dumps({'phase':'quality_full64_comparison','equal':same,'cases':26}),flush=True)
cached=run('cached')
(r/'rope-cache-layout-correctness-outcome.json').write_text(json.dumps({'completed':True,'functional16_all_numeric':True,'C4_functional8_all_numeric':True,'all16_actual_native_getter_and_cached_selection':True,'all16_full_cache_bitwise_equal':True,'all16_persistent_output_addresses_unchanged':True,'quality26x2_all_numeric':True,'quality_full64_equal_cases':same,'cached16_all_numeric':True,'scope':'Finite correctness, table bits and pointer/ownership proof; full sequence variations if any retained, not general quality or service acceptance.'},indent=2)+'\n')
for node in [32,33]:
 x=m.collect(r/'preflight_rope_cache_layout.py','rope_cache_layout_formal_preflight_'+str(node)+'_v1','rope-cache-layout-formal-preflight-'+str(node),('--disable-D-trace',) if node==33 else (),node=node,timeout=90);assert x['remaining_s']>900
c1=run('c1',('--retain-native-counters',))
c4=run('c4',('--rotate-cases','--c4-cycles','3','--run-tag','balanced1','--retain-native-counters'))
assert len(c4['blocks'])==12 and all(z['native_counter_boundary']['delta']['33']['vllm:generation_tokens_total']==3072 for z in c4['blocks'])
(r/'rope-cache-layout-formal-outcome.json').write_text(json.dumps({'completed':True,'native_marker_restored':True,'C1':c1['summary'],'C4':c4['summary'],'mode_owned_native_counters_each_block':True,'scope':'Observer-off finite C1/balanced C4, same P/native KV path, per-mode native draft/accepted/generated counters sampled and settled only outside request/block timers. Not maximum SLO capacity.'},indent=2)+'\n')
print(json.dumps({'phase':'formal_completed','C1':c1['summary'],'C4':c4['summary']}),flush=True)
