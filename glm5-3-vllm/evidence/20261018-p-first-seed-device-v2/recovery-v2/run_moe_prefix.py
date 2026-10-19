from pathlib import Path
import importlib.util,json,subprocess,sys,time
repo=Path('/Users/wio/work/Inference-Foundry-execution-plan');r=Path(__file__).resolve().parent
root='/data/wio/glm5-3-vllm-execution/20261018-state-tail/moe-prefix-histogram-live1-20261019/node33';cid='619da4b9c8e7a3d78d1ca4ff5c62e4bce5a57de72bafdbd96318137ba13810a8'
start=time.monotonic()
while not (r/'moe-prefix-startup-outcome.json').exists():
 assert time.monotonic()-start<1300;time.sleep(10)
assert json.loads((r/'moe-prefix-startup-outcome.json').read_text())['both_ready']
spec=importlib.util.spec_from_file_location('collect_base',r/'run_mtp_moe_live2_functional.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def run(phase,extra=()):
 name='moe-prefix-histogram-'+phase
 args=('--execute','--phase',phase,'--probe-kind','moe_prefix_histogram','--run-dir',root,'--expected-D-CID',cid,'--cases',root+'/prefill-padding-cases-v1.json')+extra
 x=m.collect(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/validate_sfa_replay_order.py','moe_prefix_histogram_'+phase.replace('-','_')+'_v1',name,args,timeout=1000)
 assert x['completed'] and x['numeric_all_pass'],x.get('failure');print(json.dumps({'phase':phase,'rows':len(x['rows']),'numeric_all':True}),flush=True);return x

def check_paths(results,label,name):
 paths=m.collect(r/'read_moe_prefix_paths.py',label,name,timeout=90)
 assert paths['observed_ranks']==list(range(16)) and paths['all16_paths_verified'] and not paths['marker_present'],paths
 assert paths['Rope_enabled_for_both_modes']
 (r/(name+'-summary.json')).write_text(json.dumps(paths,indent=2)+'\n')
 print(json.dumps({'phase':name,'all16_actual_target_and_MTP_paths':True,'candidate_shapes':sorted(set(n for z in paths['per_rank'] for n in z['selected_graph_tokens']))}),flush=True)

f=run('functional');check_paths(f,'moe_prefix_histogram_paths_v1','moe-prefix-histogram-paths')
proxy=m.collect(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/start_proxy.py','moe_prefix_histogram_proxy_v1','moe-prefix-histogram-proxy',('--run-dir','/data/wio/glm5-3-vllm-execution/20261018-state-tail/moe-prefix-histogram-live1-20261019/node32','--expected-P-CID','d685575688634c0f6e8878124ceae28b0a48b0def5e73cbafffc5f8ef355d3e4'),node=32,timeout=100);assert proxy['ready'];print(json.dumps({'phase':'official_proxy','CID':proxy['CID']}),flush=True)
c4f=run('c4-functional');check_paths(c4f,'moe_prefix_histogram_paths_v2','moe-prefix-histogram-paths-v2')
for node in [32,33]:
 x=m.collect(r/'preflight_moe_prefix.py','moe_prefix_histogram_preflight_'+str(node)+'_v1','moe-prefix-histogram-preflight-'+str(node),('--disable-D-trace',) if node==33 else (),node=node,timeout=90);print(json.dumps({'phase':'observer_off_preflight','node':node,'remaining_s':x['remaining_s']}),flush=True)
quality=run('quality');assert len(quality['sequence_comparisons'])==26
same=sum(z['all_same'] for z in quality['sequence_comparisons'].values());print(json.dumps({'phase':'quality_full64_comparison','equal':same,'cases':26}),flush=True)
cached=run('cached')
(r/'moe-prefix-correctness-outcome.json').write_text(json.dumps({'completed':True,'functional16_all_numeric':True,'C4_functional8_all_numeric':True,'all16_actual_native_and_prefix_target_and_MTP_paths':True,'quality26x2_all_numeric':True,'quality_full64_equal_cases':same,'cached16_all_numeric':True,'scope':'Finite ordinary correctness and actual graph/dispatcher selection. RoPE continuous table remains enabled in both prefix modes; no general quality/capacity claim.'},indent=2)+'\n')
for node in [32,33]:
 x=m.collect(r/'preflight_moe_prefix.py','moe_prefix_histogram_formal_preflight_'+str(node)+'_v1','moe-prefix-histogram-formal-preflight-'+str(node),('--disable-D-trace',) if node==33 else (),node=node,timeout=90);assert x['remaining_s']>900
c1=run('c1',('--retain-native-counters',))
c4=run('c4',('--rotate-cases','--c4-cycles','3','--run-tag','balanced1','--retain-native-counters'))
assert len(c4['blocks'])==12 and all(z['native_counter_boundary']['delta']['33']['vllm:generation_tokens_total']==3072 for z in c4['blocks'])
(r/'moe-prefix-formal-outcome.json').write_text(json.dumps({'completed':True,'native_marker_restored':True,'C1':c1['summary'],'C4':c4['summary'],'mode_owned_native_counters_each_block':True,'scope':'Observer-off finite C1/balanced C4, same P/native KV path, per-mode native draft/accepted/generated counters sampled and settled only outside request/block timers. Not maximum SLO capacity.'},indent=2)+'\n')
print(json.dumps({'phase':'formal_completed','C1':c1['summary'],'C4':c4['summary']}),flush=True)
