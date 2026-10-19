from pathlib import Path
import importlib.util,json,time
repo=Path('/Users/wio/work/Inference-Foundry-execution-plan');r=Path(__file__).resolve().parent
root='/data/wio/glm5-3-vllm-execution/20261018-state-tail/moe-cumsum-i32-live1-20261019/node33';cid='416f0dc10e34a115e2537cb62e46e2eb98d30825642ab4ba36d098c072cf11f5'
review=json.loads((r/'cumsum-i32-quality-reviewed.json').read_text());assert review['proceed_finite_formal'] and review['all7_old_and_current_candidate_sequences_observed_in_native_or_replica']
start=time.monotonic()
while not (r/'cumsum-i32-P-live2-startup-outcome.json').exists():
 assert time.monotonic()-start<1350;time.sleep(10)
assert json.loads((r/'cumsum-i32-P-live2-startup-outcome.json').read_text())['both_ready']
spec=importlib.util.spec_from_file_location('collect_base',r/'run_mtp_moe_live2_functional.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
x=m.collect(r/'prepare_cumsum_i32_P_live2_smoke.py','cumsum_i32_P_live2_prepare_smoke_v1','cumsum-i32-P-live2-prepare-smoke',timeout=40);assert x['completed']
proxy=m.collect(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/start_proxy.py','cumsum_i32_P_live2_proxy_v1','cumsum-i32-P-live2-proxy',('--run-dir','/data/wio/glm5-3-vllm-execution/20261018-state-tail/moe-cumsum-i32-live2-20261019/node32','--expected-P-CID','076b19aeee16e31ac9ec10f0f3a73adf1e04c41cdf0abf1248b774affdba8742','--minimum-lifetime-seconds','900'),timeout=100,node=32);assert proxy['ready'];print(json.dumps({'phase':'fresh_P_official_proxy','CID':proxy['CID']}),flush=True)
def run(phase,extra=(),smoke=False):
 name='cumsum-i32-'+('P-live2-' if smoke else '')+phase
 args=('--execute','--phase',phase,'--probe-kind','moe_cumsum_i32','--run-dir',root,'--expected-D-CID',cid,'--cases',root+'/prefill-padding-cases-v1.json','--run-tag','p-live2')+extra
 x=m.collect(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/validate_sfa_replay_order.py','cumsum_i32_P_live2_'+phase.replace('-','_')+'_v1',name,args,timeout=1000)
 assert x['completed'] and x['numeric_all_pass'],x.get('failure');print(json.dumps({'phase':phase,'rows':len(x['rows']),'numeric_all':True}),flush=True);return x
smoke=run('c4-functional',smoke=True)
paths=m.collect(r/'read_cumsum_i32_paths.py','cumsum_i32_P_live2_paths_v1','cumsum-i32-P-live2-paths',timeout=90);assert paths['observed_ranks']==list(range(16)) and paths['all16_paths_verified'] and paths['MTP_native_in_both_modes'] and paths['all16_native_GMM2_restore_proved'] and paths['Rope_enabled_for_both_modes']
for node in [32,33]:
 x=m.collect(r/'preflight_cumsum_i32_live2.py','cumsum_i32_P_live2_formal_preflight_'+str(node)+'_v1','cumsum-i32-P-live2-formal-preflight-'+str(node),('--disable-D-trace',) if node==33 else (),node=node,timeout=90);assert x['remaining_s']>600;print(json.dumps({'phase':'formal_preflight','node':node,'remaining_s':x['remaining_s']}),flush=True)
c1=run('c1',('--retain-native-counters',))
c4=run('c4',('--rotate-cases','--c4-cycles','3','--retain-native-counters'))
assert len(c4['blocks'])==12 and all(z['native_counter_boundary']['delta']['33']['vllm:generation_tokens_total']==3072 for z in c4['blocks'])
(r/'cumsum-i32-formal-outcome.json').write_text(json.dumps({'completed':True,'native_marker_restored':True,'fresh_P_CID':'076b19aeee16e31ac9ec10f0f3a73adf1e04c41cdf0abf1248b774affdba8742','retained_D_CID':cid,'C1':c1['summary'],'C4':c4['summary'],'mode_owned_native_counters_each_block':True,'scope':'Observer-off finite C1/balanced C4, same refreshed P/native KV path in both arms; finite quality reviewed from retained D with28 bounded native controls; fresh PD smoke8. Per-mode native draft/accepted/generated counters outside timers. Not general quality or maximum SLO capacity.'},indent=2)+'\n');print(json.dumps({'phase':'formal_completed','C1':c1['summary'],'C4':c4['summary']}),flush=True)
