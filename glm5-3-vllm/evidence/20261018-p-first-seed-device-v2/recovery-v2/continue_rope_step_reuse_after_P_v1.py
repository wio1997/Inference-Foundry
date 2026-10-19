from pathlib import Path
import datetime,hashlib,importlib.util,json,subprocess,sys,time
r=Path(__file__).resolve().parent;repo=r.parents[3]
s=importlib.util.spec_from_file_location('transport',r/'run_prefix_count_service_v1.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
start=time.monotonic()
while not(r/'rope-step-reuse-P-startup-outcome-v1.json').exists():
 assert time.monotonic()-start<1500;time.sleep(10)
assert json.loads((r/'rope-step-reuse-P-startup-outcome-v1.json').read_text())['both_ready']
binding=json.loads((r/'rope-step-reuse-formal-CID-binding-v2.json').read_text())
P_root='/data/wio/glm5-3-vllm-execution/20261018-state-tail/rope-step-reuse-p-live1-20261019/node32'
D_root='/data/wio/glm5-3-vllm-execution/20261018-state-tail/rope-step-reuse-live1-20261019/node33'
proxy=m.collect(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/start_proxy.py','rope_step_reuse_P_official_proxy_v1','rope-step-reuse-P-official-proxy-v1',('--run-dir',P_root,'--expected-P-CID',binding['P_CID'],'--minimum-lifetime-seconds','900'),timeout=100,node=32)
assert proxy['ready'];print(json.dumps(dict(phase='fresh_P_official_proxy',CID=proxy['CID'])),flush=True)
m.collect(r/'enable_reuse_observer_P_refresh_v1.py','rope_step_reuse_P_refresh_observer_v1','rope-step-reuse-P-refresh-observer-v1',timeout=60)
try:
 smoke=m.collect(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/validate_sfa_replay_order.py','rope_step_reuse_P_refresh_smoke_v1','rope-step-reuse-P-refresh-smoke-v1',('--execute','--phase','c4-functional','--probe-kind','rope_step_reuse','--run-dir',D_root,'--expected-D-CID',binding['D_CID'],'--cases',D_root+'/prefill-padding-cases-v1.json','--run-tag','p-refresh-v1'),timeout=300)
 assert smoke['completed'] and smoke['numeric_all_pass'] and len(smoke['rows'])==8
 paths=m.collect(r/'read_rope_step_reuse_paths_v1.py','rope_step_reuse_P_refresh_paths_v1','rope-step-reuse-P-refresh-paths-v1',timeout=90);assert paths['all16']
 before=json.loads(next((r/'rope-step-reuse-paths-v1').glob('*/metrics.json')).read_text());old={v['rank']:v['selected'] for v in before['rows']}
 assert all(v['selected']>old[v['rank']] for v in paths['rows'])
finally:
 m.collect(r/'disable_rope_step_reuse_observer_v1.py','rope_step_reuse_P_refresh_observer_off_v1','rope-step-reuse-P-refresh-observer-off-v1',timeout=60)
out=dict(fresh_PD_smoke8=True,all16_new_reuse_and_owned_copy_bitwise=True,P_CID=binding['P_CID'],D_CID=binding['D_CID'],scope='Same immutable P config after its watchdog, D retained. Original quality52/cached16 retained; fresh both-mode concurrent PD smoke before new-pair finite controls and service timing.')
(r/'rope-step-reuse-P-refresh-outcome-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
control=m.collect(r/'control_rope_step_reuse_quality_v1.py','rope_step_reuse_quality_controls_after_P_v1','rope-step-reuse-quality-controls-after-P-v1',timeout=300)
assert control['completed'] and all(z['numeric_pass'] for z in control['rows'])
proceed=all(v['both_initial_and_repeat_candidate_in_observed_native_set'] for v in control['membership'].values())
p=next((r/'rope-step-reuse-quality-controls-after-P-v1').glob('*/metrics.json'))
review=dict(proceed_finite_formal=proceed,reviewed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),initial_full64_equal_cases=18,cases=26,finite_controls=control['membership'],P_original_quality_CID=binding['P_original_quality_CID'],P_current_CID=binding['P_CID'],D_retained_CID=binding['D_CID'],fresh_PD_smoke8=True,control_source=str(p.relative_to(r)),control_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scope='Original quality52 before P timeout; new same-immutable-config P after watchdog, same loaded D. Initial and repeated candidate full64 variants must both be covered by current-pair native controls. No general QA, harmless-rounding or service gain claim.')
(r/'rope-step-reuse-quality-reviewed-v2.json').write_text(json.dumps(review,indent=2)+'\n');print(json.dumps(review),flush=True)
if proceed:subprocess.run([sys.executable,str(r/'run_rope_step_reuse_formal_v2.py')],check=True)
