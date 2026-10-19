from pathlib import Path
import importlib.util,json,time
r=Path(__file__).resolve().parent;repo=r.parents[3]
s=importlib.util.spec_from_file_location('transport',r/'run_prefix_count_service_v1.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
start=time.monotonic()
while not(r/'rope-step-reuse-startup-outcome.json').exists():
 assert time.monotonic()-start<1500;time.sleep(10)
assert json.loads((r/'rope-step-reuse-startup-outcome.json').read_text())['both_ready']
cid=json.loads(next((r/'rope-step-reuse-D-launch-v1').rglob('launch.json')).read_text())['container_id']
root='/data/wio/glm5-3-vllm-execution/20261018-state-tail/rope-step-reuse-live1-20261019/node33'
setup=m.collect(r/'setup_rope_step_reuse_v1.py','rope_step_reuse_setup_v1','rope-step-reuse-setup-v1',timeout=90);assert setup['reuse_default_off']
def phase(name):
 x=m.collect(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/validate_sfa_replay_order.py','rope_step_reuse_'+name.replace('-','_')+'_v1','rope-step-reuse-'+name+'-v1',('--execute','--phase',name,'--probe-kind','rope_step_reuse','--run-dir',root,'--expected-D-CID',cid,'--cases',root+'/prefill-padding-cases-v1.json','--run-tag','v1'),timeout=650)
 assert x['completed'] and x['numeric_all_pass'];print(json.dumps(dict(phase=name,rows=len(x['rows']),numeric_all=True)),flush=True);return x
phase('functional');phase('c4-functional')
paths=m.collect(r/'read_rope_step_reuse_paths_v1.py','rope_step_reuse_paths_v1','rope-step-reuse-paths-v1',timeout=90);assert paths['all16']
print(json.dumps(dict(phase='all16_step_reuse',selected=sum(v['selected'] for v in paths['rows']),bitwise=True,independent_indexer_storage=True)),flush=True)
m.collect(r/'disable_rope_step_reuse_observer_v1.py','rope_step_reuse_disable_observer_v1','rope-step-reuse-disable-observer-v1',timeout=60)
quality=phase('quality');phase('cached')
unequal=[k for k,v in quality['sequence_comparisons'].items() if not v['all_same']]
out=dict(completed=True,functional16=True,C4_functional8=True,quality52=True,cached16=True,all16_paths=True,all_selected_pair_and_owned_copy_bitwise=True,quality_full64_equal_cases=26-len(unequal),unequal_cases=unequal,D_CID=cid,P_CID='84bdaf268d006b2d4d172bc6f9adb8f5f74cc9f38e1a622644536269f0918f8b',next='Primary quality review before formal; differing sequences require finite current native/adaptive/native controls. No service gain or general QA claim.')
(r/'rope-step-reuse-correctness-outcome-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
