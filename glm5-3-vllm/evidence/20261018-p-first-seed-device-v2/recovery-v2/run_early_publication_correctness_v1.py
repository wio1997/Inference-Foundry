from pathlib import Path
import importlib.util,json,time
r=Path(__file__).resolve().parent;repo=r.parents[3]
spec=importlib.util.spec_from_file_location('transport',r/'run_prefix_count_service_v1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
started=time.monotonic()
while not(r/'early-sample-publication-startup-outcome.json').exists():
    assert time.monotonic()-started<1300
    time.sleep(15)
assert json.loads((r/'early-sample-publication-startup-outcome.json').read_text())['both_ready']
assert json.loads(next((r/'early-sample-publication-setup-v1').glob('*/metrics.json')).read_text())['publication_default_off']
binding=json.loads((r/'early-sample-publication-formal-CID-binding-v1.json').read_text())
root='/data/wio/glm5-3-vllm-execution/20261018-state-tail/early-publication-live1-20261019/node33'
proxy=m.collect(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/start_proxy.py','early_publication_official_proxy_v1','early-sample-publication-official-proxy-v1',('--run-dir',root.replace('node33','node32'),'--expected-P-CID',binding['P_CID'],'--minimum-lifetime-seconds','900'),timeout=100,node=32)
assert proxy['ready']
def phase(name):
    x=m.collect(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/validate_sfa_replay_order.py','early_publication_'+name.replace('-','_')+'_v1','early-sample-publication-'+name+'-v1',('--execute','--phase',name,'--probe-kind','early_sample_publication','--run-dir',root,'--expected-D-CID',binding['D_CID'],'--cases',root+'/prefill-padding-cases-v1.json','--run-tag','v1'),timeout=650)
    assert x['completed'] and x['numeric_all_pass']
    print(json.dumps(dict(phase=name,rows=len(x['rows']),numeric_all=True)),flush=True);return x
phase('functional');phase('c4-functional')
paths=m.collect(r/'read_early_publication_paths_v1.py','early_publication_paths_v1','early-sample-publication-paths-v1',timeout=90)
assert paths['all16_paths'] and paths['all_selected_late_copies_bitwise'] and paths['all16_retirement_ACK']
print(json.dumps(dict(phase='all16_retirement_paths',passed=True)),flush=True)
aborted=m.collect(r/'validate_early_publication_native_abort_v1.py','early_publication_native_abort_v1','early-sample-publication-native-abort-v1',timeout=240)
assert aborted['completed'] and aborted['native_abort_receipt']['native_finish_requests_called']
print(json.dumps(dict(phase='native_abort',native_request_id=aborted['native_abort_receipt']['native_request_id'],deferred_after=aborted['native_abort_receipt']['deferred_after'])),flush=True)
paths=m.collect(r/'read_early_publication_paths_v1.py','early_publication_paths_post_abort_v1','early-sample-publication-paths-post-abort-v1',timeout=90)
assert paths['all16_paths'] and paths['all_selected_late_copies_bitwise']
disabled=m.collect(r/'disable_early_publication_metadata_v1.py','early_publication_disable_metadata_v1','early-sample-publication-disable-metadata-v1',timeout=60)
assert disabled['completed']
quality=phase('quality');phase('cached')
unequal=[k for k,v in quality['sequence_comparisons'].items() if not v['all_same']]
out=dict(completed=True,functional16_numeric=True,C4_functional8_numeric=True,quality52_numeric=True,cached16_numeric=True,
         all16_paths=True,all16_late_bitwise=True,all16_retirement_ACK=True,native_abort_and_reuse=True,
         quality_full64_equal_cases=len(quality['sequence_comparisons'])-len(unequal),unequal_full64_cases=unequal,
         next='Primary finite full64 review before formal. New differing sequence requires native/adaptive/native controls; no harmless-rounding or general QA claim.')
(r/'early-sample-publication-correctness-outcome-v1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out),flush=True)
