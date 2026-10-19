from pathlib import Path
import importlib.util
import json
import time

r=Path(__file__).resolve().parent;repo=r.parents[3]
spec=importlib.util.spec_from_file_location('transport',r/'run_prefix_count_service_v1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
started=time.monotonic()
while not(r/'early-sample-shadow-startup-outcome.json').exists():
    assert time.monotonic()-started<1250
    time.sleep(15)
assert json.loads((r/'early-sample-shadow-startup-outcome.json').read_text())['both_ready']
assert json.loads((r/'early-sample-shadow-contract-v1.json').read_text())['scope'].startswith('Diagnostic only.')
root='/data/wio/glm5-3-vllm-execution/20261018-state-tail/early-copy-shadow-live1-20261019/node33'
cid='9b788715cb6e34139c2e9a70ef5bcf49009dda507164bf11e9e00cd07c163863'
proxy=m.collect(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/start_proxy.py','early_sample_shadow_official_proxy_v1','early-sample-shadow-official-proxy-v1',('--run-dir',root.replace('node33','node32'),'--expected-P-CID','a04ab7b22797797ca23431b9b7b5f0ef536ae625aad28390cde4b0550f99f14e','--minimum-lifetime-seconds','900'),timeout=100,node=32)
assert proxy['ready']
for phase in ['functional','c4-functional','cached']:
    if phase=='cached':
        x=m.collect(r/'disable_early_sample_shadow_metadata_v1.py','early_sample_shadow_metadata_v1','early-sample-shadow-metadata-v1',timeout=60)
        assert x['completed']
    x=m.collect(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/validate_sfa_replay_order.py','early_sample_shadow_'+phase.replace('-','_')+'_v1','early-sample-shadow-'+phase+'-v1',('--execute','--phase',phase,'--probe-kind','early_sample_copy_shadow','--run-dir',root,'--expected-D-CID',cid,'--cases',root+'/prefill-padding-cases-v1.json','--run-tag','v1'),timeout=500)
    assert x['completed'] and x['numeric_all_pass']
    print(json.dumps(dict(phase=phase,rows=len(x['rows']),numeric_all=True)),flush=True)
x=m.collect(r/'validate_early_sample_shadow_cancel_v1.py','early_sample_shadow_cancel_v1','early-sample-shadow-cancel-v1',timeout=500)
assert x['completed'] and x['client_close_requests']==10 and x['full_reuse_and_peer_responses']==16
print(json.dumps(dict(phase='cancel',client_closes=10,full_responses=16,backend_abort_log_request_count=x['backend_abort_log_request_count'])),flush=True)
x=m.collect(r/'read_early_sample_shadow_v1.py','early_sample_shadow_read_v1','early-sample-shadow-read-v1',timeout=90)
assert x['completed'] and x['bitwise_all'] and x['observed_ranks']==list(range(16))
(r/'early-sample-shadow-reviewed-v1.json').write_text(json.dumps(x,indent=2)+'\n')
print(json.dumps(x),flush=True)
