from pathlib import Path
import json,subprocess,sys,time
repo=Path('/Users/wio/work/Inference-Foundry-execution-plan');r=repo/'glm5-3-vllm/evidence/20261018-p-first-seed-device-v2/recovery-v2';start=time.monotonic()
while not (r/'sfa-replay-order-live2-sequence-outcome.json').exists():
 assert time.monotonic()-start<2400;time.sleep(10)
x=json.loads((r/'sfa-replay-order-live2-sequence-outcome.json').read_text());assert x['completed']
q=subprocess.run([sys.executable,str(repo/'glm5-3-vllm/tools/collect_node.py'),'--node','33','--script',str(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/validate_sfa_cancel.py'),'--label','sfa_replay_order_live2_cancel_v1','--output',str(r/'sfa-replay-order-live2-cancel'),'--control','/tmp/glm53-active-20261018.sock','--timeout','360'],cwd=repo,capture_output=True,text=True,timeout=500);(r/'sfa-replay-order-live2-cancel-collector.txt').write_text(q.stdout+q.stderr);print(q.stdout,flush=True);assert q.returncode==0,q.stderr
x=json.loads(next((r/'sfa-replay-order-live2-cancel').glob('*/metrics.json')).read_text());print(json.dumps({k:x.get(k) for k in ['completed','client_close_requests','full_reuse_and_peer_responses','backend_abort_log_request_count','abort_counter_deltas','failure']}),flush=True)
