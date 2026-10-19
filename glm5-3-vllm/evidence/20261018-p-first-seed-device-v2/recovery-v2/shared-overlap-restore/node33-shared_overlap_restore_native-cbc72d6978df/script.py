from pathlib import Path
import json,socket,subprocess
assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/shared-overlap-live1-20261018/node33');plan=json.loads((root/'plan.json').read_text());launch=json.loads((root/'launch.json').read_text());cid=launch['container_id'];assert cid=='8217b0b16541016e558c985045f10a2b146e7646233f5ee869d62d9eb7ea0dfa';actual=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert actual['State']['Running'] and actual['Image']==plan['image']['id'] and actual['Name']=='/'+plan['container']['name']
probe=root/'divergence-probe';assert not (probe/'shared-overlap-disabled').exists();rows=[json.loads(x) for f in probe.glob('overlap-policy-*.jsonl') for x in f.read_text().splitlines() if x.strip()];last={}
for row in sorted(rows,key=lambda x:x['monotonic_ns']):
 if row['kind']=='shared_overlap_actual_calls':last[row['tp_rank']]=row
assert set(last)==set(range(16)) and all(x['enabled'] is True and x['calls'].get('target_on')==75 and not x['calls'].get('target_off') for x in last.values())
for folder,name in [(probe,'disabled'),(probe,'overlap-probe-disabled'),(root/'adopt-trace','disabled')]:
 folder.mkdir(exist_ok=True);(folder/name).touch()
report={'CID':cid,'all16_last_native_target_on_75':True,'last_native_drafter_counts':{str(k):v['calls'].get('after_target_on',0) for k,v in last.items()},'shared_flag_off_marker_absent':True,'tensor_observer_disabled':True,'policy_observer_disabled':True,'host_trace_disabled':True,'model_kept_running':True,'scope':'Only owned diagnostic sentinels; no model precision/weights/operator or other job changed'};print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END')
