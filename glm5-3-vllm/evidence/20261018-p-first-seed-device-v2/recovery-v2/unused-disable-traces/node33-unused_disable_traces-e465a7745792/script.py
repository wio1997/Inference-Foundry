from pathlib import Path
import json,subprocess,socket
assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/unused-draft-live1-20261018/node33');plan=json.loads((root/'plan.json').read_text());cid=json.loads((root/'launch.json').read_text())['container_id'];assert cid=='6584fed1fd2712610c6840a6cc631914e6e4549958055a5b2175f22455c7426a';d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['State']['Running'] and d['Name']=='/'+plan['container']['name'] and d['Image']==plan['image']['id'];probe=root/'divergence-probe';assert not (probe/'shared-overlap-disabled').exists();(probe/'skip-unused-single-token-draft').unlink(missing_ok=True)
for path in [probe/'disabled',probe/'overlap-probe-disabled',probe/'unused-draft-trace-disabled',root/'adopt-trace/disabled']:
 path.parent.mkdir(exist_ok=True);path.touch()
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'CID':cid,'all4_diagnostic_writes_disabled':True,'initial_mode':'native','shared_overlap':'native on','model_kept_running':True,'scope':'Own exact model diagnostic sentinels only, no other workload changed'})+'\nGLM53_RESULT_JSON_END')
