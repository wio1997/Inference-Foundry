from pathlib import Path
import json,subprocess
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/moe-cumsum-i32-live1-20261019/node33');cid='416f0dc10e34a115e2537cb62e46e2eb98d30825642ab4ba36d098c072cf11f5';p=root/'moe-cumsum-i32-probe';d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['State']['Running'];assert (p/'target-only').exists() and not any((p/x).exists() for x in ['enable','replica-enable','MTP-only']);assert (root/'rope-cache-layout-probe/enable').exists() and (root/'rope-cache-layout-probe/disabled').exists();(p/'disabled').unlink(missing_ok=True)
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'completed':True,'D_CID':cid,'target_only':True,'observer_enabled_for_fresh_P_smoke':True,'no_source_changes':True})+'\nGLM53_RESULT_JSON_END')
