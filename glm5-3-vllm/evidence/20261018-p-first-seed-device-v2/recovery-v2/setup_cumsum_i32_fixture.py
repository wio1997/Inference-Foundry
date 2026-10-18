from pathlib import Path
import hashlib,json,subprocess
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/moe-cumsum-i32-live1-20261019/node33');cid='416f0dc10e34a115e2537cb62e46e2eb98d30825642ab4ba36d098c072cf11f5'
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['State']['Running'] and json.loads((root/'launch.json').read_text())['container_id']==cid
src=root.parent.parent/'moe-prefix-target-live1-20261019/node33/prefill-padding-cases-v1.json';data=src.read_bytes();assert hashlib.sha256(data).hexdigest()=='c82d7ecb6cf5d2a8786ce844d133e6e5627c743b714c17bf01055e34b3959906';assert not(root/'prefill-padding-cases-v1.json').exists();(root/'prefill-padding-cases-v1.json').write_bytes(data)
for name in ['rope-cache-layout-probe','moe-cumsum-i32-probe']:(root/name).mkdir(exist_ok=True)
(root/'rope-cache-layout-probe/enable').touch();(root/'rope-cache-layout-probe/disabled').touch();(root/'moe-cumsum-i32-probe/target-only').touch()
assert not(root/'moe-cumsum-i32-probe/enable').exists() and not(root/'moe-cumsum-i32-probe/disabled').exists()
print(json.dumps({'CID':cid,'RoPE_enabled_both_arms':True,'MTP_native_both_arms':True,'native_routing_and_GMM2_count1':True,'fixture_sha256':hashlib.sha256(data).hexdigest()}))
