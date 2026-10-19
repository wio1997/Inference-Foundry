from pathlib import Path
import hashlib,json,subprocess
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/prefix-count-live1-20261019/node33');cid='8f434957c821dec55a77d352a68d16c14f2558d0b4c7e11287e0e37181111fdd'
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['State']['Running'] and json.loads((root/'launch.json').read_text())['container_id']==cid
src=root.parent.parent/'moe-prefix-target-live1-20261019/node33/prefill-padding-cases-v1.json';data=src.read_bytes();assert hashlib.sha256(data).hexdigest()=='c82d7ecb6cf5d2a8786ce844d133e6e5627c743b714c17bf01055e34b3959906';assert not(root/'prefill-padding-cases-v1.json').exists();(root/'prefill-padding-cases-v1.json').write_bytes(data)
for name in ['rope-cache-layout-probe','moe-prefix-gmm2-count-probe']:(root/name).mkdir(exist_ok=True)
(root/'rope-cache-layout-probe/enable').touch();(root/'rope-cache-layout-probe/disabled').touch();(root/'moe-prefix-gmm2-count-probe/target-only').touch()
assert not(root/'moe-prefix-gmm2-count-probe/enable').exists() and not(root/'moe-prefix-gmm2-count-probe/disabled').exists()
print(json.dumps({'CID':cid,'RoPE_enabled_both_arms':True,'MTP_native_both_arms':True,'native_routing_and_GMM2_count1':True,'fixture_sha256':hashlib.sha256(data).hexdigest()}))
