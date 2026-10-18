from pathlib import Path
import hashlib,json,subprocess
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/moe-prefix-histogram-live3-20261019/node33');cid='c7be13e9c4f558e7da2aa395b764cccbd5026bf1dc4f5f3cba52abf4c7dc76db'
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['State']['Running'] and json.loads((root/'launch.json').read_text())['container_id']==cid
src=root.parent.parent/'moe-prefix-histogram-live2-20261019/node33/prefill-padding-cases-v1.json';data=src.read_bytes();assert hashlib.sha256(data).hexdigest()=='c82d7ecb6cf5d2a8786ce844d133e6e5627c743b714c17bf01055e34b3959906'
assert not(root/'prefill-padding-cases-v1.json').exists();(root/'prefill-padding-cases-v1.json').write_bytes(data)
for name in ['rope-cache-layout-probe','moe-prefix-histogram-probe']:(root/name).mkdir(exist_ok=True)
(root/'rope-cache-layout-probe/enable').touch();(root/'rope-cache-layout-probe/disabled').touch()
print(json.dumps({'CID':cid,'RoPE_enabled_observer_off':True,'prefix_native':not(root/'moe-prefix-histogram-probe/enable').exists(),'fixture_sha256':hashlib.sha256(data).hexdigest()}))
