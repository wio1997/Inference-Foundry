from pathlib import Path
import hashlib,json,subprocess
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/moe-prefix-target-live1-20261019/node33');cid='4d620dfbdf2981ee4c95e029fd1cf3ee297bdb153a548bbb39027ac0e9252041';d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['State']['Running'] and json.loads((root/'launch.json').read_text())['container_id']==cid
src=root.parent.parent/'moe-prefix-histogram-live3-20261019/node33/prefill-padding-cases-v1.json';data=src.read_bytes();assert hashlib.sha256(data).hexdigest()=='c82d7ecb6cf5d2a8786ce844d133e6e5627c743b714c17bf01055e34b3959906';assert not(root/'prefill-padding-cases-v1.json').exists();(root/'prefill-padding-cases-v1.json').write_bytes(data)
for name in ['rope-cache-layout-probe','moe-prefix-histogram-probe']:(root/name).mkdir(exist_ok=True)
(root/'rope-cache-layout-probe/enable').touch();(root/'rope-cache-layout-probe/disabled').touch();(root/'moe-prefix-histogram-probe/target-only').touch()
assert not(root/'moe-prefix-histogram-probe/enable').exists() and not(root/'moe-prefix-histogram-probe/disabled').exists()
print(json.dumps({'CID':cid,'RoPE_enabled_in_both_arms':True,'MTP_always_native_target_only_selector':True,'target_prefix_initial_native':True,'fixture_sha256':hashlib.sha256(data).hexdigest()}))
