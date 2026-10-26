"""Build frozen staged-file maps locally. Never accesses resources."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def make(framework,source_root,config_root,nodes,remote,expected_manifest_sha):
    source=ROOT/source_root;path=source/'manifest.json';assert sha(path)==expected_manifest_sha
    manifest=json.loads(path.read_text());files={}
    def add(name,local,expected=None):
        path=ROOT/local;digest=sha(path)
        if expected:assert digest==expected
        assert name not in files
        files[name]={'local':str(path.relative_to(ROOT)),'sha256':digest}
    prefix='source/' if framework=='vllm' else ''
    add(prefix+'manifest.json',source_root+'/manifest.json',expected_manifest_sha)
    for item in manifest['files']:
        add(prefix+item['local'],source_root+'/'+item['local'],item['patched_sha256'])
    for item in manifest['helpers']:
        add(prefix+item['local'],source_root+'/'+item['local'],item['sha256'])
    for node in nodes:add(f'config/node-{node}.json',config_root+f'/node-{node}.json')
    for name in ['meta.conf','local.conf','manifest.json']:add('config/'+name,config_root+'/'+name)
    for node in nodes[:2]:add(f'provider-config/node-{node}.json',f'config/standalone-pool-providers-v1/{framework}-node-{node}.json')
    add('provider.py','scripts/glm53_memcache_pool_provider_v1.py')
    add('sdk-preflight.py','scripts/sdk_localconfig_preflight_v1.py')
    entry='scripts/vllm_correctness_entry.py' if framework=='vllm' else 'scripts/sglang_correctness_entry_v1.py'
    add('entry.py',entry)
    if framework=='vllm':add('preflight.py','scripts/vllm_overlay_preflight_payload_v5.py')
    return {'framework':framework,'nodes':nodes,'remote':remote,'image':manifest.get('image_id'),'native_source_manifest_sha256':expected_manifest_sha,'files':files}

if __name__=='__main__':
    output=ROOT/'config/standalone-budget-stage-v1.json';assert not output.exists()
    bundles=[make('sglang','config/overlays/sglang-sdma-v1','config/sglang-correctness-v8',[34,35,36,41],'/data/wio/inference-foundry/glm53/overlays/sglang-sdma-v7-20261017','0eec0a4538747c6e8f30a1fced4bf262de66040db5ea182ed8330b3e7494f528'),make('vllm','config/overlays/vllm-sdma-v5','config/vllm-correctness-v6',[30,31,43,45],'/data/wio/inference-foundry/glm53/overlays/vllm-sdma-v6-20261017','1fc1d1e722b6dbb2d302980963c0285458e52acfa38abc95359828857a24d1ed')]
    output.write_text(json.dumps({'project_id':'glm53','phase':'STAGING_AND_CPU_SDK_CONFIG_PREFLIGHT_ONLY_NO_ENGINES','bundles':bundles},indent=2)+'\n')
    print(sha(output),[(b['framework'],len(b['files'])) for b in bundles])
