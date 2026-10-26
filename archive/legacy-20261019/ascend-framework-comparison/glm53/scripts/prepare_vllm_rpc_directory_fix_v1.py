"""Local new bundle: owned directory preparation only, no SDK/model-config changes."""
from pathlib import Path
import ast,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_text(json.dumps(v,indent=2)+'\n')
def main():
 assert sha(ROOT/'scripts/vllm_correctness_entry_v2.py')=='285da95489000b01ca0cd6f9b20dc1dfbe48bd991c69a698ac7a83b25e111696'
 assert json.loads((ROOT/'results/vllm-entry-directories-local-fixture-v2.json').read_text())['pass']
 for n in [30,31,43,45]:
  c=json.loads((ROOT/f'config/vllm-shared-container-v3/node-{n}.json').read_text());c['run_id']='20261017T1766-vllm-shared-pool-v4';write(ROOT/f'config/vllm-shared-container-v4/node-{n}.json',c)
 p=ROOT/'scripts/sdk_vllm_pool568_preflight_v5.py';assert not p.exists();s=(ROOT/'scripts/sdk_vllm_pool568_preflight_v4.py').read_text();needle="print('GLM53_SDK_CONFIG_PREFLIGHT='";assert needle in s
 code="""entry_path=root/'entry.py';assert hashlib.sha256(entry_path.read_bytes()).hexdigest()=='285da95489000b01ca0cd6f9b20dc1dfbe48bd991c69a698ac7a83b25e111696'
spec=importlib.util.spec_from_file_location('project_entry',entry_path);entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
prepared=entry.prepare_runtime_directories(Path('/run/glm53'),{'VLLM_RPC_BASE_PATH':'/run/glm53/rpc','VLLM_CACHE_ROOT':'/run/glm53/cache'})
import zmq,uuid
probe=Path(prepared['rpc'])/('glm53-native-ipc-preflight-'+uuid.uuid4().hex)
context=zmq.Context();channel=context.socket(zmq.PULL);channel.setsockopt(zmq.LINGER,0)
try:channel.bind('ipc://'+str(probe));assert channel.getsockopt(zmq.LAST_ENDPOINT)==('ipc://'+str(probe)).encode()
finally:channel.close();context.term();probe.unlink(missing_ok=True)
"""
 s=s.replace(needle,code+needle).replace("'actual_provider_validator_all4_passed':True", "'native_RPC_IPC_bind':True,'entrypoint_sha256':'285da95489000b01ca0cd6f9b20dc1dfbe48bd991c69a698ac7a83b25e111696','actual_provider_validator_all4_passed':True");ast.parse(s);p.write_text(s)
 m=json.loads((ROOT/'config/standalone-budget-stage-v10.json').read_text());bundle=m['bundles'][0];bundle['remote']='/data/wio/inference-foundry/glm53/overlays/vllm-sdma-v11-20261017'
 for name,item in bundle['files'].items():
  loc=item['local'].replace('scripts/vllm_correctness_entry.py','scripts/vllm_correctness_entry_v2.py').replace('sdk_vllm_pool568_preflight_v4.py','sdk_vllm_pool568_preflight_v5.py').replace('config/vllm-shared-container-v3/','config/vllm-shared-container-v4/');item.update(local=loc,sha256=sha(ROOT/loc))
 m['side_effect_scope']='New own source/config + CPU-only actual provider/SDK config validation and real ZMQ IPC bind in private tmpfs. Native SDK/model launch args unchanged; existing native CLI qualification reused by exact hashes, new runtime requires actual devices/capacity/IPC validation.'
 write(ROOT/'config/standalone-budget-stage-v11.json',m)
 print(json.dumps({'manifest_sha256':sha(ROOT/'config/standalone-budget-stage-v11.json'),'entry_sha256':sha(ROOT/'scripts/vllm_correctness_entry_v2.py'),'SDK_changed':False,'model_launch_args_changed':False,'resource_actions':0}))
if __name__=='__main__':main()
