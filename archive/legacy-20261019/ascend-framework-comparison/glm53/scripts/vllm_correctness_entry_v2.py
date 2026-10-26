"""Frozen model entrypoint with explicit project IPC/cache directory preparation."""
import hashlib,json,os,sys
from pathlib import Path

def prepare_runtime_directories(run_root,environment):
    root=Path(run_root)
    assert root.is_dir() and not root.is_symlink()
    intended={'rpc':environment.get('VLLM_RPC_BASE_PATH',str(root/'rpc')),
              'cache':environment.get('VLLM_CACHE_ROOT',str(root/'cache')),
              'profile':str(root/'profile')}
    prepared={}
    for name,value in intended.items():
        path=Path(value);assert path==root/name and not path.is_symlink(),('Runtime directory must be an owned direct child',name)
        path.mkdir(exist_ok=True);assert path.is_dir() and not path.is_symlink()
        prepared[name]=str(path)
    return prepared

def main(node):
    root=Path('/opt/glm53/source');manifest=json.loads((root/'manifest.json').read_text())
    for entry in manifest['files']:assert hashlib.sha256(Path(entry['container_path']).read_bytes()).hexdigest()==entry['patched_sha256']
    for entry in manifest['helpers']:assert hashlib.sha256((root/entry['local']).read_bytes()).hexdigest()==entry['sha256']
    config_path=Path(f'/opt/glm53/config/node-{node}.json');config=json.loads(config_path.read_text())
    assert node in (30,31,43,45) and config['node']==node
    assert config['kv_transfer']['kv_connector']=='MultiConnector' and config['parallel_intent']['role_die']==32
    directories=prepare_runtime_directories(Path('/run/glm53'),os.environ)
    print('GLM53_RUNTIME_DIRECTORIES='+json.dumps({'node':node,'directories':directories,'SDK_init_calls':0,'model_requests':0}),flush=True)
    args=[sys.executable,'-m','vllm.entrypoints.openai.api_server','--host','0.0.0.0','--port',str(config['api_port']),*config['launch_args']]
    print('GLM53_SERVE_ENTRY='+json.dumps({'project_id':'glm53','node':node,'pid':os.getpid(),'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),'args':args,'phase':config['phase']}),flush=True)
    os.execv(sys.executable,args)

if __name__=='__main__':main(int(sys.argv[1]))
