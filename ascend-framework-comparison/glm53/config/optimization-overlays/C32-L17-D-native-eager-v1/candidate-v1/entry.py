"""Verify frozen native source and exact config, then native SGLang server."""
import hashlib,json,os,pathlib,sys
root=pathlib.Path('/opt/glm53/source');manifest=json.loads((root/'manifest.json').read_text())
assert hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()=='0eec0a4538747c6e8f30a1fced4bf262de66040db5ea182ed8330b3e7494f528'
for item in manifest['files']:assert hashlib.sha256(pathlib.Path(item['container_path']).read_bytes()).hexdigest()==item['patched_sha256']
for item in manifest['helpers']:assert hashlib.sha256((root/item['local']).read_bytes()).hexdigest()==item['sha256']
node=int(sys.argv[1]);assert node in (34,35,36,41)
path=root/f'config/node-{node}.json';config=json.loads(path.read_text());assert config['node']==node and config['parallel_intent']['role_die']==32
for key,value in config['environment_intent'].items():assert os.environ[key]==value
assert os.environ['SGLANG_MOONCAKE_SEND_AUX_TCP']=='0'
args=[sys.executable,'-m','sglang.launch_server',*config['launch_args']]
print('GLM53_SGLANG_ENTRY='+json.dumps({'project_id':'glm53','node':node,'pid':os.getpid(),'config_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),'phase':config['phase'],'args':args}),flush=True)
os.execv(sys.executable,args)
