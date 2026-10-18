#!/usr/bin/env python3
"""Read installed torch_npu Python source without importing torch/torch_npu."""
import ast,datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L009-R003-stream-api';O.mkdir(parents=True,exist_ok=False)
helper=R/'scripts/collect_qk_support.py';harness={}
for n in ast.parse(helper.read_text()).body:
 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['NODE','JUMP']:harness[n.targets[0].id]=ast.literal_eval(n.value)
INNER=r'''
import importlib.metadata,json,pathlib,hashlib,subprocess,sys
j=json.loads(sys.argv[1]);root=pathlib.Path('/sgl-workspace/sglang');head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();assert head==j['expected_commit']
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
dist=importlib.metadata.distribution('torch-npu');package_root=pathlib.Path(dist.locate_file('torch_npu')).resolve()
v={'source_commit':head,'distribution':{'name':dist.metadata['Name'],'version':dist.version,'package_root':str(package_root)},'files':{},'read_only':True,'target_modules_imported':False,'generation_requests':0}
for rel in ['npu/streams.py','npu/__init__.py','contrib/transfer_to_npu.py','__init__.py']:
 p=(package_root/rel).resolve();assert p.is_relative_to(package_root)
 if p.is_file():
  b=p.read_bytes();v['files']['torch_npu/'+rel]={'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'content':b.decode()}
 else:v['files']['torch_npu/'+rel]={'path':str(p),'missing':True}
assert not v['files']['torch_npu/npu/streams.py'].get('missing'),'Stream source unavailable'
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
print(json.dumps(v))
'''
node=harness['NODE'];needle="cid=subprocess.check_output(['docker','inspect','--format','{{.Id}}',name],text=True,timeout=15).strip();assert cid==j['expected_container_id'],'Observed container identity drift'"
assert needle in node;node=node.replace(needle,needle+"\nimage=subprocess.check_output(['docker','inspect','--format','{{.Image}}',name],text=True,timeout=15).strip();assert image==j['expected_image_id'],'Observed image drift'")
request={'expected_commit':'ce0a8fea5f4245839d174b2b8c55692be1810a7f','expected_container_id':'f42cd0a62f5f0778c0e6576ff8177f3d8c71f0e77c03719e22aa9af7b2e7a9c8','expected_image_id':'sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263','inner':INNER,'node':node}
cmd=['ssh','-S','/tmp/glm53-active-20261018.sock','-o','BatchMode=yes','-o','ConnectTimeout=8','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',harness['JUMP']])]
start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps(request),capture_output=True,text=True,timeout=130)
(O/'package-source.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr)
if p.returncode==0:
 for n,v in json.loads(p.stdout)['files'].items():
  if v.get('missing'):continue
  q=R/'sources/installed-packages'/n;q.parent.mkdir(parents=True,exist_ok=True)
  if q.exists():assert hashlib.sha256(q.read_bytes()).hexdigest()==v['sha256'],'Package source snapshot drift: '+n
  else:q.write_text(v['content'])
receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'read_only':True,'target_runtime_imported':False,'input_hashes':{'controller':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'harness':hashlib.sha256(helper.read_bytes()).hexdigest()},'artifacts':{n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in ['package-source.json','stderr.log']}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'exit_code':p.returncode}));sys.exit(p.returncode)
