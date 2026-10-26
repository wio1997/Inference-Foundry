#!/usr/bin/env python3
"""Task-owned image runner: CPU only, four hashed model files mounted read-only."""
import ast,datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L025-R002-owned-cpu';O.mkdir(parents=True,exist_ok=False);helper=R/'scripts/probe_target_image_cpu.py';strings={}
for n in ast.parse(helper.read_text()).body:
 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['BOOTSTRAP','HOST','JUMP']:strings[n.targets[0].id]=ast.literal_eval(n.value)
expected=json.loads((R/'specs/L023-cpu-source-hashes.json').read_text());models=json.loads((R/'specs/L010-cpu-model-hashes.json').read_text());probe=R/'scripts/probe_owned_page_count_cpu.py'
payload={'probe':probe.read_text(),'probe_sha256':hashlib.sha256(probe.read_bytes()).hexdigest(),'sources':{f:(R/'sources/installed'/f).read_text() for f in expected},'hashes':expected,'model_hashes':models}
extra_paths=['specs/L025-owned-cpu.json', 'candidates/npu-owned-page-count/python/sglang/srt/hardware_backend/npu/allocator_npu.py', 'candidates/npu-owned-page-count/python/sglang/srt/mem_cache/allocation.py', 'candidates/npu-owned-page-count/python/sglang/srt/speculative/eagle_utils.py', 'candidates/npu-owned-page-count/python/sglang/srt/disaggregation/decode.py']
payload['extra_inputs']={f:{'content':(R/f).read_text(),'sha256':hashlib.sha256((R/f).read_bytes()).hexdigest()} for f in extra_paths}
payload['case_key']='source-contract'
bootstrap=strings['BOOTSTRAP'].replace('probe_metadata_support.py','probe_owned_page_count_cpu.py').replace('if-l007-cpu-','if-l025-cpu-');needle="json.dumps(j['hashes'])]";assert bootstrap.count(needle)==1;bootstrap=bootstrap.replace(needle,"json.dumps(j['hashes']),'--model-hashes',json.dumps(j['model_hashes'])]")
EXTRA_BOOTSTRAP=r"""
for filename,item in j['extra_inputs'].items():
 assert not filename.startswith('/') and '..' not in pathlib.PurePosixPath(filename).parts
 p=source/filename;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(item['content']);assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
sys.argv+=['--case-key',j['case_key'],'--protocol',str(source/'specs/L025-owned-cpu.json'),'--fixture-root',str(source)]
"""
needle="runpy.run_path(str(script),run_name='__main__')";assert bootstrap.count(needle)==1;bootstrap=bootstrap.replace(needle,EXTRA_BOOTSTRAP+'\n'+needle)
VENV_BOOTSTRAP=r"""
import os,sysconfig,venv,subprocess
venv_root=root/'cpu-env';venv.EnvBuilder(with_pip=False,system_site_packages=False,symlinks=True).create(venv_root)
base_site=pathlib.Path(sysconfig.get_paths()['purelib']);target_site=venv_root/'lib'/('python'+str(sys.version_info.major)+'.'+str(sys.version_info.minor))/'site-packages';excluded=[];linked=0
for entry in base_site.iterdir():
 if entry.name.startswith('torch_npu') or entry.suffix=='.pth' or entry.name in ['sitecustomize.py','usercustomize.py','__pycache__']:
  excluded.append(entry.name);continue
 (target_site/entry.name).symlink_to(entry,target_is_directory=entry.is_dir());linked+=1
child_env=os.environ.copy();child_env['PYTHONPATH']='/sgl-workspace/sglang/python'
code="import importlib.util,sys,runpy; assert importlib.util.find_spec('torch_npu') is None; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
p=subprocess.run([str(venv_root/'bin/python'),'-c',code,*sys.argv],capture_output=True,text=True,env=child_env,timeout=100)
sys.stderr.write(p.stderr)
if p.stdout:
 try:
  result=json.loads(p.stdout);result['cpu_dependency_view']={'base_site_packages':str(base_site),'venv_prefix':str(venv_root),'linked_top_level_entries':linked,'excluded_entries':sorted(excluded),'torch_npu_spec_absent':True,'system_site_packages':False,'source_files_reused_without_edits':True};print(json.dumps(result))
 except json.JSONDecodeError:sys.stdout.write(p.stdout)
sys.exit(p.returncode)
"""
needle="runpy.run_path(str(script),run_name='__main__')";assert bootstrap.count(needle)==1;bootstrap=bootstrap.replace(needle,VENV_BOOTSTRAP)
host=strings['HOST'].replace('L007-R004','L025-R002').replace('l007-cpu','l025-cpu').replace('20261019-r002','20261019-r002');host=host.replace("'2g'","'4g'").replace("h['Memory']==2*1024**3","h['Memory']==4*1024**3");host=host.replace("'NUMEXPR_NUM_THREADS':'1'","'NUMEXPR_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','SGLANG_USE_CPU_ENGINE':'1','SGLANG_CACHE_DIR':'/tmp/task-sglang-cache','TORCHINDUCTOR_CACHE_DIR':'/tmp/task-sglang-cache/inductor','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','HF_HOME':'/tmp/task-hf-cache','TORCH_HOME':'/tmp/task-torch-cache'")
needle="absent=run(['docker','container','inspect',name],check=False);assert absent.returncode!=0,'Name already exists; do not reuse'";assert host.count(needle)==1
host=host.replace(needle,needle+"\n model_root=pathlib.Path('/data/wio/GLM53/GLM-5.3-w8a8c8').resolve();model_hashes=j['payload']['model_hashes'];assert set(model_hashes)=={'config.json','tokenizer_config.json','tokenizer.json','chat_template.jinja'}\n mounts={}\n for filename,digest in model_hashes.items():\n  model_file=(model_root/filename).resolve();assert model_file.is_relative_to(model_root);assert hashlib.sha256(model_file.read_bytes()).hexdigest()==digest;mounts['/model/'+filename]=str(model_file)\n rec['model_file_mounts']={'paths':mounts,'sha256':model_hashes,'read_only':True,'weights_mounted':False}")
needle="args+=['--entrypoint','python3',image,'-c',j['bootstrap']]";assert host.count(needle)==1;host=host.replace(needle,"for target,source in mounts.items():args+=['--mount','type=bind,src='+source+',dst='+target+',readonly']\n "+needle)
needle="assert set(h['Tmpfs'])=={'/tmp'} and not c.get('Mounts')";assert host.count(needle)==1;host=host.replace(needle,"assert set(h['Tmpfs'])=={'/tmp'} and len(c['Mounts'])==4\n for binding in c['Mounts']:\n  assert binding['Type']=='bind' and binding['Destination'] in mounts and binding['Source']==mounts[binding['Destination']] and binding['RW'] is False")
needle="result=json.loads(p.stdout);assert result['correctness']=='PASS' and result['source_hashes']==j['payload']['hashes'] and result['script_sha256']==j['payload']['probe_sha256']";assert host.count(needle)==1;host=host.replace(needle,"result=json.loads(p.stdout);assert result['probe_status'] in ['PASS','UNAVAILABLE'] and result['source_hashes']==j['payload']['hashes'] and result['script_sha256']==j['payload']['probe_sha256']\n rec['probe_status']=result['probe_status']\n for filename,digest in model_hashes.items():assert hashlib.sha256((model_root/filename).read_bytes()).hexdigest()==digest\n rec['model_files_unchanged']=True")
host=host.replace("'correctness'","'orchestration'").replace('New CPU-only container; no model requests and no accelerator mounts. Source is hashed installed snapshot, not asserted to be image bundled source.','New CPU-only container; only4 verified model metadata/tokenizer files mounted read-only; no weights/NPU/HTTP work. Probe asserts image code identity before import.')
cmd=['ssh','-S','/tmp/glm53-active-20261018.sock','-o','BatchMode=yes','-o','ConnectTimeout=8','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',strings['JUMP']])]
start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps({'host':host,'bootstrap':bootstrap,'payload':payload}),capture_output=True,text=True,timeout=230)
(O/'remote-result.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr)
remote=json.loads(p.stdout) if p.stdout else {};result=json.loads(remote['probe_stdout']) if remote.get('probe_stdout') else None
if result:(O/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'probe_status':result.get('probe_status') if result else 'NOT_EXECUTED','controller_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'helper_sha256':hashlib.sha256(helper.read_bytes()).hexdigest(),'artifacts':{q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in O.iterdir() if q.is_file()}};(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'exit_code':p.returncode,'probe_status':receipt['probe_status']}));sys.exit(p.returncode)
