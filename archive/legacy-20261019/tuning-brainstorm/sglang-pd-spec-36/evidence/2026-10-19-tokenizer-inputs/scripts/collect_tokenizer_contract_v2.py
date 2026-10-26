#!/usr/bin/env python3
"""Read canonical tokenizer source and a strictly whitelisted model-file manifest."""
import ast,datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L010-R002-tokenizer-sources';O.mkdir(parents=True,exist_ok=False);helper=R/'scripts/collect_qk_support.py';harness={}
for n in ast.parse(helper.read_text()).body:
 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['INNER','NODE','JUMP']:harness[n.targets[0].id]=ast.literal_eval(n.value)
MODEL_CODE=r'''
model=pathlib.Path('/model');resolved_model=model.resolve()
v['model_candidate']={'container_path':'/model','config_expected_sha256':'ca8f2f47b07919a514c0ca223dc2ea2bc7445afaa5ac76c013a3784e096426ca','files':{},'scope':'Observed candidate only; no tokenizer/runtime instance or task PD baseline'}
for name in ['config.json','tokenizer_config.json','tokenizer.json','special_tokens_map.json','chat_template.jinja','chat_template.json','added_tokens.json','vocab.json','merges.txt','tokenizer.model']:
 p=model/name
 if not p.is_file():v['model_candidate']['files'][name]={'missing':True};continue
 if not p.resolve().is_relative_to(resolved_model):v['model_candidate']['files'][name]={'skipped':'symlink_outside_model_root'};continue
 b=p.read_bytes();item={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()};v['model_candidate']['files'][name]=item
 if name=='config.json':assert item['sha256']==v['model_candidate']['config_expected_sha256'],'Model candidate config drift'
 if name in ['tokenizer_config.json','special_tokens_map.json']:
  z=json.loads(b);allowed={'tokenizer_class','auto_map','model_max_length','bos_token','eos_token','pad_token','unk_token','add_bos_token','add_eos_token','chat_template','clean_up_tokenization_spaces','extra_special_tokens'};item['selected_fields']={k:x for k,x in z.items() if k in allowed}
 if name.startswith('chat_template') and len(b)<=65536:item['content']=b.decode()
import importlib.metadata
v['dependency_versions']={}
for name in ['transformers','tokenizers','jinja2','tiktoken','sglang']:
 try:v['dependency_versions'][name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:v['dependency_versions'][name]='NOT_INSTALLED'
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
v['target_runtime_imported']=False
'''
marker='print(json.dumps(v))';assert harness['INNER'].count(marker)==1
request=json.loads((R/'specs/L010-source-request.json').read_text());request.update({'inner':harness['INNER'].replace(marker,MODEL_CODE+'\n'+marker),'node':harness['NODE']})
cmd=['ssh','-S','/tmp/glm53-active-20261018.sock','-o','BatchMode=yes','-o','ConnectTimeout=8','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',harness['JUMP']])]
start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps(request),capture_output=True,text=True,timeout=130)
(O/'source-and-model.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr)
if p.returncode==0:
 for n,v in json.loads(p.stdout)['files'].items():
  q=R/'sources/installed'/n;q.parent.mkdir(parents=True,exist_ok=True)
  if q.exists():assert hashlib.sha256(q.read_bytes()).hexdigest()==v['sha256'],'Source snapshot drift: '+n
  else:q.write_text(v['content'])
receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'read_only':True,'input_hashes':{str(q.relative_to(R)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [pathlib.Path(__file__),helper,R/'specs/L010-source-request.json']},'artifacts':{n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in ['source-and-model.json','stderr.log']}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'exit_code':p.returncode}));sys.exit(p.returncode)
