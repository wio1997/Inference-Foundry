#!/usr/bin/env python3
"""Read fixed installed parser source and safe filename/config metadata only."""
import base64,json,pathlib,shlex,subprocess
remote=r'''import hashlib,json,os,pathlib
r=pathlib.Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis')
spec=[('prof_bean/_torch_op_bean.py',1,180),('prof_bean/_param_tensor_bean.py',1,180),('prof_parse/_event_tree_parser.py',70,240),('prof_parse/_event_tree_parser.py',400,530),('prof_parse/_event_tree_parser.py',545,620),('prof_config/_fwk_file_parser_config.py',1,85),('prof_view/prof_db_parse/_fwk_api_db_parser.py',1,150)]
s=[]
for rel,a,b in spec:
 p=r/rel
 if not p.exists():s.append({'path':str(p),'missing':True});continue
 x=p.read_bytes();lines=x.decode().splitlines();s.append({'path':str(p),'sha256':hashlib.sha256(x).hexdigest(),'start':a,'end':b,'lines':lines[a-1:b]})
base=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces')
rank=base/'dp1_pp0_tp0_dcp0_ep8_rank0_737_20261029023525648_ascend_pt';files=[];cfg=[]
for p in rank.rglob('*'):
 if not p.is_file() or p.is_symlink():continue
 rel=str(p.relative_to(rank))
 if 'FRAMEWORK/' in rel.upper() or 'tensor_info' in rel:files.append({'path':rel,'bytes':p.stat().st_size})
 if p.name.startswith('profiler_info') and p.suffix=='.json':
  try:
   data=json.loads(p.read_text());found=[]
   def walk(x,key=''):
    if isinstance(x,dict):
     for k,v in x.items():walk(v,key+'.'+k)
    elif isinstance(x,list):
     for i,v in enumerate(x):walk(v,key+'['+str(i)+']')
    elif key.rsplit('.',1)[-1] in ('record_shapes','profile_memory','with_stack','record_op_args') and isinstance(x,bool):found.append({'key':key,'value':x})
   walk(data);cfg.append({'path':rel,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'selected':found})
  except Exception as e:cfg.append({'path':rel,'error_type':type(e).__name__})
print(json.dumps({'source_sections':s,'ep8_framework_or_tensor_info_files':files,'ep8_config_selected':cfg}))'''
e=base64.b64encode(remote.encode()).decode();cmd='docker exec wio-vllm-v41-repro169-stage-20261028 nice -n 15 python3 -c '+shlex.quote("import base64;exec(base64.b64decode('"+e+"'))")
r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],capture_output=True,text=True,timeout=90,check=True);v=json.loads(r.stdout);o=pathlib.Path(__file__).parent/'tensorinfo-parser-sources.json';o.write_text(json.dumps(v,indent=2)+'\n');print('sections',len(v['source_sections']),'files',len(v['ep8_framework_or_tensor_info_files']),'configs',len(v['ep8_config_selected']))
