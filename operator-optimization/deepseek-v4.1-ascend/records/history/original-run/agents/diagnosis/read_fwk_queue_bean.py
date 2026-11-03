#!/usr/bin/env python3
"""Follow only OP_MARK's configured bean, plus fixed thread helper and parser."""
import ast,hashlib,json
from pathlib import Path
root=Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis')
cfg=root/'prof_config/_fwk_file_parser_config.py';const=root/'prof_common_func/_constant.py';parser=root/'prof_parse/_fwk_file_parser.py'
def read(p,expected=None):
 raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest()
 if expected and h!=expected:raise RuntimeError('source changed '+str(p))
 if len(raw)>500000:raise RuntimeError('file too large')
 return raw.decode(),h
text,h=read(cfg);tree=ast.parse(text);mapping=[]
for node in ast.walk(tree):
 if isinstance(node,ast.Dict):
  for k,v in zip(node.keys,node.values):
   if isinstance(k,ast.Attribute) and k.attr=='OP_MARK' and isinstance(v,ast.Dict):
    for kk,vv in zip(v.keys,v.values):
     if isinstance(kk,ast.Constant) and kk.value=='bean' and isinstance(vv,ast.Name):mapping.append(vv.id)
out=[{'path':str(cfg),'sha256':h,'text':text,'hash_status':'first recorded now'}]; resolution={'bean_names':mapping,'path':None,'error':None}
if len(mapping)==1:
 matches=[]
 for node in ast.walk(tree):
  if isinstance(node,ast.ImportFrom):
   for alias in node.names:
    if (alias.asname or alias.name)==mapping[0] and node.module and 'prof_bean.' in node.module:
     matches.append(node.module.split('prof_bean.',1)[1])
 if len(matches)==1 and all(repr(x).startswith("'") and x.isidentifier() for x in matches[0].split('.')):
  bean=root/'prof_bean'/Path(*matches[0].split('.')).with_suffix('.py')
  if bean.is_file() and not bean.is_symlink():
   t,bh=read(bean);out.append({'path':str(bean),'sha256':bh,'text':t,'hash_status':'first recorded now'});resolution['path']=str(bean)
  else:resolution['error']='bean file missing or symlink'
 else:resolution['error']='unable to uniquely resolve mapped bean import'
else:resolution['error']='unable to uniquely resolve OP_MARK bean mapping'
text,h=read(const,'4643479478fb31a8cce3b5838edb4ec61bb159d04846de99ec62f383dede5b31');lines=text.splitlines();tree=ast.parse(text)
nodes=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='contact_2num']
out.append({'path':str(const),'sha256':h,'sections':[{'start':n.lineno,'end':n.end_lineno,'text':'\n'.join(lines[n.lineno-1:n.end_lineno])} for n in nodes]})
text,h=read(parser,'5368902817cf6344706f5faad6511d4c06aa297054779bf409a361333b198430');lines=text.splitlines();out.append({'path':str(parser),'sha256':h,'sections':[{'start':181,'end':min(340,len(lines)),'text':'\n'.join(lines[180:340])}]})
print(json.dumps({'scope':'Fixed OP_MARK config, its unique imported bean when resolvable, exact contact_2num helper and remaining parser lines','resolution':resolution,'files':out,'limits':['Config and bean SHA are first recorded now; must independently bind their source version to the actual export.','No inferred clock frequency, no cross-domain timing, no NPU work.']},ensure_ascii=False,indent=2))
