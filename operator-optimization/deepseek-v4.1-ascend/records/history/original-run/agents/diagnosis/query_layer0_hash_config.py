#!/usr/bin/env python3
"""Read exact historical model metadata and narrow installed config sources only."""
import ast,hashlib,importlib.util,json,re,sys
from pathlib import Path

# Exact model path from frozen-serve.sh. Reading metadata only; no weights payload.
model=Path('/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8')
MAX_JSON=64*1024*1024
MAX_SRC=2*1024*1024

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def json_file(p):
 if not p.is_file():return None,'NOT_VISIBLE'
 if p.stat().st_size>MAX_JSON:raise RuntimeError('STOP oversized metadata')
 return json.loads(p.read_text()),'PRESENT'

config_path=model/'config.json';index_path=model/'quant_model_weights.safetensors.index.json'
cfg,cstatus=json_file(config_path);idx,istatus=json_file(index_path)
model_result={'declared_model_path':str(model),'config':{'status':cstatus},'checkpoint_index':{'status':istatus}}
if cfg is not None:
 text=cfg.get('text_config',{})
 model_result['config'].update(sha256=sha(config_path),model_type=cfg.get('model_type'),
   text_model_type=text.get('model_type') if isinstance(text,dict) else None,
   root_num_hash_layers=cfg.get('num_hash_layers','ABSENT'),
   text_num_hash_layers=text.get('num_hash_layers','ABSENT') if isinstance(text,dict) else 'UNKNOWN',
   auto_map=cfg.get('auto_map',None))
if idx is not None:
 weights=idx.get('weight_map')
 if not isinstance(weights,dict) or len(weights)>500000:raise RuntimeError('STOP index schema/size')
 matches={term:sum(term in k.casefold() for k in weights) for term in ('tid2eid','token_to_expert','hash')}
 gate0=sorted(k for k in weights if k.startswith('layers.0.ffn.gate.'))
 gate1=sorted(k for k in weights if k.startswith('layers.1.ffn.gate.'))
 if len(gate0)>30 or len(gate1)>30:raise RuntimeError('STOP unexpected gate keys')
 model_result['checkpoint_index'].update(sha256=sha(index_path),weight_key_count=len(weights),
   name_match_counts=matches,layer0_gate_keys=gate0,layer1_gate_keys=gate1)

packages={};source_results=[]
for pkg in ('transformers','vllm'):
 spec=importlib.util.find_spec(pkg)
 roots=[] if spec is None or spec.submodule_search_locations is None else [Path(p) for p in spec.submodule_search_locations]
 if len(roots)>2:raise RuntimeError('STOP ambiguous package roots')
 packages[pkg]=[str(p) for p in roots]
 for root in roots:
  # Target only model config directories; no full filesystem scan.
  basedir=(root/'models') if pkg=='transformers' else (root/'transformers_utils'/'configs')
  if not basedir.is_dir():continue
  candidates=[]
  if pkg=='transformers':
   for d in basedir.iterdir():
    if d.is_dir() and 'deepseek' in d.name.casefold():candidates.extend(d.glob('configuration*.py'))
  else:candidates=[p for p in basedir.glob('*.py') if 'deepseek' in p.name.casefold()]
  if len(candidates)>128:raise RuntimeError('STOP too many candidate files')
  for p in sorted(candidates):
   if p.stat().st_size>MAX_SRC:raise RuntimeError('STOP oversized source')
   content=p.read_text(); lines=content.splitlines(); tree=ast.parse(content)
   classes=[]
   for node in tree.body:
    if isinstance(node,ast.ClassDef) and ('config' in node.name.casefold() or 'deepseek' in node.name.casefold()):
     classes.append({'name':node.name,'line':node.lineno,'bases':[ast.unparse(v) for v in node.bases]})
   hash_lines=[{'line':i,'text':s.strip()[:240]} for i,s in enumerate(lines,1) if 'num_hash_layers' in s]
   # Content limited to class declarations and the exact target attribute.
   source_results.append({'path':str(p),'sha256':sha(p),'classes':classes,'num_hash_layers_lines':hash_lines})
if len(source_results)>128:raise RuntimeError('STOP excess sources')
print(json.dumps({'scope':'Read model config/index exact paths and only installed deepseek configuration sources; no model weights or NPU',
 'python_executable':sys.executable,'model_metadata':model_result,'package_roots':packages,'config_sources':source_results,
 'limitations':['Disk config/index cannot alone establish historical constructed runtime object','File path absent means not visible in container, not globally absent','Config class source not a runtime instance or loaded tensor proof',
 'No inference of model quality, router outputs, or cause of performance'], 'no_device_work':True},ensure_ascii=False,indent=2))
