"""Read a named quantization-description JSON only; never read weight tensors."""
import collections,hashlib,json,pathlib,re,subprocess,sys
j=json.loads(sys.argv[1]);root=pathlib.Path('/sgl-workspace/sglang');head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();assert head==j['expected_commit'];source=root/j['modelslim_source'];b=source.read_bytes();assert hashlib.sha256(b).hexdigest()==j['modelslim_sha256'] and subprocess.check_output(['git','-C',str(root),'show',head+':'+j['modelslim_source']])==b
model=pathlib.Path('/model').resolve();config=model/'config.json';config_bytes=config.read_bytes();assert hashlib.sha256(config_bytes).hexdigest()==j['config_sha256'];cfg=json.loads(config_bytes);target_layers=cfg['num_hidden_layers'];path=(model/'quant_model_description.json').resolve();assert path.is_relative_to(model)
v={'source_commit':head,'files':{},'read_only':True,'generation_requests':0,'native_executed':False,'model_weights_read':False,'config_sha256':j['config_sha256'],'modelslim_source_sha256':j['modelslim_sha256'],'expected_target_layers_from_config':target_layers,'metadata_path':str(path),'raw_file_persisted':False,'scope':'Artifact quantization-description only; not actual task backend, loaded weight dtype or runtime layout. Post-target layer labels are index-based candidates.'}
if not path.is_file():v['probe_status']='UNAVAILABLE_METADATA_FILE_MISSING'
elif path.stat().st_size>j['max_bytes']:v.update(probe_status='UNAVAILABLE_METADATA_TOO_LARGE',bytes=path.stat().st_size)
else:
 raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest();desc=json.loads(raw);v.update(bytes=len(raw),sha256=digest)
 if not isinstance(desc,dict):v.update(probe_status='UNAVAILABLE_UNSUPPORTED_JSON_TYPE',json_type=type(desc).__name__)
 else:
  counts=collections.Counter();layers={};examples={};value_types=collections.Counter(type(x).__name__ for x in desc.values())
  for key,value in desc.items():
   if not isinstance(key,str) or not key.endswith('.weight'):continue
   m=re.search(r'(?:^|\.)layers\.(\d+)\.',key)
   if m is None:continue
   layer=int(m.group(1));role='target' if layer<target_layers else 'post_target_layer_candidate'
   component='shared' if ('.shared_experts.' in key or '.shared_expert.' in key) else ('routed' if '.experts.' in key else None)
   if component is None:continue
   scheme=value if isinstance(value,str) and re.fullmatch(r'[A-Z][A-Z0-9_]{0,63}',value) else 'OTHER_'+type(value).__name__
   group=role+'/'+component;counts[group+'/'+scheme]+=1;layers.setdefault(str(layer),{}).setdefault(component,{})[scheme]=layers.setdefault(str(layer),{}).setdefault(component,{}).get(scheme,0)+1
   if len(examples.setdefault(group,[]))<4:examples[group].append({'key':key,'scheme':scheme})
  v.update(probe_status='MODEL_QUANT_METADATA_FOUND',top_level_entries=len(desc),value_type_counts=dict(value_types),moe_weight_scheme_counts=dict(counts),per_layer_moe_weight_counts=layers,examples=examples)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==digest;v['metadata_unchanged']=True
assert hashlib.sha256(config.read_bytes()).hexdigest()==j['config_sha256'];v['config_unchanged']=True;print(json.dumps(v))
