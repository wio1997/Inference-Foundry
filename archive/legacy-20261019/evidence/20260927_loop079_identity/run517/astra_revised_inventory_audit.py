import hashlib,importlib.util,json,pathlib,shutil,subprocess,sys,tempfile
sys.dont_write_bytecode=True
R=pathlib.Path('/data/wio/Inference_Foundry');O=R/'evidence/20260927_loop079_identity/run517';S=R/'scripts/loop079_formal_witness_inventory_run516.py';T=R/'evidence/20260927_loop079_identity/run516/formal_witness_inventory.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(S)=='55c823241403b2523ab44628f831238072d7acc1e2b977fc503609bfb38903c9';assert sha(T)=='e2a71531beec4d2d4d086e4b17be50645feaaa636111e0e490d0cd711cbedddc'
spec=importlib.util.spec_from_file_location('revised_inventory',S);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
x=m.build();assert (json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()==T.read_bytes()
cp=subprocess.run([sys.executable,'-B',str(S),'--output',str(O/'rebuild_revised.json')],cwd='/tmp',capture_output=True,text=True);assert cp.returncode==0 and (O/'rebuild_revised.json').read_bytes()==T.read_bytes()
old=json.loads((O/'original_inventory.json').read_text());assert x['inputs']==old['inputs'] and {k:v for k,v in x.items() if k not in ['client_recursive_key_paths','runtime_recursive_key_paths']}==old
ind=json.loads((O/'saved_artifact_audit.json').read_text())
runtime_paths={p[1:].replace('[]','[*]') for p in ind['recursive_path_union']['runtime']};assert set(x['runtime_recursive_key_paths'])==runtime_paths
client_paths={p.removeprefix('.requests[].').replace('[]','[*]') for p in ind['recursive_path_union']['formal_clients'] if p.startswith('.requests[].')};assert set(x['client_recursive_key_paths'])==client_paths
neg=[];original_formal=m.FORMAL
aliases={'client':['request_id','req_id','id','token_ids','output_token_ids','raw_token_ids'],'runtime':['target_argmax','first_target_token','sampled_token_ids','target_generation','selected_replay_generation','scheduler_pre_append_count','pre_append_G','published_token_ids','external_token_ids','last_layer_wo_a_group','first_position_dense_group']}
with tempfile.TemporaryDirectory(prefix='SYNTHETIC_ONLY_V2_',dir=O) as tmp:
 tmp=pathlib.Path(tmp);shutil.copytree(original_formal,tmp/'formal');m.FORMAL=tmp/'formal'
 for group,keys in aliases.items():
  f=m.FORMAL/('bench48_1.json' if group=='client' else 'runtime/rank0_cohort1.json');before=f.read_bytes()
  for k in keys:
   a=json.loads(before);dest=a['requests'][0] if group=='client' else a;dest['synthetic_nested']=[{k:'SYNTHETIC_CANDIDATE'}];f.write_text(json.dumps(a))
   try:m.build()
   except ValueError as exc:assert 'witness field inventory changed' in str(exc);neg.append({'group':group,'nested_single_row_key':k,'reject':True})
   else:raise AssertionError(k)
   f.write_bytes(before)
 # Residual exact-alias limitation: report path but do not invent semantics.
 f=m.FORMAL/'runtime/rank0_cohort1.json';a=json.loads(f.read_text());a['witness']={'generation_uuid':'SYNTHETIC_OTHER_NAME'};f.write_text(json.dumps(a));z=m.build();assert 'witness.generation_uuid' in z['runtime_recursive_key_paths'] and not any(z['witness_fields_present'].values())
m.FORMAL=original_formal
assert m.build()==x
manifest=json.loads((O/'input_sha256_manifest.json').read_text())
for p in [S,T,O/'original_inventory_script.py']:
 manifest[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
for p,info in manifest.items():assert sha(pathlib.Path(p))==info['sha256']
(O/'final_input_sha256_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
result={'verdict':'PASS_FROZEN_REVISED_INVENTORY','script_sha256':sha(S),'output_sha256':sha(T),'byte_identical_build_and_absolute_cli':True,'all132_inputs_unchanged':True,'independent_recursive_union_matches':True,'negative_controls':neg,'negative_count':len(neg),'residual_alias_control':{'unknown_name':'witness.generation_uuid','path_reported':True,'not_semantically_recognized':True},'scope':'saved inspected artifacts only, no universal absence claim','input_count':len(manifest)}
(O/'revised_audit_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'verdict':result['verdict'],'negative_controls':len(neg),'input_count':len(manifest)}))
