import hashlib,json,re,subprocess
from pathlib import Path
ROOT=Path('/data/wio/Inference_Foundry')
OUT=ROOT/'evidence/20260927_loop079_identity/run510'
OUT.mkdir(parents=True,exist_ok=True)
C='vllm-ascend26-dsv4f-w4a8'
def command(*args): return subprocess.check_output(args)
def cat(path): return command('docker','exec',C,'cat',path)
def sha(b): return hashlib.sha256(b).hexdigest()
installed={}
base='/usr/local/Ascend/cann-9.1.0/aarch64-linux'
for p in [base+'/lib64/libascendcl.so',base+'/lib64/libruntime.so',base+'/lib64/libruntime_v100.so',base+'/include/version/runtime_version.h',base+'/include/acl/acl_rt.h',base+'/ascend_toolkit_install.info']:
 b=cat(p);installed[p]={'sha256':sha(b),'bytes':len(b)}
 if not p.endswith('.so'):(OUT/'source'/Path(p).name).write_bytes(b)
obj='/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b/ops_legacy/reduce_mean/ReduceMean_9a2479d4dc5148645cb3f4b9e765c637_high_precision.o'
b=cat(obj);installed[obj]={'sha256':sha(b),'bytes':len(b)}
assert sha(b)=='c549abdc92fe1cbc446fb47872c32112973266b5218948bf88318db91f0f96db'
(OUT/'source'/'ReduceMean_high_precision.o').write_bytes(b)
sec='.ascend.meta.ReduceMean_9a2479d4dc5148645cb3f4b9e765c637_high_precision_2100900_mix_aiv'
for label,args in [('kernel_symbols',['readelf','-sW',obj]),('kernel_metadata_hex',['readelf','-x',sec,obj]),('kernel_disassembly',['/usr/local/Ascend/cann-9.1.0/tools/bisheng_compiler/bin/llvm-objdump','-d','--start-address=474328','--stop-address=474580',obj])]:
 (OUT/(label+'.txt')).write_bytes(command('docker','exec',C,*args))
(OUT/'installed_files.json').write_text(json.dumps(installed,indent=2)+'\n')
refs=['run503/astra_run502_graph_review.md','run503/independent_results.json','run503/mean_kernel.json','run503/reduce_mean.py','run503/rms_kernel.json','run503/rms_norm.cpp','run503/deepseek_v4.py','run506_source_only.md','run469/astra_frontier_review.md','run469/source/run469_NPUGraph.cpp','run469/source/run469_AclInterface.cpp','run469/source/run469_NPUStream.cpp','run469/source/run469_NPUEvent.cpp','run469/source/run469_execute.html','run469/source/run469_cross_capture.html','run469/source/run469_record_actual.html']
e=ROOT/'evidence/20260927_loop079_identity'
inputs={str(e/x):sha((e/x).read_bytes()) for x in refs}
rows=[]
for p in sorted((e/'run502/b_candidate/graph_dump').glob('*.json')):
 inputs[str(p)]=sha(p.read_bytes())
 if p.name.endswith('.meta.json'):continue
 tasks=json.loads(p.read_text());chosen=[]
 for task in tasks:
  a=task['args']
  if a['Stream Id']!=1 or a['Task Id'] not in (2788,2848,2925,2944,3984):continue
  d={'name':task['name'],'args':a}
  if 'Kernel Args' in a:
   tokens=a['Kernel Args'].split();pat=re.compile(r'(?P<label>\(placeHolder(?:Addr|Data)\d+\))?(?P<word>0x[0-9a-fA-F]{16})')
   matches=[pat.fullmatch(t) for t in tokens];assert all(matches)
   assert len(tokens)*8==a['Kernel Args Size']
   d['words']=[{'offset':i*8,'label':m['label'],'word':m['word']} for i,m in enumerate(matches)]
   if a['Task Id']!=2944:
    assert len(tokens)==20 and matches[3]['label'] is None
    assert matches[2]['label']=='(placeHolderAddr0)' and matches[6]['label']=='(placeHolderAddr1)'
  chosen.append(d)
 assert len(tasks)==5412 and len(chosen)==5
 rows.append({'file':str(p),'tasks':len(tasks),'selected':chosen})
(OUT/'raw_argument_checks.json').write_text(json.dumps(rows,indent=2)+'\n')
(OUT/'review_input_hashes.json').write_text(json.dumps(inputs,indent=2,sort_keys=True)+'\n')
print(json.dumps({'ranks':len(rows),'raw_tasks_per_rank':5412,'selected_per_rank':5,'inputs_hashed':len(inputs),'installed_files':installed},indent=2))
