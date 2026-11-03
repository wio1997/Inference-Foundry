#!/usr/bin/env python3
import base64,json,pathlib,shlex,subprocess
root=pathlib.Path(__file__).resolve().parent
q=root/'binary-binding-draft/second-binding-run'
a=json.loads((q/'engram-copy-producers-recheck.json').read_text());p=json.loads((q/'copy-provenance-audit.json').read_text());copies={}
for sample in p['samples']:
 for c in sample['correlations']:
  for s in c['scopes']:
   for x in s['scopes']:
    if x['name']=='aten::copy_' and any('engram/embedding.py(342)' in (y.get('value') or '') for y in x.get('callchain',[])):copies[int(x['startNs'])]=x
T=[]
for x in a['rows']:
 t=x['copy_start_ns'];c=copies[t];label=x['anchor'];T.extend([
 {'label':label+'-select','name':'aten::select','start_ns':x['select_start_ns'],'end_ns':x['select_end_ns'],'shape':x['select_input_shape']+';;','stack_substring':'model.py(1217): prepare_engram'},
 {'label':label+'-contiguous','name':'aten::contiguous','start_ns':x['contiguous_start_ns'],'end_ns':x['contiguous_end_ns'],'shape':x['copy_shape']+';','stack_substring':'engram/embedding.py(342): embed_gathered'},
 {'label':label+'-copy','name':'aten::copy_','start_ns':t,'end_ns':int(c['endNs']),'shape':c['inputShapes'],'stack_substring':'engram/embedding.py(342): embed_gathered'}])
source=('import json\nTARGETS=json.loads('+repr(json.dumps(T,separators=(',',':')))+')\n'+(root/'extract_engram_raw_stride_inside.py').read_text()).encode();encoded=base64.b64encode(source).decode();inner='import base64;exec(compile(base64.b64decode('+repr(encoded)+'),"<targeted-raw-stride>","exec"))';outer='import subprocess;raise SystemExit(subprocess.call('+repr(['docker','exec','wio-vllm-v41-repro169-stage-20261028','nice','-n','15','python3','-c',inner])+'))';r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',shlex.join(['python3','-c',outer])],capture_output=True,text=True,timeout=600)
if r.returncode:print('safe error',r.stderr[-500:]);raise SystemExit(r.returncode)
v=json.loads(r.stdout);o=root/'engram-raw-stride-metadata.json';o.write_text(json.dumps(v,indent=2)+'\n');print('matched',v['matched_count'],'of',v['target_count'],'type5',sum(x['field5_present'] for x in v['matched']))
