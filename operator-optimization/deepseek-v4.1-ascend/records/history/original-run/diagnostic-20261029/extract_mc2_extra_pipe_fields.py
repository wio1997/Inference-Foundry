#!/usr/bin/env python3
"""Read-only extraction of per-task saved profiler fields, no NPU operation."""
import base64, csv, hashlib, json, pathlib, statistics, subprocess
root=pathlib.Path(__file__).resolve().parent
src=root/'target-layer-mc2-16rank.json'
data=json.loads(src.read_text())
want=[]
for rank in data['ranks']:
    if rank['ep'] not in range(8,16): continue
    for layer in rank['layers']:
        if layer['layer'] not in (14,39) or not layer['scheduler_digest'].startswith('1b826883'): continue
        if len(layer['tasks']) != 1: raise RuntimeError('nonunique task')
        c=layer['tasks'][0]['csv']
        want.append({'ep':rank['ep'],'layer':layer['layer'],'rank_dir':rank['rank'],'task':c['Task ID'],'stream':c['Stream ID'],'model':c['Model ID'],'start':c['Start Time(us)'].strip()})
if len(want)!=16: raise RuntimeError('expected 16 scoped tasks')
remote=r'''import csv, hashlib, json, pathlib, sys
items=json.loads(sys.argv[1]); base=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces')
out=[]
for e in items:
 p=base/e['rank_dir']/'ASCEND_PROFILER_OUTPUT/kernel_details.csv'; matches=[]
 with p.open(newline='') as f:
  for row in csv.DictReader(f):
   if row['Name']=='DispatchFFNCombine' and row['Task ID']==e['task'] and row['Stream ID']==e['stream'] and row['Model ID']==e['model'] and row['Start Time(us)'].strip()==e['start']: matches.append(row)
 if len(matches)!=1: raise RuntimeError('identity match count '+str(len(matches))+' '+str(e))
 out.append({'scope':e,'csv_path':str(p),'csv_sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),'row':matches[0]})
print(json.dumps(out))'''
encoded=base64.b64encode(remote.encode()).decode()
args=json.dumps(want,separators=(',',':'))
import shlex
cmd="docker exec wio-vllm-v41-repro169-stage-20261028 python3 -c "+shlex.quote("import base64;exec(base64.b64decode('"+encoded+"'))")+' '+shlex.quote(args)
p=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],check=True,capture_output=True,text=True,timeout=90)
rows=json.loads(p.stdout)
fields=['Duration(us)','aic_mac_ratio','aic_scalar_ratio','aic_mte1_ratio','aic_mte2_ratio','aic_fixpipe_ratio','aiv_vec_ratio','aiv_scalar_ratio','aiv_mte2_ratio','aiv_mte3_ratio']
summary={}
for layer in [14,39]:
 select=[r for r in rows if r['scope']['layer']==layer]
 summary[str(layer)]={}
 for field in fields:
  vals=[float(r['row'][field]) for r in select]
  summary[str(layer)][field]={'min':min(vals),'median':statistics.median(vals),'max':max(vals)}
output={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'scope':'1b826883 digest, EP8-15, exact full device identity, layer 14 and 39; whole fused tasks','method':'remote read-only CSV, no profiler replay or NPU job','ratios':'display precision and aggregation as supplied by existing tool; no sum-to-one assumption','rows':rows,'summary':summary}
tgt=root/'mc2-extra-pipe-fields-16-tasks.json'; tgt.write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(summary,indent=2)); print('saved',tgt)
