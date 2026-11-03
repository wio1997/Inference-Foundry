#!/usr/bin/env python3
"""Read-only extraction: remote CPU and saved artifacts only, output anonymous JSON."""
import base64, json, shlex, subprocess
from pathlib import Path

remote = r'''
import collections, gzip, hashlib, json, pathlib, re, statistics
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/official-20261030T084726Z-cb252d1d')
def load(q): return json.loads(q.read_text())
def sha(q): return hashlib.sha256(q.read_bytes()).hexdigest()
out={'run_id':p.name,'phases':{},'frontend':[],'source_sha256':{}}
for phase in ('warmup','baseline','profile'):
 q=p/'runtime'/('client-'+phase); sf=q/'window-status.json'; rf=q/'responses.json'; s=load(sf); rr=load(rf)
 rows=[]
 for x in rr:
  first=x.get('first_nonempty_ns'); chunks=[v['at_ns'] for v in x['chunks'] if v['nonempty']]
  rows.append({'start_ns':x['start_ns'],'first_nonempty_ns':first,'end_ns':x['end_ns'],'prompt_tokens':x.get('usage',{}).get('prompt_tokens'),'completion_tokens':x.get('usage',{}).get('completion_tokens'),'nonempty_chunk_count':len(chunks),'chunk_gaps_ns':[b-a for a,b in zip(chunks,chunks[1:])]})
 start=min(x['start_ns'] for x in rows); end=max(x['end_ns'] for x in rows)
 out['phases'][phase]={'valid':s.get('valid'),'input_sha256':s.get('input_sha256'),'counter_deltas':s.get('counter_deltas'),'counter_deltas_by_engine':s.get('counter_deltas_by_engine'),'windows':s.get('windows'),'decode_gap':s.get('decode_gap'),'start_ns':start,'end_ns':end,'requests':rows}
 out['source_sha256'][str(sf.relative_to(p))]=sha(sf);out['source_sha256'][str(rf.relative_to(p))]=sha(rf)
samples=p/'npu-smi-samples.json';out['source_sha256'][str(samples.relative_to(p))]=sha(samples)
pat=re.compile(r'^\|\s*(\d+)\s+(\d+)\s*\|\s*([0-9A-Fa-f:.]+)\s*\|\s*(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)\s*\|')
for phase,d in out['phases'].items():
 matches=[]
 for rec in load(samples):
  if d['start_ns']<=rec['monotonic_ns']<=d['end_ns']:
   chips=[]
   for line in rec.get('npu_smi','').splitlines():
    m=pat.match(line)
    if m: chips.append({'phy_id':int(m[2]),'aicore_pct':int(m[4]),'hbm_used_display_mb':int(m[5]),'hbm_total_display_mb':int(m[6])})
   matches.append({'monotonic_ns':rec['monotonic_ns'],'error':rec.get('error'),'chips':chips})
 d['npu_samples']=matches
for f in sorted((p/'runtime'/'official-traces').glob('*.async_llm.*.gz')):
 with gzip.open(f,'rt') as stream: trace=json.load(stream)
 ev=trace.get('traceEvents',[]); out['frontend'].append({'file':f.name,'bytes':f.stat().st_size,'sha256':sha(f),'ph_counts':dict(collections.Counter(x.get('ph') for x in ev)),'name_counts':dict(collections.Counter(x.get('name') for x in ev)),'duration_us':[x.get('dur') for x in ev if x.get('ph')=='X']})
for n in ('comparison-validation.json','runtime-versions.json','controller-state.json'):
 q=p/n;v=load(q);out[n]={'state':v.get('state'),'utc':v.get('utc')} if n=='controller-state.json' else v;out['source_sha256'][n]=sha(q)
print(json.dumps(out,separators=(',',':')))
'''
encoded=base64.b64encode(remote.encode()).decode()
command=shlex.join(['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',f'import base64;exec(base64.b64decode({encoded!r}))'])
result=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',command],capture_output=True,text=True,timeout=120,check=True)
out=json.loads(result.stdout)
path=Path(__file__).with_name('workload-framework.json')
path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(path)
