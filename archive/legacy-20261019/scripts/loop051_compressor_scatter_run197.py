#!/usr/bin/env python3
"""Run107 exact DSA compressor-following-scatter source/trace screen."""
import glob,json,statistics
from collections import Counter
from pathlib import Path
root=Path(__file__).resolve().parents[1]
prior=json.loads((root/'evidence/20260924_loop038_cycle/run107/target_window.json').read_text())
excluded={(x['rank'],x['cycle']) for x in prior['excluded_windows']}
records=[]
for rank in range(8):
 p=glob.glob(str(root/f'evidence/20260924_loop038_cycle/run107/profile/rank{rank}_*/ASCEND_PROFILER_OUTPUT/trace_view.json'))
 assert len(p)==1
 events=json.loads(Path(p[0]).read_text())
 scopes=sorted((e for e in events if e.get('name')=='extreme::target' and e.get('cat')=='cpu_op'),key=lambda e:float(e['ts']))
 for cycle,scope in zip((64,65),scopes):
  if (rank,cycle) in excluded:continue
  start=float(scope['ts']);end=start+float(scope['dur'])
  kernels=sorted((e for e in events if start<=float(e.get('ts',-1))<end and e.get('args',{}).get('Task Type','').startswith('KERNEL_')),key=lambda e:float(e['ts']))
  sparse=[i for i,e in enumerate(kernels) if e['name']=='SparseAttnSharedkv']
  assert len(sparse)==43
  prev=-1
  for layer,idx in enumerate(sparse):
   seg=kernels[prev+1:idx+1]
   compressors=[i for i,e in enumerate(seg) if e['name']=='Compressor']
   cls={0:'c0',1:'c128',2:'c4'}[len(compressors)]
   scatters=[i for i,e in enumerate(seg) if e['name'].startswith('aclnnScatterNdUpdateSk_ScatterNdUpdateSkAiCore')]
   # Class segments also include SWA scatter (before compressors) and c4 indexer scatter.
   after=[]
   for ci in compressors:
    next_ci=next((j for j in compressors if j>ci),len(seg))
    match=[j for j in scatters if ci<j<next_ci]
    if match:after.append(match[0])
   records.append({'rank':rank,'cycle':cycle,'layer':layer,'class':cls,'compressor_count':len(compressors),'scatter_count':len(scatters),'matched_compressor_scatter_count':len(after),'matched_compressor_scatter_ms':sum(float(seg[j]['dur']) for j in after)/1000,'all_scatter_ms':sum(float(seg[j]['dur']) for j in scatters)/1000,'scatter_positions':scatters,'compressor_positions':compressors})
   prev=idx
assert len(records)==645
windows=[]
for rank,cycle in sorted(set((x['rank'],x['cycle']) for x in records)):
 x=[r for r in records if r['rank']==rank and r['cycle']==cycle]
 windows.append({'rank':rank,'cycle':cycle,'c128_compressor_scatter_ms':sum(r['matched_compressor_scatter_ms'] for r in x if r['class']=='c128'),'c4_compressor_scatter_ms':sum(r['matched_compressor_scatter_ms'] for r in x if r['class']=='c4'),'all_matched_compressor_scatter_ms':sum(r['matched_compressor_scatter_ms'] for r in x),'all_scatter_ms':sum(r['all_scatter_ms'] for r in x),'c128_matched_count':sum(r['matched_compressor_scatter_count'] for r in x if r['class']=='c128'),'c4_matched_count':sum(r['matched_compressor_scatter_count'] for r in x if r['class']=='c4')})
def desc(key):
 v=[x[key] for x in windows];return {'median':statistics.median(v),'min':min(v),'max':max(v)}
result={'run':'run197','source':'Run107 15 valid profiled target windows; borrowed dsa_v1.py decode compressor->scatter and DeviceOperator non-A5 ScatterNdUpdateSk','counts':{'class_rows':dict(Counter(x['class'] for x in records)),'c128_compressor_scatter_per_window':desc('c128_matched_count'),'c4_compressor_scatter_per_window':desc('c4_matched_count')},'duration_ms':{k:desc(k) for k in ('c128_compressor_scatter_ms','c4_compressor_scatter_ms','all_matched_compressor_scatter_ms','all_scatter_ms')},'windows':windows,'sample_rows':records[:10],'source_findings':['c128 decode path: compressor outputs a dense cmp_kv tensor; DeviceOperator.dsa_kv_compress_scatter invokes npu_scatter_nd_update_sk with final cache and slot mapping.','Compressor op has explicit cmp_kv and state_cache outputs but no compressed-KV cache or slot mapping input; direct destination write requires custom-op ABI/kernel/tiling work, not a Python substitution.','c4 has two distinct compressor products and additional indexer scatter; preserve both state writes and QLI dependency.'],'limits':['Matched task durations are profiled local kernel sum, not unprofiled target saving.','A fused compressor may still pay per-slot address calculation and cache write; at most some intermediate traffic/launch can disappear.','Source inspection does not yet prove alias/write-set or local numerical feasibility.']}
out=root/'evidence/20260925_loop051_compressor_cache/run197/analysis.json';out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'counts':result['counts'],'duration_ms':result['duration_ms']},indent=2))
