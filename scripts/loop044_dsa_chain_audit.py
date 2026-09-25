#!/usr/bin/env python3
"""Map target DSA compressor/indexer/sparse sequence to product layer classes."""
import glob,json,statistics
from collections import Counter
from pathlib import Path
root=Path(__file__).resolve().parents[1]
prior=json.loads((root/'evidence/20260924_loop038_cycle/run107/target_window.json').read_text())
excluded={(x['rank'],x['cycle']) for x in prior['excluded_windows']}
rows=[]
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
   counts=Counter(e['name'] for e in seg)
   cm=counts['Compressor'];im=counts['VllmQuantLightningIndexer'];md=counts['CompressorMetadata']
   assert (cm,im) in ((0,0),(1,0),(2,1)),(rank,cycle,layer,cm,im,md)
   next_slice=kernels[idx+1:idx+8]
   transpose=next((j for j,e in enumerate(next_slice) if e['name'].startswith('aclnnTransposeBatchMatMul')),None)
   alltoall_before=transpose is not None and any(e['name']=='hcom_alltoall_' for e in next_slice[:transpose])
   rows.append({'rank':rank,'cycle':cycle,'layer':layer,'compressor':cm,'indexer':im,'metadata':md,
    'class':{(0,0):'c0',(1,0):'c128',(2,1):'c4'}[(cm,im)],
    'transpose_within_next7':transpose is not None,'alltoall_before_transpose':alltoall_before,
    'compressor_ms':sum(float(e['dur']) for e in seg if e['name']=='Compressor')/1000,
    'indexer_ms':sum(float(e['dur']) for e in seg if e['name']=='VllmQuantLightningIndexer')/1000,
    'sparse_ms':float(kernels[idx]['dur'])/1000})
   prev=idx
assert len(rows)==15*43
classes=Counter(r['class'] for r in rows)
assert classes=={'c0':30,'c4':315,'c128':300},classes
def desc(v):return {'median':statistics.median(v),'min':min(v),'max':max(v)}
by_window=[]
for rank,cycle in sorted(set((r['rank'],r['cycle']) for r in rows)):
 x=[r for r in rows if r['rank']==rank and r['cycle']==cycle]
 by_window.append({'rank':rank,'cycle':cycle,'compressor_ms':sum(r['compressor_ms'] for r in x),'indexer_ms':sum(r['indexer_ms'] for r in x),'sparse_ms':sum(r['sparse_ms'] for r in x)})
out={'run':'run151','source':'Run107 15 valid target windows; borrowed vllm_ascend/attention/dsa_v1.py decode source',
 'layer_class_counts':dict(classes),
 'per_window_compressor_ms':desc([w['compressor_ms'] for w in by_window]),
 'per_window_indexer_ms':desc([w['indexer_ms'] for w in by_window]),
 'per_window_sparse_ms':desc([w['sparse_ms'] for w in by_window]),
 'transpose_after_sparse_count':sum(r['transpose_within_next7'] for r in rows),
 'alltoall_before_transpose_count':sum(r['alltoall_before_transpose'] for r in rows),
 'source_mapping':{'c4':'two distinct compressor calls: attention compressed-KV state and indexer compressed-KV state, then lightning indexer and sparse attention',
                   'c128':'one attention compressed-KV compressor then sparse attention; no lightning indexer',
                   'c0':'sparse attention without compressor/indexer',
                   'transpose':'appears after sparse attention and HCCL alltoall in trace; not part of pre-attention DSA chain'},
 'rows':rows,'windows':by_window,
 'decision':'The 62 Compressor invocations are exactly 21 c4*2 +20 c128*1 and correspond to distinct mandatory state/cache products, not duplicate recomputation. Indexer topk feeds c4 sparse attention. Trace/source do not expose a semantics-preserving redundant intermediate or source-level >=5ms fusion. Do not sum mandatory DSA stages as removable gain.'}
p=root/'evidence/20260925_loop044_target/run151/dsa_chain_audit.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('rows','windows')},indent=2))
