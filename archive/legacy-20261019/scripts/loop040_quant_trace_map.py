#!/usr/bin/env python3
"""Map quant-matmul device sequence to layer boundaries in Run107 traces."""
import glob,json,statistics
from pathlib import Path
ROOT=Path('/data/wio/Inference_Foundry')
trace=json.loads((ROOT/'evidence/20260924_loop038_cycle/run107/target_window.json').read_text())
excluded={(x['rank'],x['cycle']) for x in trace['excluded_windows']}
config=json.loads(Path('/data/yxy/DeepSeek-V4-Flash-0731-w4a8/config.json').read_text())
ratios=config['compress_ratios'][:43]
assert len(ratios)==43 and ratios[:2]==[0,0]
qname='aclnnQuantMatmulWeightNz_QuantBatchMatmulV3_QuantBatchMatmulV3'
gname='aclnnGroupedMatmulSwigluQuantWeightNzV2_GroupedMatmulSwigluQuantV2_GroupedMatmulSwigluQuantV2'
expected=[4,5]+[6 if r==4 else 5 for r in ratios[2:]]
assert sum(expected)+1==236
windows=[]
for rank in range(8):
 path=glob.glob(str(ROOT/f'evidence/20260924_loop038_cycle/run107/profile/rank{rank}_*/ASCEND_PROFILER_OUTPUT/trace_view.json'))
 assert len(path)==1,(rank,path)
 events=json.loads(Path(path[0]).read_text())
 scopes=sorted([x for x in events if x.get('name')=='extreme::target' and x.get('cat')=='cpu_op'],key=lambda x:float(x['ts']))
 assert len(scopes)==2,(rank,len(scopes))
 quant=sorted([x for x in events if x.get('name')==qname],key=lambda x:float(x['ts']))
 gmm=sorted([x for x in events if x.get('name')==gname],key=lambda x:float(x['ts']))
 for cycle,scope in zip((64,65),scopes):
  if (rank,cycle) in excluded: continue
  start=float(scope['ts']);end=start+float(scope['dur'])
  q=[x for x in quant if start<=float(x['ts'])<end]
  g=[x for x in gmm if start<=float(x['ts'])<end]
  assert len(q)==236 and len(g)==43,(rank,cycle,len(q),len(g))
  cuts=[start]+[float(x['ts']) for x in g]+[end]
  bins=[[x for x in q if cuts[i]<=float(x['ts'])<cuts[i+1]] for i in range(44)]
  counts=[len(x) for x in bins]
  assert counts[:43]==expected and counts[43]==1,(rank,cycle,counts)
  layer_us=[sum(float(x['dur']) for x in group) for group in bins[:43]]
  windows.append({'rank':rank,'cycle':cycle,'q_kernel_count':len(q),
                  'gmm1_count':len(g),'quant_sum_ms':sum(float(x['dur']) for x in q)/1000,
                  'layer_quant_kernel_counts':counts[:43],
                  'layer_quant_sum_us':layer_us,
                  'tail_quant_kernel_count':counts[43]})
assert len(windows)==15
by_ratio={}
for ratio in (0,4,128):
 xs=[w['layer_quant_sum_us'][i] for w in windows for i,r in enumerate(ratios) if r==ratio]
 by_ratio[str(ratio)]={'layer_count':ratios.count(ratio),'median_quant_sum_us':statistics.median(xs),
                       'min_quant_sum_us':min(xs),'max_quant_sum_us':max(xs)}
out={'run':'run131','source':'Run107 synchronized target trace_view, all 15 valid rank-cycle windows',
     'product_config':'/data/yxy/DeepSeek-V4-Flash-0731-w4a8/config.json',
     'model_layers':43,'compress_ratios':ratios,'expected_quant_kernels_by_layer':expected,
     'tail_quant_kernel_count':1,'quant_kernels_per_target':236,
     'quant_sum_ms':{'median':statistics.median(w['quant_sum_ms'] for w in windows),
                     'min':min(w['quant_sum_ms'] for w in windows),
                     'max':max(w['quant_sum_ms'] for w in windows)},
     'by_ratio':by_ratio,
     'source_call_family':'vllm_ascend/quantization/methods/w8a8_dynamic.py AscendW8A8DynamicLinearMethod.apply runs npu_dynamic_quant then npu_quant_matmul; wq_b may split into two quant_matmul calls when output>=65536.',
     'interpretation':'Kernel count pattern matches 43 model layers and alternating c4/c128 compression ratios, establishing ordered layer-family association. It does not uniquely identify each quant kernel with a Python module or prove repeated x pointer; Run132 bounded capture is needed.',
     'windows':windows}
p=ROOT/'evidence/20260925_loop040_quant/run131'
p.mkdir(parents=True,exist_ok=True)
(p/'trace_map.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ('quant_sum_ms','by_ratio','model_layers','quant_kernels_per_target')},indent=2))
