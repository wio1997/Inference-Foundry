#!/usr/bin/env python3
"""Validate 21 selected P device kernels' CPU-step candidate coverage."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCES={'evidence/p_heavy_052/stage.json':'833fd119ccdb33ed17405ef625d97beabb87f3abd6b0c606d9fb437aebe9e09a','evidence/p_heavy_052/step_detail.json':'363153a544c94cee7bf5d7ba4495ef796bacf4244d64e29edbc94cee9c6ef6cd','evidence/p_step_051/step_detail.json':'2ce7a374cdc738efcdcf64bd2d2e2fe813fdcfd4c4da4d5fa74068d2a51156d0'}
SG_PER_EXTEND={'chunk_kda_fwd':34,'SparseFlashAttention':11,'_glm5_next_lightning_indexer_score_kernel':11,'CamMoeDispatchNormal':42,'CamMoeCombineNormal':42,'aclnnGroupedMatmulWeightNz_GroupedMatmul_GroupedMatmul':84,'_swiglu_quant_kernel':42,'aclnnDynamicQuantV2_DynamicQuant_DynamicQuant':123}
SG_PER_DRAFT={'chunk_kda_fwd':0,'SparseFlashAttention':1,'_glm5_next_lightning_indexer_score_kernel':1,'CamMoeDispatchNormal':1,'CamMoeCombineNormal':1,'aclnnGroupedMatmulWeightNz_GroupedMatmul_GroupedMatmul':2,'_swiglu_quant_kernel':1,'aclnnDynamicQuantV2_DynamicQuant_DynamicQuant':5}


def need(v,m):
 if not v:raise ValueError(m)


def load(path):
 b=(ROOT/path).read_bytes();need(hashlib.sha256(b).hexdigest()==SOURCES[path],'SHA mismatch '+path);return json.loads(b)


def verify():
 a=load('evidence/p_heavy_052/stage.json');b=load('evidence/p_heavy_052/step_detail.json');p=load('evidence/p_step_051/step_detail.json');old={(r['framework'],r['rank']):r for r in a['ranks']};new={(r['framework'],r['rank']):r for r in b['ranks']};prior={(r['framework'],r['rank']):r for r in p['ranks']};need(set(old)==set(new)==set(prior)=={('sglang',0),('vllm',0)},'rank coverage');summary={}
 for key,r in new.items():
  for field in ('source_csv_sha256','trace_sha256','selected_kernel_count','kernels','step_markers_total'):need(r[field]==old[key][field],'stage reread mismatch')
  steps=r['step_details'];need(len(steps)==r['step_markers_total']==len(prior[key]['step_details']),'step marker count')
  for x,y in zip(steps,prior[key]['step_details']):
   for field in ('ordinal','stage','cpu_marker_duration_ms','context_kv_tokens','step_tokens','batch_size','forward_id_recorded'):need(x[field]==y[field],'step metadata mismatch')
  for x in r['kernels']:
   s=x['counts'];n=s['csv_events'];need(sum(s.get(k,0) for k in ('incomplete_key','csv_duplicate_composite_key','device_zero','device_multiple','device_unique'))==n,'device partition')
   need(sum(s.get(k,0) for k in ('device_duration_outside_tolerance','connection_missing','node_zero','node_multiple','node_unique'))==s.get('device_unique',0),'node partition')
   need(sum(s.get(k,0) for k in ('dequeue_zero','dequeue_multiple','dequeue_unique'))==s.get('node_unique',0),'dequeue partition')
   need(sum(s.get(k,0) for k in ('correlation_missing','queue_nonunique_or_missing','typed_flow_nonunique_or_missing','typed_flow_unique'))==s.get('dequeue_unique',0),'queue partition')
   stages=('EXTEND','DRAFT','VERIFY','TARGET_VERIFY','DECODE','OTHER');need(s.get('cpu_step_zero',0)+s.get('cpu_step_multiple',0)+sum(s.get('cpu_step_'+q,0) for q in stages)==s.get('typed_flow_unique',0),'step partition')
   rows=[y for st in steps for y in st['kernels'] if y['kernel_name']==x['kernel_name']];need(sum(y['count'] for y in rows)==sum(v['count'] for v in x['cpu_step_device_event_stats'].values()),'step aggregation')
  if key[0]=='sglang':
   ext=[x for x in steps if x['stage']=='EXTEND'];dr=[x for x in steps if x['stage']=='DRAFT'];need(len(ext)==len(dr)==8,'SG step roles')
   for rows,expected in ((ext,SG_PER_EXTEND),(dr,SG_PER_DRAFT)):
    for row in rows:
     by={x['kernel_name']:x['count'] for x in row['kernels']};need(all(by.get(k,0)==v for k,v in expected.items()),'per-step operator counts')
   summary['sglang']={'selected_kernels':len(r['kernels']),'extend_steps':len(ext),'draft_steps':len(dr),'indexer_8192_p50_us':[round(next(k['p50_us'] for k in x['kernels'] if k['kernel_name']=='_glm5_next_lightning_indexer_score_kernel'),3) for x in ext if x['context_kv_tokens']==8192],'indexer_32768_p50_us':[round(next(k['p50_us'] for k in x['kernels'] if k['kernel_name']=='_glm5_next_lightning_indexer_score_kernel'),3) for x in ext if x['context_kv_tokens']==32768]}
  else:
   need(not steps and all(x['counts'].get('cpu_step_zero',0)==x['counts'].get('typed_flow_unique',0) for x in r['kernels']),'vLLM stage unknown')
   summary['vllm']={'selected_kernels':len(r['kernels']),'sg_style_step_markers':0,'dequeue_missing':sum(x['counts'].get('dequeue_zero',0) for x in r['kernels'])}
 return {'rank0':summary,'scope':'CPU enqueue containment candidates, not per-request, per-layer, or device critical path; vLLM SG-style stage unknown'}


if __name__=='__main__':
 try:print(json.dumps(verify(),ensure_ascii=False,indent=2))
 except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as e:raise SystemExit('FAILED: '+str(e))
