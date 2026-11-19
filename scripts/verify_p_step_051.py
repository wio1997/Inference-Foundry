#!/usr/bin/env python3
"""Verify P rank0 CPU step containment, not request/layer/device completion."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'evidence/p_step_051'
SHAS={'stage_counts.json':'ddeaf429af978d170ea4a7fb480b35e24c9d127dae04f2b12dabe60ed528baa9','step_detail.json':'2ce7a374cdc738efcdcf64bd2d2e2fe813fdcfd4c4da4d5fa74068d2a51156d0'}


def need(v,m):
 if not v:raise ValueError(m)


def load(name):
 b=(ROOT/name).read_bytes();need(hashlib.sha256(b).hexdigest()==SHAS[name],'source SHA');return json.loads(b)


def verify():
 a=load('stage_counts.json');b=load('step_detail.json');old={(r['framework'],r['rank']):r for r in a['ranks']};new={(r['framework'],r['rank']):r for r in b['ranks']};need(set(old)==set(new)=={('sglang',0),('vllm',0)},'rank coverage')
 for key,r in new.items():
  x=old[key]
  for f in ('source_csv_sha256','trace_sha256','target_count','groups','nesting','queue_timing','step_markers_total'):need(r[f]==x[f],'prior mismatch '+f)
  rows=r['step_details'];need(len(rows)==r['step_markers_total'],'marker count')
  need(sum(z['target_count'] for z in rows)==sum(g.get('step_cpu_contains_enqueue_one',0) for g in r['nesting'].values()),'assigned target count')
  for z in rows:need(z['fast_count']+z['slow_count']==z['target_count']==z['device_duration_us']['count']==z['marker_gap_us']['count'],'event counts')
 sg=new[('sglang',0)];ext=[x for x in sg['step_details'] if x['stage']=='EXTEND'];draft=[x for x in sg['step_details'] if x['stage']=='DRAFT'];need(len(ext)==len(draft)==8,'SG step coverage')
 need(all(x['target_count']==14 and x['fast_count']==4 and x['slow_count']==10 and x['step_tokens']==8192 and x['batch_size']==1 for x in ext),'extend profile')
 need(all(x['target_count']==1 and x['fast_count']==1 and x['slow_count']==0 for x in draft),'draft profile')
 vl=new[('vllm',0)];need(vl['step_markers_total']==0 and vl['nesting']['at_or_below_100us']['step_cpu_contains_enqueue_zero']==399,'vLLM unknown')
 return {'sglang_rank0':{'step_markers':16,'extend_markers':8,'draft_markers':8,'extend_fast_per_marker':4,'extend_slow_per_marker':10,'draft_fast_per_marker':1},'vllm_rank0':{'sg_style_step_markers':0,'unmapped_enqueue_candidates':399},'scope':'same PID/tid CPU enqueue interval inside CPU marker; not request mapping, device completion, model layer, or cross-framework step pair'}


if __name__=='__main__':
 try:print(json.dumps(verify(),ensure_ascii=False,indent=2))
 except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as e:raise SystemExit('FAILED: '+str(e))
