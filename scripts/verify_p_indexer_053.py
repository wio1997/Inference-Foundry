#!/usr/bin/env python3
"""Validate indexer/attention shape-dtype-duration and CPU-step candidates."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE='evidence/p_indexer_053/joint.json';SHA='1842ca83d8878351edfd270dce940ca80684b2733bf67ec6e57cd0ba44ce03f5'
PRIOR='evidence/p_heavy_052/step_detail.json';PRIOR_SHA='363153a544c94cee7bf5d7ba4495ef796bacf4244d64e29edbc94cee9c6ef6cd'
NAMES=('SparseFlashAttention','_glm5_next_lightning_indexer_score_kernel')


def need(v,m):
 if not v:raise ValueError(m)


def load(path,sha):
 b=(ROOT/path).read_bytes();need(hashlib.sha256(b).hexdigest()==sha,'source SHA mismatch');return json.loads(b)


def verify():
 d=load(SOURCE,SHA);p=load(PRIOR,PRIOR_SHA);new={(r['framework'],r['rank']):r for r in d['ranks']};old={(r['framework'],r['rank']):r for r in p['ranks']};need(set(new)==set(old)=={('sglang',0),('vllm',0)},'rank coverage');by={}
 for key,r in new.items():
  o=old[key];need(all(r[t]==o[t] for t in ('source_csv_sha256','trace_sha256','step_markers_total')),'source and marker binding');kernels={x['kernel_name']:x for x in r['kernels']};prior={x['kernel_name']:x for x in o['kernels']};need(set(kernels)==set(NAMES),'kernel coverage');by[key]=kernels
  for name,x in kernels.items():
   y=prior[name];need(x['counts']==y['counts'] and x['cpu_step_device_event_stats']==y['cpu_step_device_event_stats'],'candidate counts');need(sum(g['distribution']['count'] for g in x['all_csv_shape_distributions'])==x['counts']['csv_events'],'shape distribution count')
  need(len(r['step_details'])==len(o['step_details']),'step detail count')
  for x,y in zip(r['step_details'],o['step_details']):
   need(all(x[t]==y[t] for t in ('ordinal','stage','cpu_marker_duration_ms','context_kv_tokens','step_tokens','batch_size','forward_id_recorded')),'step identity')
   py={q['kernel_name']:q for q in y['kernels']}
   for z in x['kernels']:
    need(all(z[t]==py[z['kernel_name']][t] for t in ('count','sum_us_nonwallclock','p50_us')),'step duration binding');need(sum(s['count'] for s in z['shape_joint'])==z['count'],'step shape coverage')
 for name in NAMES:
  left={x['input_shape'] for x in by[('sglang',0)][name]['all_csv_shape_distributions']};right={x['input_shape'] for x in by[('vllm',0)][name]['all_csv_shape_distributions']};need(not left&right,'unexpected literal intersection')
 sg=new[('sglang',0)];ext=[x for x in sg['step_details'] if x['stage']=='EXTEND'];draft=[x for x in sg['step_details'] if x['stage']=='DRAFT'];need(len(ext)==len(draft)==8,'SG stages')
 for marker in ext+draft:
  op=next(x for x in marker['kernels'] if x['kernel_name']==NAMES[1]);j=op['shape_joint'];need(len(j)==1 and j[0]['count']==(11 if marker['stage']=='EXTEND' else 1),'indexer per-step shape coverage')
  dims=j[0]['input_shape'].split(';')[1].split(',');need(len(dims)==4 and dims[0]=='1' and int(dims[1])*4==marker['context_kv_tokens'],'recorded context/slot relation')
 sg_i=by[('sglang',0)][NAMES[1]]['all_csv_shape_distributions'];vl_i=by[('vllm',0)][NAMES[1]]['all_csv_shape_distributions'];need(len(sg_i)==4 and len(vl_i)==14,'indexer groups')
 return {'sg_indexer_groups':len(sg_i),'vllm_indexer_groups':len(vl_i),'sg_attention_groups':len(by[('sglang',0)][NAMES[0]]['all_csv_shape_distributions']),'vllm_attention_groups':len(by[('vllm',0)][NAMES[0]]['all_csv_shape_distributions']),'same_full_recorded_input_shape_intersection':0,'sg_steps_indexer_shape_matches_csk_rule':16,'scope':'one non-scored capture/rank, recorded strings only; no effective work, full shape semantics, request or causal attribution'}


if __name__=='__main__':
 try:print(json.dumps(verify(),ensure_ascii=False,indent=2))
 except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError,StopIteration) as e:raise SystemExit('FAILED: '+str(e))
