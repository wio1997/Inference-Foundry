#!/usr/bin/env python3
"""Verify target same-name connection and Node count; not a causal graph."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'evidence/p_flow_050'
SHA='1628886d6306b8d25304f07004750ad9a98b21875ccb4ced20010791583a7f22'
NESTING_SHA='0cdf3d9ebc06e7878351bc22b678b97847f20f0f0bcf75e817b6b9f1134db940'
CPU_SHA='867ab9ce01bd7a44d4708ee829f0c2a6ed2ca7a36d3f9a4bdc6ecc9b7e01900e'
NAMES_SHA='81274624dd176c0cf5065a64d17e870fda3c18b68d7857971e18e5ee573d6bd0'
TIMING_SHA='be25673a2b017e34359508d8a7e62b9958843a5bfc135743ab06ee309fa18690'
DIMS=('selected_targets_per_connection','hardware_target','hardware_other','node_target','node_other_or_missing_item','cpu_op_same_named_connection','enqueue_same_named_connection','dequeue_same_named_connection','flow_start_same_named_connection','flow_finish_same_named_connection')


def need(c,m):
 if not c:raise ValueError(m)


def verify():
 raw=(ROOT/'connection.json').read_bytes();nr=(ROOT/'nesting.json').read_bytes();cr=(ROOT/'cpu_queue.json').read_bytes();pr=(ROOT/'cpu_names.json').read_bytes();tr=(ROOT/'queue_timing.json').read_bytes();need(hashlib.sha256(raw).hexdigest()==SHA and hashlib.sha256(nr).hexdigest()==NESTING_SHA and hashlib.sha256(cr).hexdigest()==CPU_SHA and hashlib.sha256(pr).hexdigest()==NAMES_SHA and hashlib.sha256(tr).hexdigest()==TIMING_SHA,'hash mismatch');d=json.loads(raw);nd=json.loads(nr);cd=json.loads(cr);pd=json.loads(pr);td=json.loads(tr);r={(x['framework'],x['rank']):x for x in d['ranks']};nmap={(x['framework'],x['rank']):x for x in nd['ranks']};cmap={(x['framework'],x['rank']):x for x in cd['ranks']};pmap={(x['framework'],x['rank']):x for x in pd['ranks']};tmap={(x['framework'],x['rank']):x for x in td['ranks']};need(set(r)==set(nmap)==set(cmap)==set(pmap)==set(tmap)=={('sglang',0),('vllm',0)},'rank coverage');out={}
 for fw,n in (('sglang',120),('vllm',399)):
  x=r[(fw,0)];need(x['target_count']==x['unique_device_candidates']==x['distinct_numeric_connection_count']==n,'target count')
  groups=x['groups'];need(sum(g['events'] for g in groups.values())==n,'partition count')
  for g in groups.values():
   count=g['events'];need(g['connection_available']==count and g['connection_unavailable']==0,'connection availability')
   for k in DIMS:need(sum(g[k].values())==count,'bucket denominator')
   for k in ('selected_targets_per_connection','hardware_target','node_target'):need(g[k]['one']==count,'expected one-to-one sample relation')
   for k in ('hardware_other','node_other_or_missing_item','cpu_op_same_named_connection','enqueue_same_named_connection','dequeue_same_named_connection','flow_start_same_named_connection','flow_finish_same_named_connection'):need(g[k]['zero']==count,'unexpected same-name connection association')
  nx=nmap[(fw,0)];need(nx['source_csv_sha256']==x['source_csv_sha256'] and nx['trace_sha256']==x['trace_sha256'] and nx['groups']==x['groups'],'nesting source/groups mismatch')
  for label,g in nx['nesting'].items():
   cnt=x['groups'][label]['events'];need(g.get('events',0)==cnt and g.get('node_timing_unique',0)==cnt and g.get('dequeue_containment_one',0)==cnt and g.get('unique_dequeue_name_dynamicquant',0)==cnt and g.get('unique_dequeue_enqueue_correlation_one',0)==cnt,'queue nesting count')
  cx=cmap[(fw,0)];need(cx['source_csv_sha256']==x['source_csv_sha256'] and cx['trace_sha256']==x['trace_sha256'] and cx['groups']==x['groups'],'CPU source/groups mismatch')
  for label,g in cx['nesting'].items():
   count=x['groups'][label]['events']
   for key in ('node_timing_unique','dequeue_containment_one','unique_dequeue_name_dynamicquant','unique_dequeue_enqueue_correlation_one','unique_enqueue_timing_available','unique_enqueue_name_dynamicquant','dynamic_cpu_op_contains_enqueue_multiple','flow_endpoints_both_unique'):
    need(g.get(key,0)==count,'CPU/flow candidate coverage')
  px=pmap[(fw,0)];need(px['source_csv_sha256']==x['source_csv_sha256'] and px['trace_sha256']==x['trace_sha256'] and px['groups']==x['groups'],'CPU name source/groups mismatch')
  for label,g in px['nesting'].items():
   count=x['groups'][label]['events']
   for key in ('shortest_cpu_interval_tie_one','shortest_cpu_kind_aclnnDynamicQuantV2','containing_cpu_kind_aclnnDynamicQuantV2','containing_cpu_kind_npu::npu_dynamic_quant'):
    need(g.get(key,0)==count,'CPU name candidate coverage')
  tx=tmap[(fw,0)];need(tx['source_csv_sha256']==x['source_csv_sha256'] and tx['trace_sha256']==x['trace_sha256'] and tx['groups']==x['groups'] and tx['nesting']==px['nesting'],'queue timing source/group mismatch')
  for label,g in tx['queue_timing'].items():
   count=x['groups'][label]['events']
   for z in g.values():need(z['count']==count and 0<=z['negative_count']<=count,'queue sample coverage')
  if fw=='sglang':need(tx['queue_timing']['above_100us']['dequeue_start_minus_enqueue_end_us']['p50_us']==434.25,'SG slow marker-gap p50')
  if fw=='vllm':need(tx['queue_timing']['at_or_below_100us']['dequeue_start_minus_enqueue_end_us']['negative_count']==1,'vLLM negative marker-gap')
  out[fw]={'targets':n,'distinct_connection_ids':n,'node_target_one':n,'dequeue_containment_one':n,'matching_enqueue_correlation_one':n,'dynamic_cpu_containment_multiple':n,'typed_flow_endpoints_one_each':n,'shortest_cpu_label_aclnnDynamicQuantV2':n,'outer_cpu_label_npu_dynamic_quant':n,'cpu_op_same_named_connection_zero':n}
 return {'rank0':out,'scope':'connection_id equality and item_id equality only; zero means no matching named field, not absence of CPU operator or other flow namespace'}


if __name__=='__main__':
 try:print(json.dumps(verify(),ensure_ascii=False,indent=2))
 except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as e:raise SystemExit('FAILED: '+str(e))
