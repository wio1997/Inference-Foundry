#!/usr/bin/env python3
"""Bounded, CPU-only audit of existing local checkpoint metadata; no weights."""
import hashlib,json
from collections import Counter
from pathlib import Path
T=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent
root=Path('/Users/wio/work/169-vllm-ascend-reproduce-172')
p=root/'source172/model-metadata/quant_model_description.json';prov=root/'evidence/002-source-files.json'
b=p.read_bytes(); digest=hashlib.sha256(b).hexdigest();expected=json.loads(prov.read_text())['model-metadata/quant_model_description.json']['sha256']
if digest!=expected: raise RuntimeError('local metadata SHA mismatch')
j=json.loads(b);summary={}
for part in ('wq_a','wq_b','wkv'):
 keys=[f'layers.{i}.attn.{part}.weight' for i in range(40)]
 summary[part]={'quant_types':dict(Counter(j.get(k,'MISSING') for k in keys)),'keys_checked':len(keys),'layers_with_missing':[i for i,k in enumerate(keys) if k not in j]}
selected={k:j.get(k) for k in ('layers.0.attn.wq_a.weight','layers.0.attn.wq_b.weight','layers.0.attn.wq_b.weight_scale','layers.0.attn.wq_b.weight_offset','layers.0.attn.wkv.weight','layers.0.ffn.experts.0.w1.weight','layers.0.ffn.shared_experts.w1.weight')}
out={'scope':'Existing local 172-derived metadata snapshot; no 172 service access, not independently hash-matched to 169 runtime checkpoint','source_path':str(p),'sha256':digest,'source_evidence':str(prov),'target_layers':40,'results':summary,'selected':selected,'not_proven':['169 runtime checkpoint metadata hash','runtime quant_method object','all source-to-loader name remapping','whether each captured graph execution used this branch','weight_scale/weight_offset internal arithmetic'],'no_device_work':True}
receipt=O/'checkpoint-metadata-comparison.json'
if receipt.exists():
 c=json.loads(receipt.read_text())
 if c.get('status')!='MATCH' or c.get('local_metadata',{}).get('sha256')!=digest:
  raise RuntimeError('checkpoint comparison missing or differs from local metadata')
 out['scope']='Local reference metadata SHA and size match the 169 frozen-container checkpoint metadata at the time of read; historical loaded runtime object and per-step branches remain unproven.'
 out['checkpoint_hash_comparison']={'path':'agents/fusion/checkpoint-metadata-comparison.json','sha256':hashlib.sha256(receipt.read_bytes()).hexdigest(),'stage':'operator-ceiling169-v2-084-quant-hash','status':'MATCH'}
 out['not_proven']=['historical process loaded the exact same bytes','runtime quant_method object','all source-to-loader name remapping','whether each captured graph execution used this branch','weight_scale/weight_offset internal arithmetic']
(O/'quant-metadata-summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
