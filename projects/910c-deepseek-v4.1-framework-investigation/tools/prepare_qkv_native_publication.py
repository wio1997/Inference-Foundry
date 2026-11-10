#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publish bounded QKV findings on fresh maps and disclose full per-delivery projection."""
from pathlib import Path
from hashlib import sha256
import json
from update_qkv_native_maps import SUMMARY,NEXT,TITLE

ROOT=Path('audit/execution-search');PREFIX='projects/910c-deepseek-v4.1-framework-investigation/'
def main():
 base=json.loads(Path('/tmp/qkv-native-public-base.json').read_text());files=[]
 def add(dest,content,source):files.append(dict(path=PREFIX+dest,content=content,local_source=source))
 def raw(n):p=ROOT/n;add('handoff/evidence/'+n,p.read_text(),str(p))
 for n in ['QKV原生反量化与贯穿Attention的head布局判断.md','qkv-native-source-review.json','qkv-native-frontier.json','qkv-native-catalog.json','qkv-native-dependencies.json','qkv-extra-dependencies.json','qkv-route-identity.json','qkv-dequant-order-cpu.json','qkv-native-exploration-notes.json','attention-head-layout-entry.json','attention-head-layout-entry-review.json']:raw(n)
 p=ROOT/'qkv-joint-delivery-price.json';d=json.loads(p.read_text());count=0
 for r in d['profiles']:
  values=r.pop('deliveries');count+=len(values);r['first_two_delivery_examples']=values[:2]
 projection=dict(type='summary_projection_not_full_raw',full_local_path=str(p),full_raw_sha256=sha256(p.read_bytes()).hexdigest(),full_raw_bytes=p.stat().st_size,omitted_record_count=count,kept='All324 price/cost scenarios, groups and negative counts retained; full1032 per-delivery records local, two examples per representative rank public.')
 d['public_representation']=projection;add('handoff/evidence/'+p.name,json.dumps(d,ensure_ascii=False,indent=2)+'\n',str(p)+' summary projection')
 rec={}
 for fn in ['qkv-native-source-review.json','attention-head-layout-entry-review.json']:
  for r in json.loads((ROOT/fn).read_text())['sources']:
   rp=r['runtime_path'];p=Path(r['path']);b=p.read_bytes();assert sha256(b).hexdigest()==r['sha256'];L=b.decode().splitlines()
   if rp not in rec:rec[rp]=dict(runtime_path=rp,full_raw_sha256=r['sha256'],full_raw_local_path=str(p),full_raw_bytes=len(b),reviewed_ranges=[],excerpts=[])
   else:assert rec[rp]['full_raw_sha256']==r['sha256']
   for a,z in r['reviewed_ranges']:
    assert 1<=a<=z<=len(L)
    if [a,z] not in rec[rp]['reviewed_ranges']:rec[rp]['reviewed_ranges'].append([a,z]);rec[rp]['excerpts'].append(dict(first=a,last=z,text='\n'.join(L[a-1:z])+'\n'))
   if r.get('reviewed_json_fields'):
    data=json.loads(b);rec[rp]['selected_json_fields']={k:data[k] for k in ['binFileName','blockDim','coreType','sha256','kernelList','supportInfo']};rec[rp]['json_scope']='Exact selected parsed fields, not full raw body or active invocation key.'
 ex=dict(strategy='Exact listed source line excerpts/selected compiler JSON fields, full raw hashes/local paths; empty ranges identity only. All full raw bodies retained local. Counts do not prove all branches or loadedbinary precision.',sources=list(rec.values()),whole_chain_complete=False,every_replacement_bitwise=True)
 p=ROOT/'qkv-public-source-excerpts.json';p.write_text(json.dumps(ex,ensure_ascii=False,indent=2)+'\n');raw(p.name)
 for n in ['read_qkv_native_catalog.py','read_qkv_native_dependencies.py','read_qkv_extra_dependencies.py','read_qkv_route_identity.py','analyze_qkv_joint_delivery_price.py','probe_qkv_dequant_order_cpu.py','analyze_qkv_native_frontier.py','read_attention_head_layout_entry.py','update_qkv_native_maps.py','prepare_qkv_native_publication.py']:
  p=Path('tools')/n;add('tools/'+n,p.read_text(),str(p))
 front=json.loads((ROOT/'source-frontier.json').read_text());shared={k:front[k] for k in ['qkv_native_review','attention_head_layout_entry']}
 body=SUMMARY+'\n\n[当前原生/付成本判断](evidence/QKV原生反量化与贯穿Attention的head布局判断.md) · [有限公式反例](evidence/qkv-dequant-order-cpu.json) · [SMLA head8入口](evidence/attention-head-layout-entry-review.json)。\n\n下一条待扫：'+NEXT+'\n\n待确认：head布局贯穿Attention/O投影的逐位、global query/selected-ID代次、资源/graph和请求净价；已确认可实现：本轮0；已停止：两个当前窗固定cache到达的Qb-only和普遍scale重结合，不剪更大parent。compiled source/guard/CPU反例均不当NPU或模型验收。\n'
 for name,r in base['files'].items():
  s=r['content']
  if name=='handoff/README.md':
   marker='## 4. 优化地图与研究阶段';assert marker in s;s=s.replace(marker,marker+'\n\n### '+TITLE+'\n\n'+body,1)
   i=s.index('**2026-11-10 当前资格**');j=s.index('\n\n',i)
   replacement='**2026-11-10 当前资格**：每个替换原语逐位一致。沿实际Q/KV发现[head布局贯穿Attention](evidence/QKV原生反量化与贯穿Attention的head布局判断.md)的较大交付，付旧AG参考/new Qb quarter/h5us条件1.228%–1.299%窗口，不是请求预测。INT32无溢出仍须原反量化次序；[CPU公式反例](evidence/qkv-dequant-order-cpu.json)最终BF16不同。两处head8 guard通过只属源码入口。继续global query/selected-ID/SMLA原MMAD与TP O投影；既有C09/K03更正与Engram lookup停止等限定继续有效。'
   s=s[:i]+replacement+s[j:]
  elif name.endswith('.md'):
   i=s.find('\n')+1;section=body.replace('(evidence/','(../handoff/evidence/') if name.startswith('modules/') else body;s=s[:i]+'\n## '+TITLE+'\n\n'+section+'\n'+s[i:]
  else:
   d=json.loads(s);d.update(shared);d.update(status='ACTIVE_LOOP039_HEAD_LAYOUT_THROUGH_ATTENTION',next_action=NEXT,current_analysis_base_commit='dd987034c9729a97d62e07e2596bb21e98585785',whole_chain_complete=False)
   if name=='progress.json':d.update(overall_complete=False,active_loop='loop-039',active_module='M00/M03/M11 larger QKV head layout through SMLA to TP O-projection',previous_goal_turn_classification='PROGRESS: actualQKV30pins/142headers payload0,516parents/rank and324 paidcases. Finite scale-order counterexample, larger headlayout1.23-1.30%window conditional; no nativebit/request admission.',latest_loop038_completed_review='run050354Z:bounded currentroute/native/cost PASS,correctnessNA; no admitted replacement',latest_loop039_completed_review='run054111Z:eight source identities+two headguard checks PASS,correctnessNA; run054436Z remainsPLANNED')
   else:d['completeness_established']=False
   s=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
  add(name,s,'Fresh public base '+base['base']+' reconciled, parallel profile/capture sections preserved')
 manifest=dict(base_commit=base['base'],paths=[r['path'] for r in files],price_projection=projection,source_pins=len(rec),read_ranges_nonempty=sum(bool(r['reviewed_ranges']) for r in rec.values()),whole_chain_complete=False,every_replacement_bitwise=True,new_primitive_admissions=0,qualification='Finite CPU formula counterexample not native bit proof; head8 guards not full implementation; window sensitivities not request gains; weight read difference not released VRAM.')
 p=ROOT/'qkv-publication-manifest.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');raw(p.name)
 out=dict(base_commit=base['base'],files=files);p=Path('/tmp/qkv-native-publication-payload.json');p.write_text(json.dumps(out,ensure_ascii=False));print(json.dumps(dict(files=len(files),characters=len(p.read_text()),path=str(p))))
if __name__=='__main__':main()
