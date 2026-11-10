#!/usr/bin/env python3
"""Publish bounded O/HC evidence over a freshly fetched workbench base."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
import json
from update_o_hc_maps import TITLE,SUMMARY,NEXT,BODY
ROOT=Path('audit/execution-search');PREFIX='projects/910c-deepseek-v4.1-framework-investigation/'
def main():
 base=json.loads(Path('/tmp/o-hc-public-base.json').read_text());files=[]
 def add(dest,content,source):files.append(dict(path=PREFIX+dest,content=content,local_source=source))
 def raw(name):p=ROOT/name;add('handoff/evidence/'+name,p.read_text(),str(p))
 for n in ['O投影原生资源与SP残差FFN交付判断.md','HC原生舍入与premix提前交付判断.md','o-projection-native-catalog.json','o-projection-native-dependencies.json','o-projection-fixpipe-dependencies.json','o-projection-native-source-review.json','o-projection-native-source-excerpts.json','o-projection-native-frontier.json','o-projection-saved-clocks.json','o-projection-saved-clocks-a1-invalid.json','o-projection-resource-ownership.json','hc-delivery-catalog.json','hc-delivery-dependencies.json','hc-delivery-source-review.json','hc-delivery-source-excerpts.json','hc-delivery-frontier.json']:raw(n)
 for n in ['o-projection-parent-price.json','hc-early-delivery-price.json']:
  p=ROOT/n;d=json.loads(p.read_text());counts=[]
  for profile in d['profiles']:counts.append(dict(ep=profile['ep'],deliveries=len(profile.pop('deliveries'))))
  d['publication_projection']=dict(qualification='Explicit summary projection omits only per-delivery records; allgroups/costscenarios/counterexample/assumptions retained. Full immutable local original below, not represented as uploaded raw.',full_local_path=str(p.resolve()),full_local_bytes=p.stat().st_size,full_local_sha256=sha256(p.read_bytes()).hexdigest(),omitted_delivery_counts=counts)
  add('handoff/evidence/'+n,json.dumps(d,ensure_ascii=False,indent=2)+'\n',str(p)+' explicitprojection')
 for n in ['read_o_projection_native_catalog.py','read_o_projection_native_dependencies.py','read_o_projection_fixpipe_dependencies.py','read_o_projection_saved_clocks.py','analyze_o_projection_parent_price.py','analyze_o_projection_resource_ownership.py','review_o_projection_native.py','read_hc_delivery_catalog.py','read_hc_delivery_dependencies.py','analyze_hc_early_delivery.py','review_hc_delivery.py','update_o_hc_maps.py','prepare_o_hc_publication.py']:
  p=Path('tools')/n;add('tools/'+n,p.read_text(),str(p))
 f=json.loads((ROOT/'source-frontier.json').read_text());shared={k:f[k] for k in ['o_projection_native_review','hc_delivery_review']}
 for name,s in base['files'].items():
  if name.endswith('.md'):
   section=BODY.replace('(evidence/','(../handoff/evidence/') if name.startswith('modules/') else BODY
   i=s.find('\n')+1;s=s[:i]+'\n## 当前断点 · '+TITLE+'\n\n'+section+'\n'+s[i:]
   if name=='handoff/README.md':
    marker='## 4. 优化地图与研究阶段';assert marker in s;s=s.replace(marker,marker+'\n\n当前源码排查：loop041完成限定O投影资源/整段价，loop042沿HC真实premix提前交付到后续读者。'+SUMMARY+' [原生与交付判断](evidence/HC原生舍入与premix提前交付判断.md)。并发详细性能核准/可加总树及恢复历史保持原样。\n',1)
    i=s.index('**2026-11-10 当前资格**');j=s.index('\n\n',i);s=s[:i]+'**2026-11-10 当前资格**：每替换原语逐位一致，BF16末位差异失败。O投影单owner M24与擦掉OA舍入的预合成停止，多owner/大prefill保留具体位/资源缺口。HC premix-present早交付有源码独立性，仍待原系数尾部/跨层代次、独立事件/FULL图、真实争用及逐原语位；FP32router来自舍入BF16扩宽。loop042 Run075729Z为限定source/price PASS、正确性NA；Run080543Z仍PLANNED。准入0，全链路未扫全，既有C09/K03更正与小范围停止限定有效。'+s[j:]
  else:
   d=json.loads(s);d.update(shared,status='ACTIVE_LOOP042_HC_PREMIX_NATIVE_EARLY_DELIVERY',next_action=NEXT,whole_chain_complete=False,current_analysis_base_commit='e9c4a0a',updated_at_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
   if name=='progress.json':d.update(overall_complete=False,active_loop='loop-042',active_module='M04/M05 Oprojection SP HCpremix earlydelivery; M10/M11 laterreaders pending',previous_goal_turn_classification='PROGRESS: original Oprojection native resource/paidparent pruning, HC originalrounding and premixindependence to laterFFNPost, three paidsensitivity grids. No nativebit/request admission.',latest_loop041_completed_review='run070506Z PASS boundedsource/resource/price,correctnessNA; loop041PIVOTED',latest_loop042_completed_review='run075729Z PASS boundednative/source/price,correctnessNA; run080543Z PLANNED latercoeff/nextdecoder and FULLsplitjoin')
   else:d['completeness_established']=False
   s=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
  add(name,s,'Fresh public base '+base['base']+' reconciled; concurrent profile/reconciliation and earlierhistory preserved')
 manifest=dict(base_commit=base['base'],paths=[r['path'] for r in files],whole_chain_complete=False,every_replacement_bitwise=True,new_primitive_admissions=0,qualification='All144parent+108resource+144forkjoin scenarios retained; explicitpriceprojection excludes per-delivery originals preserved locally withSHA/size. Sourceexcerpts exact, identity-only emptyranges. Native key/primitivebit/requestgain not admitted; Run080543Z PLANNED.')
 p=ROOT/'o-hc-publication-manifest.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');raw(p.name)
 out=dict(base_commit=base['base'],files=files);p=Path('/tmp/o-hc-publication-payload.json');p.write_text(json.dumps(out,ensure_ascii=False));print(json.dumps(dict(files=len(files),characters=len(p.read_text()),path=str(p))))
if __name__=='__main__':main()
