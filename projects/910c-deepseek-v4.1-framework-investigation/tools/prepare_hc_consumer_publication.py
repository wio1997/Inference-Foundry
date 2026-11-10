#!/usr/bin/env python3
"""Publish HC consumer evidence and next Engram joint entry on fresh maps."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
import json
from update_hc_consumer_maps import TITLE,SUMMARY,NEXT,BODY
ROOT=Path('audit/execution-search');PREFIX='projects/910c-deepseek-v4.1-framework-investigation/'
def main():
 base=json.loads(Path('/tmp/hc-consumers-public-base.json').read_text());files=[]
 def add(dest,content,source):files.append(dict(path=PREFIX+dest,content=content,local_source=source))
 def raw(name):p=ROOT/name;add('handoff/evidence/'+name,p.read_text(),str(p))
 for n in ['HC双支路系数跨层读者与FULL图判断.md','hc-coefficient-consumer-dependencies.json','hc-coefficient-consumer-source-review.json','hc-coefficient-consumer-source-excerpts.json','hc-coefficient-consumer-frontier.json','hc-fork-join-cpu.json','hc-pmu-a1-invalid.json','engram-joint-entry.json','engram-joint-entry-review.json','engram-joint-entry-source-excerpts.json','engram-joint-entry-price.json']:raw(n)
 p=ROOT/'hc-both-delivery-price.json';d=json.loads(p.read_text());counts=[]
 for profile in d['profiles']:counts.append(dict(ep=profile['ep'],omitted_delivery_pairs=len(profile.pop('deliveries'))))
 d['publication_projection']=dict(qualification='Only per-delivery pairrecords omitted; full144cost+16PMUpressurecases including57missingEP8calls, allgroups/assumptions retained. Full local artifact below, not represented as uploadedraw.',full_local_path=str(p.resolve()),full_local_bytes=p.stat().st_size,full_local_sha256=sha256(p.read_bytes()).hexdigest(),omitted_counts=counts)
 add('handoff/evidence/'+p.name,json.dumps(d,ensure_ascii=False,indent=2)+'\n',str(p)+' explicitprojection')
 for n in ['read_hc_coefficient_consumers.py','analyze_hc_both_delivery_price.py','probe_hc_fork_join_cpu.py','review_hc_coefficient_consumers.py','read_engram_joint_entry.py','analyze_engram_joint_entry_price.py','review_engram_joint_entry.py','update_hc_consumer_maps.py','prepare_hc_consumer_publication.py']:
  p=Path('tools')/n;add('tools/'+n,p.read_text(),str(p))
 f=json.loads((ROOT/'source-frontier.json').read_text());shared={k:f[k] for k in ['hc_coefficient_consumer_review','engram_joint_entry']};stamp=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
 for name,s in base['files'].items():
  if name.endswith('.md'):
   section=BODY.replace('(evidence/','(../handoff/evidence/') if name.startswith('modules/') else BODY;i=s.find('\n')+1;s=s[:i]+'\n## 当前断点 · '+TITLE+'\n\n'+section+'\n'+s[i:]
   if name=='handoff/README.md':
    marker='## 4. 优化地图与研究阶段';assert marker in s;s=s.replace(marker,marker+'\n\n当前源码排查：loop042已追HC新系数至跨层读者/FULL图并扩大双parent价，转loop043 Engram全局rows/WKV/gate/HC大交付。'+SUMMARY+' [跨层与图判断](evidence/HC双支路系数跨层读者与FULL图判断.md)。既有详细性能核准及恢复历史保留。\n',1)
    i=s.index('**2026-11-10 当前资格**');j=s.index('\n\n',i);s=s[:i]+'**2026-11-10 当前资格**：每替换原语逐位一致，末位差异失败。target/DSpark两入口均seed非None premix，实际首层也可作source提前交付分析；旧firstattention None解释纠正，历史原件保留。ownPost前整coeff join只获源码/有限顺序证据，native位、FULL/allocator/真实争用待确认；双parent无争用条件价不能当请求收益，PMU代理未覆盖57次不填0。loop043当前沿全局Engram rows/WKV/rotatedgate/HC大parent，旧独立小预算停止保留。新准入0，全链路未扫全。'+s[j:]
  else:
   d=json.loads(s);d.update(shared,status='ACTIVE_LOOP043_ENGRAM_GLOBAL_ROWS_WKV_GATE_HC',next_action=NEXT,whole_chain_complete=False,current_analysis_base_commit='e338117685ae4ae498b69877936c452dd5f7e97e',updated_at_utc=stamp)
   if name=='progress.json':d.update(overall_complete=False,active_loop='loop-043',active_module='M01 globalEngramrows WKV gate HC, M02/M05/M07/M08/M10/M11 HC nativegates pending',previous_goal_turn_classification='PROGRESS: twoactualmodelpremixseeds, coefficienttail and laterreaders, sixfinitegraphs/threebadprotocols,144bothparent/16PMUpressure and9sourceEngram/54paidwholeentry. No nativebit/requestadmission.',latest_loop042_completed_review='run080543Z PASS boundedsource/order/price;correctnessNA;loop042PIVOTED',latest_loop043_completed_review='run083335Z PASS boundedsource/header/wholeentryprice;correctnessNA;run084215Z PLANNED originalMpanel/Nowner and wholejointvalue')
   else:d['completeness_established']=False
   s=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
  add(name,s,'Fresh public base '+base['base']+' reconciled; all earlierprofile/reconciliation/historicalrecords preserved')
 manifest=dict(base_commit=base['base'],paths=[r['path'] for r in files],whole_chain_complete=False,every_replacement_bitwise=True,new_primitive_admissions=0,qualification='All144bothparent/16PMUpressure/54Engramentry cases; explicitpairprojection withfullSHA/size.6normal/3badfiniteDAG, notruntimeproof. Correctactualfirstlayerpremix. Run084215Z remainsPLANNED; no request/nativebit admission.')
 p=ROOT/'hc-consumer-publication-manifest.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');raw(p.name)
 payload=dict(base_commit=base['base'],files=files);p=Path('/tmp/hc-consumer-publication-payload.json');p.write_text(json.dumps(payload,ensure_ascii=False));print(json.dumps(dict(files=len(files),characters=len(p.read_text()),path=str(p))))
if __name__=='__main__':main()
