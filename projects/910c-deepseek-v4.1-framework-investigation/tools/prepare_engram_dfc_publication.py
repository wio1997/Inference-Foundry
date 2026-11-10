#!/usr/bin/env python3
"""Reconcile explicit Engram/DFC evidence onto fresh public maps."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
import json
from update_engram_dfc_maps import TITLE,SUMMARY,NEXT,BODY
ROOT=Path('audit/execution-search');PREFIX='projects/910c-deepseek-v4.1-framework-investigation/'
def main():
    base=json.loads(Path('/tmp/engram-dfc-public-base.json').read_text());files=[]
    def add(dest,content,source):files.append(dict(path=PREFIX+dest,content=content,local_source=source))
    def raw(name):p=ROOT/name;add('handoff/evidence/'+name,p.read_text(),str(p))
    for name in ['Engram全局行Nowner几何与付成本请求判断.md','engram-row-owner-dependencies.json','engram-row-owner-cpu.json','engram-row-owner-source-review.json','engram-row-owner-source-excerpts.json','engram-row-owner-frontier.json','engram-row-owner-guard-a1.json','engram-row-owner-container-state.json','engram-row-price-a1-invalid.json','DFC输入行量化共享与首GMM交付判断.md','dfc-input-delivery-entry.json','dfc-input-host-dependencies.json','dfc-input-host-cpu.json','dfc-input-delivery-source-review.json','dfc-input-delivery-source-excerpts.json','dfc-input-delivery-frontier.json','dfc-input-host-cpu/driver.cpp']:raw(name)
    for name in ['engram-row-owner-price.json','dfc-input-delivery-price.json']:
        p=ROOT/name;d=json.loads(p.read_text());omissions=[]
        if name.startswith('engram'):
            omissions.append(dict(field='source_consistent_C8_prefill_records',records=len(d.pop('source_consistent_C8_prefill_records'))))
            for r in d['balanced_profiles']:omissions.append(dict(field='balanced_profiles.reference_records',ep=r['ep'],records=len(r.pop('reference_records'))))
            qualification='Only128C8sourcebindings and48balancedreferencebindings omitted; all108C8/216balancedcases, all544rho samples, sixordinary and failure/proxy qualifications preserved.'
        else:
            for r in d['profiles']:omissions.append(dict(field='profiles.deliveries',ep=r['ep'],records=len(r.pop('deliveries'))))
            qualification='Only1032individualDFCbindingrecords omitted; all60sensitivitycases/groupstats, exactlocalM3geometry and originalC8thresholds retained. Full rawlocalSHA/size below, not represented as uploadedraw.'
        d['publication_projection']=dict(qualification=qualification,omissions=omissions,full_local_path=str(p.resolve()),full_local_bytes=p.stat().st_size,full_local_sha256=sha256(p.read_bytes()).hexdigest())
        add('handoff/evidence/'+name,json.dumps(d,ensure_ascii=False,indent=2)+'\n',str(p)+' explicitprojection')
    for name in ['read_engram_row_owner_dependencies.py','probe_engram_row_owner_cpu.py','analyze_engram_row_owner_price.py','review_engram_row_owner.py','read_dfc_input_delivery_entry.py','read_dfc_quant_host_dependencies.py','probe_dfc_input_host_cpu.py','analyze_dfc_input_delivery_price.py','review_dfc_input_delivery.py','update_engram_dfc_maps.py','prepare_engram_dfc_publication.py']:
        p=Path('tools')/name;add('tools/'+name,p.read_text(),str(p))
    local=json.loads((ROOT/'source-frontier.json').read_text());shared={k:local[k] for k in ['engram_row_owner_review','dfc_input_delivery_review']};stamp=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    for name,s in base['files'].items():
        if name.endswith('.md'):
            section=BODY.replace('(evidence/','(../handoff/evidence/') if name.startswith('modules/') else BODY;i=s.find('\n')+1;s=s[:i]+'\n## 当前断点 · '+TITLE+'\n\n'+section+'\n'+s[i:]
            if name=='handoff/README.md':
                marker='## 4. 优化地图与研究阶段';assert marker in s;s=s.replace(marker,marker+'\n\n当前源码扫描已转loop044 DFC输入行量化/路由/跨EP交付至首GMM。'+SUMMARY+' [DFC实际读者与缺口](evidence/DFC输入行量化共享与首GMM交付判断.md)。既有完整性能页、采集/恢复记录和历史停止判断保留。\n',1)
                i=s.index('**2026-11-10 当前资格**');j=s.index('\n\n',i);s=s[:i]+'**2026-11-10 当前资格**：每个替换原语逐位一致，末位差异失败。本轮新准入0；全链路未扫全。Engram仅明确独立子情景按旧排序参考停止，完整owner/lookup/gate/HC保留具体缺口。DFC原host CPU回放发现localM3跨核重复18次/3行，prefill多数已共享；这不是activekey/NPU位或性能验收。完整DFC大段仍有价值，继续稳定路由map、mask/dummy/zero-work flags、peer/gather/GMM operand及bank末读。Run092257Z PASS仅源码/host/旧价格，Run094003Z仍PLANNED。'+s[j:]
        else:
            d=json.loads(s);d.update(shared,status='ACTIVE_LOOP044_DFC_INPUT_ROW_OWNERS_AND_FIRST_GMM',next_action=NEXT,whole_chain_complete=False,current_analysis_base_commit='72dda64a883d869c658ab252189cfca895f0a11b',updated_at_utc=stamp)
            if name=='progress.json':d.update(overall_complete=False,active_loop='loop-044',active_module='M06 DFC inputrowquant/route/peer/gmA firstGMM; M01 largerEngramowner nativegaps retained',previous_goal_turn_classification='PROGRESS: originalEngramdispatch/56geometry/byteoracles/paidC8+ordinary and actualDFC9source/12hostshapes/1032matchedparents. No nativebit/performanceadmission.',latest_loop043_completed_review='run084215Z PASS boundedsource/geometry/price;correctnessNA;loop043PIVOTED',latest_loop044_completed_review='run092257Z PASS boundedsource/hostCPU/oldprice;correctnessNA;run094003Z PLANNED routedrowmap/mask/count/gather/GMMlastreader')
            else:d['completeness_established']=False
            s=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
        add(name,s,'Fresh public base '+base['base']+' reconciled; otherprofile/timing/historicalrecords preserved')
    dest='handoff/evidence/engram-dfc-publication-manifest.json'
    manifest=dict(base_commit=base['base'],paths=[r['path'] for r in files]+[PREFIX+dest],whole_chain_complete=False,every_replacement_bitwise=True,new_primitive_admissions=0,qualification='ExactEngram/DFCboundedsource/CPU/paidmodels only. Explicit128+48+1032bindingprojections; all108+216Engram and60DFCsensitivitycases preserved. Nativebits/activekey/phase/requestgains unqualified; Run094003Z PLANNED. SourcefullSHA retained, newlyfetchedstockparallelbody localonly.')
    p=ROOT/'engram-dfc-publication-manifest.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');raw(p.name)
    assert len(files)==len({r['path'] for r in files})
    payload=dict(base_commit=base['base'],files=files);p=Path('/tmp/engram-dfc-publication-payload.json');p.write_text(json.dumps(payload,ensure_ascii=False));print(json.dumps(dict(files=len(files),characters=len(p.read_text()),path=str(p))))
if __name__=='__main__':main()
