#!/usr/bin/env python3
"""Publish bounded route readers and measured common-frame entry on fresh maps."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
import json
from update_dfc_frame_maps import TITLE,SUMMARY,NEXT,BODY
ROOT=Path('audit/execution-search');PREFIX='projects/910c-deepseek-v4.1-framework-investigation/'
def main():
    base=json.loads(Path('/tmp/dfc-frame-public-base.json').read_text());files=[]
    def add(dest,content,source):files.append(dict(path=PREFIX+dest,content=content,local_source=source))
    def raw(name):p=ROOT/name;add('handoff/evidence/'+name,p.read_text(),str(p))
    for n in ['DFC原路由行映射与GMM末读判断.md','dfc-row-map-dependencies.json','dfc-row-map-source-review.json','dfc-row-map-source-excerpts.json','dfc-row-map-frontier.json','共同metadata函数自身归因与保留首生产者的净价.md','common-frame-measured-entry.json','common-frame-measured-entry-review.json','common-frame-measured-source-excerpts.json']:raw(n)
    for name in ['dfc-row-map-delivery.json','common-frame-measured-price.json']:
        p=ROOT/name;d=json.loads(p.read_text());omissions=[]
        if name.startswith('dfc'):
            omissions.append(dict(field='groups',records=len(d.pop('groups'))));qual='Only454perinvocation receivergeometrygroups omitted; all80savednativeinverse witnesses and all225pairedsamples/fiveoldroutes, groupstats, scope andinputSHA retained. OpaqueCPU is not newnativebit.'
        else:
            for i,s in enumerate(d['scenarios']):omissions.append(dict(field='scenarios.rank_values',scenario=i,records=len(s.pop('rank_values'))))
            qual='Only144x16scenario-specificranknetrows omitted; all16rankoriginalselfcosts,144scenario parameters/mean/minmax/negativevalues andprobe/clipping limits retained. Not requestgain.'
        d['publication_projection']=dict(qualification=qual,omissions=omissions,full_local_path=str(p.resolve()),full_local_bytes=p.stat().st_size,full_local_sha256=sha256(p.read_bytes()).hexdigest())
        add('handoff/evidence/'+name,json.dumps(d,ensure_ascii=False,indent=2)+'\n',str(p)+' explicitprojection')
    for n in ['read_dfc_row_map_dependencies.py','analyze_dfc_row_map_delivery.py','review_dfc_row_map.py','read_common_frame_measured_entry.py','analyze_common_frame_measured_price.py','review_common_frame_measured_entry.py','update_dfc_frame_maps.py','prepare_dfc_frame_publication.py']:
        p=Path('tools')/n;add('tools/'+n,p.read_text(),str(p))
    local=json.loads((ROOT/'source-frontier.json').read_text());shared={k:local[k] for k in ['dfc_row_map_review','common_frame_measured_entry']};stamp=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    for name,s in base['files'].items():
        if name.endswith('.md'):
            section=BODY.replace('(evidence/','(../handoff/evidence/') if name.startswith('modules/') else BODY;i=s.find('\n')+1;s=s[:i]+'\n## 当前断点 · '+TITLE+'\n\n'+section+'\n'+s[i:]
            if name=='handoff/README.md':
                marker='## 4. 优化地图与研究阶段';assert marker in s;s=s.replace(marker,marker+'\n\n源码扫描：loop044已追DFC原R/dummy/count/zero-group flags至GMM与Unpermute末读，旧固定消费者输入-only停止保留；loop045转有新函数归因的共同frame。'+SUMMARY+' [函数自身与付成本价](evidence/共同metadata函数自身归因与保留首生产者的净价.md)。保留8c237b2完整timeline/覆盖与恢复记录，本文未重采。\n',1)
                i=s.index('**2026-11-10 当前资格**');j=s.index('\n\n',i);s=s[:i]+'**2026-11-10 当前资格**：每替换原语逐位一致，末位差异失败。本轮新准入0、全链路未扫全。DFC原R/flags/lastreader仅限定源码与opaque CPU，历史五路由output/packet证据只覆盖其合成positive slots，非新原语验收；固定GMM输入-only旧小预算维持停止，改变GMM/提前发布/大M仍有具体缺口。新metadata16rank函数自身归因只提供144条件情景，base整段保留、marker边界已扣、首producer均值proxy明示，不当请求或生产收益。loop045 Run100200Z PASS仅入口/价格，Run100641Z仍PLANNED。'+s[j:]
        else:
            d=json.loads(s);d.update(shared,status='ACTIVE_LOOP045_MEASURED_COMMON_FRAME_CALLERS_AND_READERS',next_action=NEXT,whole_chain_complete=False,current_analysis_base_commit='6a5b03f8cedb83f1343e9a4de10376a3ae83d485',updated_at_utc=stamp)
            if 'frontiers' in d:
                for r in d['frontiers']:
                    if r['region']=='D-M1':r.update(next(x for x in local['frontiers'] if x['region']=='D-M1'))
            if name=='progress.json':d.update(overall_complete=False,active_loop='loop-045',active_module='M00 measuredcommonCPframe/callers/keys and M11epoch/fences; M06changedGMM nativephase/resourcegaps retained',previous_goal_turn_classification='PROGRESS:11currentDFCsource/10boundedbodies,454routegroups/80nativeinverse/old5routepaireddifferences;5currentCPmodules/newdisjointself/144paidfirstproducer cases. No nativebit or performanceadmission.',latest_loop044_completed_review='run094003Z PASS boundedroute/readers/opaqueCPU/oldpairedcost;correctnessNA;loop044PIVOTED',latest_loop045_completed_review='run100200Z PASS entry/sourcefunctionbinding/paidmodels;correctnessNA;run100641Z PLANNED exactframekey/callers/returnmutations/stablepublication')
            else:d['completeness_established']=False
            s=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
        add(name,s,'Freshpublicbase '+base['base']+' reconciled; completeframeworktiming/coverage/restoration and otherrecords preserved')
    dest='handoff/evidence/dfc-frame-publication-manifest.json';manifest=dict(base_commit=base['base'],paths=[f['path'] for f in files]+[PREFIX+dest],whole_chain_complete=False,new_primitive_admissions=0,every_replacement_bitwise=True,qualification='Boundedsource/CPU/oldnativecounterfactual and alreadycollecteddiagnosticselfprice only. Explicit454group and144x16ranknet projections withfullSHA/size; all80inverse/225paireddeltas/144scenarios retained. No new NPU/profile/service; Run100641Z PLANNED.')
    p=ROOT/'dfc-frame-publication-manifest.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');raw(p.name)
    assert len(files)==len({r['path'] for r in files})
    payload=dict(base_commit=base['base'],files=files);p=Path('/tmp/dfc-frame-publication-payload.json');p.write_text(json.dumps(payload,ensure_ascii=False));print(json.dumps(dict(files=len(files),characters=len(p.read_text()),path=str(p))))
if __name__=='__main__':main()
