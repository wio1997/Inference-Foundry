#!/usr/bin/env python3
"""Prepare exact owned publication files on fresh public maps; preserve concurrent data."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
import json
from update_attention_head_maps import TITLE,SUMMARY,NEXT,BODY

ROOT=Path('audit/execution-search');PREFIX='projects/910c-deepseek-v4.1-framework-investigation/'
def main():
    base=json.loads(Path('/tmp/head-layout-public-base.json').read_text());files=[]
    def add(dest,content,source):files.append(dict(path=PREFIX+dest,content=content,local_source=source))
    raw=['head布局贯穿SMLA的分支与净价判断.md','同代CP共同metadata的实际生产与价格判断.md','attention-head-layout-dependencies.json','attention-head-layout-source-review.json','attention-head-layout-frontier.json','attention-head-layout-delivery.json','attention-head-layout-source-excerpts.json','common-metadata-dependencies.json','common-metadata-frontier.json','common-metadata-delivery-price.json','common-metadata-source-excerpts.json']
    for n in raw:
        p=ROOT/n;add('handoff/evidence/'+n,p.read_text(),str(p))
    projections=[]
    for n in ['attention-head-schedule-cpu-a1/result.json','attention-head-schedule-cpu-a2/result.json','attention-head-schedule-cpu-a3/result.json']:
        p=ROOT/n;d=json.loads(p.read_text())
        for case in d.get('cases',[]):
            words=case.pop('defined900');case['defined900_canonical_json_sha256']=sha256(json.dumps(words,separators=(',',':')).encode()).hexdigest()
        d['public_representation']=dict(kind='summary_projection_not_full_packet_words' if d.get('cases') else 'full_result',full_local_path=str(p),full_raw_sha256=sha256(p.read_bytes()).hexdigest(),full_raw_bytes=p.stat().st_size,kept='All case attrs, FA ranges, statuses and limits retained. All900 words per case and driver/stdout remain local; canonical compact JSON SHA per case retained.')
        projections.append(dict(path=str(p),**d['public_representation']));add('handoff/evidence/'+n,json.dumps(d,ensure_ascii=False,indent=2)+'\n',str(p)+' disclosed packet projection')
    for name in ['read_attention_head_layout_dependencies.py','probe_attention_head_schedule_cpu.py','analyze_attention_head_delivery.py','build_attention_head_review.py','read_common_metadata_dependencies.py','analyze_common_metadata_delivery.py','update_attention_head_maps.py','prepare_attention_head_publication.py']:
        p=Path('tools')/name;add('tools/'+name,p.read_text(),str(p))
    shared={k:json.loads((ROOT/'source-frontier.json').read_text())[k] for k in ['attention_head_layout_review','common_metadata_review']}
    for name,s in base['files'].items():
        if name.endswith('.md'):
            section=BODY.replace('(evidence/','(../handoff/evidence/') if name.startswith('modules/') else BODY
            i=s.find('\n')+1;s=s[:i]+'\n## 当前断点 · '+TITLE+'\n\n'+section+'\n'+s[i:]
            if name=='handoff/README.md':
                marker='## 4. 优化地图与研究阶段';assert marker in s
                s=s.replace(marker,marker+'\n\n当前源码排查：loop039限定source/CPU/价格已PIVOT；loop040继续共同metadata真实寿命。'+SUMMARY+' [判断](evidence/同代CP共同metadata的实际生产与价格判断.md)。另一流程的同窗84类详细账单及旧profile保持原样。\n',1)
                i=s.index('**2026-11-10 当前资格**');j=s.index('\n\n',i)
                new='**2026-11-10 当前资格**：每个替换原语逐位一致，BF16末位也不能不同。head贯穿SMLA已追到CSA/dense分支与条件净价，增加KV/M-padding成本后仍待native bit、globalquery plan和请求净价。当前loop040继续同代共同metadata的输入observer/alias/闭包attrs与producer_done+consumer_done；本轮准入0，全链路未扫全。旧C09/K03更正、Engram lookup停止等限定继续有效。'
                s=s[:i]+new+s[j:]
        else:
            d=json.loads(s);d.update(shared,status='ACTIVE_LOOP040_COMMON_METADATA_FRAME_LIFETIME',next_action=NEXT,whole_chain_complete=False,current_analysis_base_commit='e77e8a8',updated_at_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
            if name=='progress.json':d.update(overall_complete=False,active_loop='loop-040',active_module='M00/M03/M11 same-generation CP common metadata to real consumers',previous_goal_turn_classification='PROGRESS: bounded17head source/25CPU packets and CSA/dense branch cost;9metadata sources and72 conditionalprices. No nativebit/request admission.',latest_loop039_completed_review='run054436Z PASS limitedsource/CPU/resource,correctnessNA; a1ERROR/a2INVALID retained',latest_loop040_completed_review='run062847Z PASS source/conditionalprice,correctnessNA; run063608Z PLANNED observer/fence review')
            else:d['completeness_established']=False
            s=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
        add(name,s,'Fresh public '+base['base']+' reconciled; concurrent detailed performance ledger preserved')
    manifest=dict(base_commit=base['base'],paths=[r['path'] for r in files],packet_projections=projections,qualification='Source/CPU/address checks not native bit acceptance; all72metadata cost cases and head price margins retained. No complete request prediction or universal coverage. Concurrent performance-breakdown files untouched.',whole_chain_complete=False,every_replacement_bitwise=True,new_primitive_admissions=0)
    p=ROOT/'attention-head-publication-manifest.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');add('handoff/evidence/'+p.name,p.read_text(),str(p))
    out=dict(base_commit=base['base'],files=files);p=Path('/tmp/attention-head-publication-payload.json');p.write_text(json.dumps(out,ensure_ascii=False));print(json.dumps(dict(files=len(files),characters=len(p.read_text()),path=str(p))))
if __name__=='__main__':main()
