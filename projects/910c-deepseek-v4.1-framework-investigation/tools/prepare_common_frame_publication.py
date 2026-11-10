#!/usr/bin/env python3
"""Prepare bounded frame-lifetime and next O-projection findings on fresh maps."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
import json
from update_common_frame_maps import TITLE,SUMMARY,NEXT,BODY

ROOT=Path('audit/execution-search');PREFIX='projects/910c-deepseek-v4.1-framework-investigation/'
def main():
    base=json.loads(Path('/tmp/common-frame-public-base.json').read_text());files=[]
    def add(dest,content,source):files.append(dict(path=PREFIX+dest,content=content,local_source=source))
    def raw(name):p=ROOT/name;add('handoff/evidence/'+name,p.read_text(),str(p))
    for n in ['共同frame的真实覆写与双端退役判断.md','common-frame-lifetime-dependencies.json','common-frame-lifetime-frontier.json','common-frame-literal-accesses.json','common-frame-lifetime-source-excerpts.json','metadata-frame-order-cpu.json','global-query-packet-price.json','o-projection-native-entry.json','o-projection-native-entry-review.json']:raw(n)
    excerpts=[]
    for r in json.loads((ROOT/'o-projection-native-entry-review.json').read_text())['sources']:
        lines=Path(r['local_path']).read_text().splitlines();assert sha256(Path(r['local_path']).read_bytes()).hexdigest()==r['runtime_sha256']
        excerpts.append(dict(runtime_path=r['runtime_path'],full_source_sha256=r['runtime_sha256'],reviewed_ranges=r['reviewed_ranges'],excerpts=[dict(first=a,last=b,text='\n'.join(lines[a-1:b])) for a,b in r['reviewed_ranges']]))
    p=ROOT/'o-projection-entry-source-excerpts.json';p.write_text(json.dumps(dict(sources=excerpts,qualification='Only bounded actual caller/consumer entry ranges; empty identity-only; no nativebit/fullpath admission.'),ensure_ascii=False,indent=2)+'\n');raw(p.name)
    for name in ['read_common_frame_lifetime_dependencies.py','probe_metadata_frame_order_cpu.py','analyze_common_frame_lifetime.py','analyze_global_query_packet_price.py','read_o_projection_native_entry.py','update_common_frame_maps.py','prepare_common_frame_publication.py']:
        p=Path('tools')/name;add('tools/'+name,p.read_text(),str(p))
    f=json.loads((ROOT/'source-frontier.json').read_text());shared={k:f[k] for k in ['common_frame_lifetime_review','global_query_packet_price','o_projection_native_entry']}
    for name,s in base['files'].items():
        if name.endswith('.md'):
            section=BODY.replace('(evidence/','(../handoff/evidence/') if name.startswith('modules/') else BODY
            i=s.find('\n')+1;s=s[:i]+'\n## 当前断点 · '+TITLE+'\n\n'+section+'\n'+s[i:]
            if name=='handoff/README.md':
                marker='## 4. 优化地图与研究阶段';assert marker in s;s=s.replace(marker,marker+'\n\n当前源码排查：loop040完成限定frame寿命/packet价格判断并转loop041 O投影原生读者。'+SUMMARY+' [判断](evidence/共同frame的真实覆写与双端退役判断.md)。同窗详细性能账单及此前原件保留。\n',1)
                i=s.index('**2026-11-10 当前资格**');j=s.index('\n\n',i);s=s[:i]+'**2026-11-10 当前资格**：每个替换原语逐位一致，BF16末位差异失败。loop040已分清回调scalar与CPU/device存储，firstwrite、双端完成、pinned快照与partial-submit错误协议已具体化；仍待native/alias/位与净价。head8 packet按替换/共存分别收费。当前loop041继续OA/OB→SP→mHC/FFN大交付；本轮准入0，全链路未扫全。既有C09/K03更正与小范围停止限定继续有效。'+s[j:]
        else:
            d=json.loads(s);d.update(shared,status='ACTIVE_LOOP041_O_PROJECTION_NATIVE_TO_SP_HC_FFN',next_action=NEXT,whole_chain_complete=False,current_analysis_base_commit='e359f8846b5d604ea6d066c3e6e3341104aa38a0',updated_at_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
            if name=='progress.json':d.update(overall_complete=False,active_loop='loop-041',active_module='M04/M05 Oprojection SP roundedHC/FFN, M00/M11 frame pending',previous_goal_turn_classification='PROGRESS: actual15source frame callbacks/overwrites,20finite models/1033orders,144packet cost cases and7source Oprojection consumer entry. No runtimebit or request admission.',latest_loop040_completed_review='run063608Z PASS limitedsource/CPUorder/pricing,correctnessNA; loop040PIVOTED',latest_loop041_completed_review='run070221Z PASS actualcaller/consumer entry,correctnessNA; run070506Z PLANNED native and fullparent price')
            else:d['completeness_established']=False
            s=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
        add(name,s,'Fresh public base '+base['base']+' reconciled; earlier frame/head and concurrent profile sections preserved')
    manifest=dict(base_commit=base['base'],paths=[r['path'] for r in files],whole_chain_complete=False,every_replacement_bitwise=True,new_primitive_admissions=0,qualification='Full144cost scenarios and20order model cases retained; source ranges exact, literal inventory not complete def-use. No runtime race/bug, bit admission or requestprediction. Native OA entry only; newRun070506Z remainsplanned.')
    p=ROOT/'common-frame-publication-manifest.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');raw(p.name)
    out=dict(base_commit=base['base'],files=files);p=Path('/tmp/common-frame-publication-payload.json');p.write_text(json.dumps(out,ensure_ascii=False));print(json.dumps(dict(files=len(files),characters=len(p.read_text()),path=str(p))))
if __name__=='__main__':main()
