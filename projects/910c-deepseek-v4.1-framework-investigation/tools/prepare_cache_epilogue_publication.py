#!/usr/bin/env python3
"""Build explicit fresh-base public reconciliation; disclose large evidence projections."""
from pathlib import Path
from hashlib import sha256
import json
from update_cache_epilogue_maps import SUMMARY,NEXT,TITLE

ROOT=Path('audit/execution-search')
PREFIX='projects/910c-deepseek-v4.1-framework-investigation/'
def main():
    base=json.loads(Path('/tmp/cache-epilogue-public-base.json').read_text());files=[]
    def add(path,content,local):files.append(dict(path=PREFIX+path,content=content,local_source=local))
    def raw(name):
        p=ROOT/name;add('handoff/evidence/'+name,p.read_text(),str(p))
    for name in ['原生缓存发布与SWA整段融合判断.md','SWA主辅流join与较大输入交付判断.md','cache-epilogue-native-frontier.json','cache-epilogue-native-source-review.json','cache-epilogue-native-exploration-notes.json','cache-epilogue-native-dependencies.json','cache-epilogue-selected-sources.json','cache-epilogue-registry-callers.json','cache-epilogue-balanced-call-pins.json','swa-publication-parent-call-pins.json','swa-query-producer-entry.json','swa-query-producer-source-review.json','swa-query-producer-retrieval-attempts.json','swa-publication-exploration-notes.json']:raw(name)
    projections=[]
    for name,key in [('cache-epilogue-balanced-sites.json','chains_detail'),('swa-publication-join-price.json','binds')]:
        p=ROOT/name;d=json.loads(p.read_text());count=0
        if key=='chains_detail':
            for profile in d['profiles']:
                for f in profile['families']:
                    v=f.pop(key);count+=len(v);f['first_two_chain_examples']=v[:2]
        else:
            for profile in d['profiles']:
                v=profile.pop(key);count+=len(v);profile['first_two_binding_examples']=v[:2]
        d['public_representation']=dict(type='summary_projection_not_full_raw',full_local_path=str(p),full_raw_sha256=sha256(p.read_bytes()).hexdigest(),full_raw_bytes=p.stat().st_size,omitted_key=key,omitted_record_count=count,kept='All aggregate groups/prices/negative cases/input pins/limits plus first two examples per group; full per-chain original retained locally.')
        add('handoff/evidence/'+name,json.dumps(d,ensure_ascii=False,indent=2)+'\n',str(p)+' summary projection');projections.append(d['public_representation'])
    p=ROOT/'cache-epilogue-catalog.json';d=json.loads(p.read_text());d.pop('name_inventory');d['public_representation']=dict(type='catalog_projection',full_local_path=str(p),full_raw_sha256=sha256(p.read_bytes()).hexdigest(),omitted='First700filename inventory entries; full local inventory itself truncated at700, total982. Absence not inferred. All symbol source hits retained.');add('handoff/evidence/'+p.name,json.dumps(d,ensure_ascii=False,indent=2)+'\n',str(p)+' catalog projection');projections.append(d['public_representation'])
    review=json.loads((ROOT/'cache-epilogue-native-source-review.json').read_text());entry=json.loads((ROOT/'swa-query-producer-source-review.json').read_text());rec={}
    for row in review['sources']+entry['sources']:
        rp=row['runtime_path'];p=Path(row['path']);rawdata=p.read_bytes();assert sha256(rawdata).hexdigest()==row['sha256'];lines=rawdata.decode().splitlines()
        if rp not in rec:rec[rp]=dict(runtime_path=rp,full_raw_sha256=row['sha256'],full_raw_bytes=len(rawdata),full_raw_local_path=str(p),reviewed_ranges=[],excerpts=[])
        else:assert rec[rp]['full_raw_sha256']==row['sha256']
        for a,b in row['reviewed_ranges']:
            assert 1<=a<=b<=len(lines)
            if [a,b] not in rec[rp]['reviewed_ranges']:
                rec[rp]['reviewed_ranges'].append([a,b]);rec[rp]['excerpts'].append(dict(first=a,last=b,text='\n'.join(lines[a-1:b])+'\n'))
    excerpts=dict(strategy='Exact listed current-source excerpts, full raw SHA and local path for every source; no full raw binary/large trace published. Empty ranges identity only. Counts not completeness or loadedbinary/bit proof.',sources=list(rec.values()),whole_chain_complete=False,every_replacement_bitwise=True)
    p=ROOT/'cache-epilogue-public-source-excerpts.json';p.write_text(json.dumps(excerpts,ensure_ascii=False,indent=2)+'\n');raw(p.name)
    for name in ['read_cache_epilogue_catalog.py','read_cache_epilogue_native_dependencies.py','read_cache_epilogue_selected_sources.py','read_cache_epilogue_balanced_calls.py','read_cache_epilogue_registry_callers.py','analyze_cache_epilogue_balanced_sites.py','analyze_cache_epilogue_frontier.py','read_swa_publication_parent_calls.py','analyze_swa_publication_join.py','read_swa_query_producer_entry.py','update_cache_epilogue_maps.py','prepare_cache_epilogue_publication.py']:
        p=Path('tools')/name;add('tools/'+name,p.read_text(),str(p))
    front=json.loads((ROOT/'source-frontier.json').read_text());shared={k:front[k] for k in ['cache_epilogue_review','swa_publication_join_review','swa_query_producer_entry']}
    evidence='[原生交付与局部价格](evidence/原生缓存发布与SWA整段融合判断.md) · [Qb重叠与join净价](evidence/SWA主辅流join与较大输入交付判断.md)'
    body=SUMMARY+'\n\n'+evidence+'。\n\n下一条待扫：'+NEXT+'\n\n状态：待确认——更大Q/KV/input-ready与原生逐位/资源/请求净价；已确认可实现——本轮0；已停止——旧独立index尾段，固定Qb的SWA-only不以局部3%立项，不剪ordinary/任意资源重排或更大parent。全部大trace/逐链完整原件留本地，公开summary projection明确列SHA，不冒称全原件公开。\n'
    for name,r in base['files'].items():
        s=r['content']
        if name=='handoff/README.md':
            marker='## 4. 优化地图与研究阶段';assert marker in s;s=s.replace(marker,marker+'\n\n### '+TITLE+'\n\n'+body,1)
            old=s.index('**2026-11-10 当前资格**');end=s.index('\n\n',old)
            s=s[:old]+'**2026-11-10 当前资格**：每个替换原语逐位一致。原生缓存发布已扫限定ranges；[SWA主辅流join](evidence/SWA主辅流join与较大输入交付判断.md)固定Qb上界仅.649%–.661%窗口，不用局部3%直接立项。继续[较大Q/KV producer](evidence/swa-query-producer-source-review.json)。C09原序L1与K03 wholeDFC更正、Engram lookup-only停止及较大容量/input-owner缺口继续有效。'+s[end:]
        elif name.endswith('.md'):
            i=s.find('\n')+1;section=body
            if name.startswith('modules/'):
                section=section.replace('(evidence/','(../handoff/evidence/')
            s=s[:i]+'\n## '+TITLE+'\n\n'+section+'\n'+s[i:]
        else:
            d=json.loads(s);d.update(shared);d.update(status='ACTIVE_LOOP038_LARGER_QKV_PRODUCER_AFTER_SWA_JOIN',next_action=NEXT,current_analysis_base_commit='74b0640',whole_chain_complete=False)
            if name=='progress.json':
                d.update(overall_complete=False,active_loop='loop-038',active_module='M03/M00/M11 Q/KV joint native producer; bounded smaller native cache publisher',previous_goal_turn_classification='PROGRESS:39 current pins/29ranges; typed516 SWA chains/rank. Local3.07% collapses to fixed-Qb join .649-.661%window. Seven stopped-container disk source pins, not live loadedkernel; no new primitive admission.',latest_loop037_completed_review='run-20261110T042435Z bounded source/typed window PASS; numerical correctness NA',latest_loop038_completed_review='run-20261110T045533Z join correlation and050309Z CV Pythonentry PASS; native bit/request not certified')
            else:d['completeness_established']=False
            s=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
        add(name,s,'fresh public base '+base['base']+' reconciled; preserve prior fields/sections')
    manifest=dict(base_commit=base['base'],paths=[r['path'] for r in files],projections=projections,source_excerpt_count=len(rec),source_reviewed_nonempty=sum(bool(r['reviewed_ranges']) for r in rec.values()),omitted_archives=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in [ROOT/'cache-epilogue-balanced-calls.json.zlib',ROOT/'swa-publication-parent-calls.json.zlib']],whole_chain_complete=False,every_replacement_bitwise=True,new_primitive_admissions=0)
    p=ROOT/'cache-epilogue-publication-manifest.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');raw(p.name)
    payload=dict(base_commit=base['base'],files=files);p=Path('/tmp/cache-epilogue-publication-payload.json');p.write_text(json.dumps(payload,ensure_ascii=False));print(json.dumps(dict(files=len(files),characters=len(p.read_text()),chunks12000=(len(p.read_text())+11999)//12000,path=str(p))))
if __name__=='__main__':main()
