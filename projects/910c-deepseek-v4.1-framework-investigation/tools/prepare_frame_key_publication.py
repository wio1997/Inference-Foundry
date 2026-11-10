#!/usr/bin/env python3
"""Publish key/caller/CPU qualification, preserving fresh public performance work."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
import json
from update_frame_key_maps import TITLE,SUMMARY,NEXT,BODY
ROOT=Path('audit/execution-search');PREFIX='projects/910c-deepseek-v4.1-framework-investigation/'
def main():
    base=json.loads(Path('/tmp/frame-key-public-base.json').read_text());files=[]
    def add(dest,content,source):files.append(dict(path=PREFIX+dest,content=content,local_source=source))
    def raw(name):p=ROOT/name;add('handoff/evidence/'+name,p.read_text(),str(p))
    for n in ['共同frame分层键与真实caller及CPU原语判断.md','共同frame_CPU模式资格补充.md','common-frame-key-dependencies.json','common-frame-key-source-review.json','common-frame-key-source-excerpts.json','common-frame-key-frontier.json','common-frame-key-fetch-a1-invalid.json','common-frame-cpu-import-guard.json','common-frame-torch-cpu-launch.json','common-frame-torch-cpu.json','common-frame-cpu-modes-launch.json','common-frame-cpu-modes.json','common-frame-cpu-mode-guarded.py','common-frame-cpu-mode-frontier.json']:raw(n)
    for name in ['common-frame-key-cpu.json','common-frame-caller-logs.json','common-frame-caller-price.json']:
        p=ROOT/name;d=json.loads(p.read_text());omissions=[]
        if name=='common-frame-key-cpu.json':
            omissions.append(dict(field='cases',records=len(d.pop('cases'))));qualification='1728scalar-ASTcase records omitted; all8negativecases,tiers/source/scope retained. ActualTorchCPU1728parameter/outputSHA records remainfullypublished separately.'
        elif name=='common-frame-caller-logs.json':
            for r in d['ranks']:omissions.append(dict(field='ranks.selected',ep=r['ep'],records=len(r.pop('selected'))))
            qualification='Onlypercall selectedparent rows omitted; all16rankcaller counts/176targetbatchfirsthelper records/rawfileSHA andclock/FTmiss retained. Full local file below, not uploadedraw.'
        else:
            for i,s in enumerate(d['scenarios']):omissions.append(dict(field='scenarios.rank_values',scenario=i,records=len(s.pop('rank_values'))))
            qualification='Only144x16rank scenario netrows omitted; all16rank original/retainedbodycosts and144scenario parameters/mean/negativecases retained. Full local SHA below.'
        d['publication_projection']=dict(qualification=qualification,omissions=omissions,full_local_path=str(p.resolve()),full_local_bytes=p.stat().st_size,full_local_sha256=sha256(p.read_bytes()).hexdigest())
        add('handoff/evidence/'+name,json.dumps(d,ensure_ascii=False,indent=2)+'\n',str(p)+' explicitprojection')
    for n in ['read_common_frame_key_dependencies.py','probe_common_frame_key_cpu.py','read_common_frame_caller_logs.py','analyze_common_frame_caller_price.py','read_cpu_torch_import_guard.py','probe_common_frame_torch_cpu.py','launch_common_frame_torch_cpu.py','review_common_frame_key.py','probe_common_frame_cpu_modes.py','launch_common_frame_cpu_modes.py','update_frame_key_maps.py','prepare_frame_key_publication.py']:
        p=Path('tools')/n;add('tools/'+n,p.read_text(),str(p))
    local=json.loads((ROOT/'source-frontier.json').read_text());shared={k:local[k] for k in ['common_frame_key_review','common_frame_CPU_mode_overlay']};stamp=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    for name,s in base['files'].items():
        if name.endswith('.md'):
            section=BODY.replace('(evidence/','(../handoff/evidence/') if name.startswith('modules/') else BODY;i=s.find('\n')+1;s=s[:i]+'\n## 当前断点 · '+TITLE+'\n\n'+section+'\n'+s[i:]
            if name=='handoff/README.md':
                marker='## 4. 优化地图与研究阶段';assert marker in s;s=s.replace(marker,marker+'\n\n共同frame当前：loop045已做分层key/真实26+draft1 caller与CPU原语核验；loop046继续原主流deviceprojection读者。'+SUMMARY+' [CPU模式资格](evidence/共同frame_CPU模式资格补充.md)。既有完整timeline、诊断/恢复、DFC及其它结果保留。\n',1)
                i=s.index('**2026-11-10 当前资格**');j=s.index('\n\n',i);s=s[:i]+'**2026-11-10 当前资格**：每替换原语逐位一致。隔离CPU结果复用原型受限确认1，服务框架替换准入仍0，全链路未扫全。v1普通CPU1728例/44928比较通过，inference-created Tensor 的_version读取实际失败；guard变体3模式/78比较通过，仍须不可变输入租约，不能靠版本号发现NumPy写入。六组配对CPU成本属于v1普通模式，不充当guard服务收益。新caller价1.964%/3.275%只是独立诊断敏感性，非请求预测。目标26与draft1、CPU双mirror/原base缓存/每资源地址分开；Run110329Z CPU模式门槛PASS，Run105846Z device读者仍PLANNED。'+s[j:]
        else:
            d=json.loads(s);d.update(shared,status='ACTIVE_LOOP046_ORIGINAL_MAINSTREAM_COMMON_DEVICE_READERS',next_action=NEXT,whole_chain_complete=False,current_analysis_base_commit='004ce4ea35e4545de78f32429f3bbcf375103a84',updated_at_utc=stamp,isolated_CPU_primitive_prototypes_passed=1,new_framework_replacement_admissions=0)
            if name=='progress.json':d.update(overall_complete=False,active_loop='loop-046',active_module='M00 originalmainstream commondeviceprojection andM03/M11 persistentreader/capture; M10target26/draft1 separated',previous_goal_turn_classification='PROGRESS:12currentkey/caller sources,16archivedCPUparentlogs, actualTorchCPU1728cases/44928bytes, inference-modev1counterexample andguard78bytes,144callerboundfirstbodyprices. No framework/NPUadmission.',latest_loop045_completed_review='run100641Z source/model/log/pricePASSNA;run104319Z actualCPUprimitivePASS;loop045PIVOTED',latest_loop046_completed_review='run110329Z CPUmodehardgatePASS;sourceRun105846Z stillPLANNED mainstreamprojection toallreaders')
            else:d['completeness_established']=False
            s=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
        add(name,s,'Freshpublicbase '+base['base']+' reconciled; priorframeworktiming/coverage/restoration preserved')
    dest='handoff/evidence/frame-key-publication-manifest.json';manifest=dict(base_commit=base['base'],paths=[f['path'] for f in files]+[PREFIX+dest],whole_chain_complete=False,new_framework_replacement_admissions=0,new_primitive_admissions=0,isolated_CPU_prototypes_passed=1,every_replacement_bitwise=True,qualification='All1728actualTorchCPUcase parameter/outputSHA records and6x15pairedCPUcases/3modeguard cases published. Explicit1728scalarcase/percaller/144x16ranknet projections retainfullSHA/size. v1inference failure retained, guard testednotservingintegrated; newNPU/model/profile/serviceactions0, ownedCPUfiles only. SourceRun105846Z PLANNED.')
    p=ROOT/'frame-key-publication-manifest.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');raw(p.name)
    assert len(files)==len({r['path'] for r in files})
    p=Path('/tmp/frame-key-publication-payload.json');p.write_text(json.dumps(dict(base_commit=base['base'],files=files),ensure_ascii=False));print(json.dumps(dict(files=len(files),characters=len(p.read_text()),path=str(p))))
if __name__=='__main__':main()
