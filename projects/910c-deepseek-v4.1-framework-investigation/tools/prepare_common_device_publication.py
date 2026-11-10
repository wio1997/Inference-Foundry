#!/usr/bin/env python3
# coding: utf-8
"""Reconcile dead-projection and new forward-entry evidence onto fresh public state."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
import json
from update_common_device_maps import TITLE,SUMMARY,NEXT,BODY
ROOT=Path('audit/execution-search');PREFIX='projects/910c-deepseek-v4.1-framework-investigation/'
def main():
    base=json.loads(Path('/tmp/common-device-public-base.json').read_text());files=[]
    def add(path,text,source):files.append(dict(path=PREFIX+path,content=text,local_source=source))
    def raw(name):p=ROOT/name;add('handoff/evidence/'+name,p.read_text(),str(p))
    for n in ['共同设备长度投影的死工作与真实消费者判断.md','common-device-entry.json','common-device-source-review.json','common-device-source-excerpts.json','common-device-reader-poison.json','common-device-hit-elision.diff','common-device-price.json','common-device-frontier.json','whole-forward-entry.json','common-device-trace-0.json','common-device-trace-0-reader-a1.py']:raw(n)
    p=ROOT/'common-device-trace-all.json';full=json.loads(p.read_text());digest=sha256(p.read_bytes()).hexdigest();summary=[]
    for rank in full['ranks']:
        name=f'common-device-trace-ranks/ep{rank["ep"]:02}.json';out=ROOT/name;out.parent.mkdir(exist_ok=True);d=dict(rank=rank,publication_source=dict(full_local_path=str(p.resolve()),full_local_sha256=digest,full_local_bytes=p.stat().st_size,qualification='This rank is complete, all phase times/IDs/firstowners/patterns/samples/raw source pins retained. Reassemble ranks to reproduce original root.'))
        out.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');raw(name)
        summary.append(dict(ep=rank['ep'],trace=rank['trace'],target_frames=rank['target_frames'],phase_count=len(rank['phases']),full_rank_evidence=name))
    full['ranks']=summary;full['publication_projection']=dict(full_local_path=str(p.resolve()),full_local_sha256=digest,full_local_bytes=p.stat().st_size,qualification='Root rank rows split into16complete public rank files for per-file readback. No phase record omitted; all4576 phase records and4400 eligiblehit records retained across rank files. All other original root fields retained.')
    add('handoff/evidence/'+p.name,json.dumps(full,ensure_ascii=False,indent=2)+'\n',str(p)+' explicitprojection')
    for n in ['read_common_device_entry.py','read_common_device_trace.py','probe_common_device_readers.py','review_common_device_delivery.py','read_whole_forward_entry.py','update_common_device_maps.py','prepare_common_device_publication.py']:
        p=Path('tools')/n;add('tools/'+n,p.read_text(),str(p))
    local=json.loads((ROOT/'source-frontier.json').read_text());shared={k:local[k] for k in ['common_device_delivery_review','whole_forward_entry']};stamp=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    for name,s in base['files'].items():
        if name.endswith('.md'):
            section=BODY.replace('(evidence/','(../handoff/evidence/') if name.startswith('modules/') else BODY;i=s.find('\n')+1;s=s[:i]+'\n## 当前断点 · '+TITLE+'\n\n'+section+'\n'+s[i:]
            if name=='handoff/README.md':
                marker='## 4. 优化地图与研究阶段';assert marker in s;s=s.replace(marker,marker+'\n\n共同设备投影：保留首owner与各builderQSL，已有batch hit的25段死工作已源码闭合并生成未安装补丁；新框架准入0。继续当前代码全模型分支覆盖，不把静态入口计数当执行。'+SUMMARY+' [消费者、补丁与价](evidence/共同设备长度投影的死工作与真实消费者判断.md)。既有完整timeline/恢复/覆盖与先前结果全部保留。\n',1)
                a=s.index('**2026-11-10 当前资格**');z=s.index('\n\n',a);s=s[:a]+'**2026-11-10 当前资格**：每替换原语逐位一致。隔离CPU原型受限通过1，新增框架替换准入0，全链路未扫全。本轮17current source、39common读取、960原base符号哨兵及4576旧trace序列闭合已有batch hit的死投影边界；哨兵不是原生位验收。精确补丁未安装，原首投影/base复制/各builderQSL/native输出与stage-group保留。device-only转移.5/h.1ms条件净2.098217%独立诊断窗；全请求1.336721%/2.817479%仅人口与转移假设敏感性、默认0，不能迁移原不同promptC8/TTFT/普通。loop046PIVOTED；loop047静态入口16源/18root/295语法调用/156分支位，selected dispatch及合法分支仍UNKNOWN，Run114154Z仍PLANNED。'+s[z:]
        else:
            d=json.loads(s);d.update(shared,status='ACTIVE_LOOP047_WHOLE_SELECTED_MODEL_CALLSITE_AND_LEGAL_BRANCH_DELTA',next_action=NEXT,whole_chain_complete=False,current_analysis_base_commit='0cad2af63fe473ec87850843f4cefc081a79278d',updated_at_utc=stamp,new_framework_replacement_admissions=0)
            if name=='progress.json':d.update(overall_complete=False,active_loop='loop-047',active_module='M00/M03/M10/M11 commondevicehit bounded; wholemodel current-code dispatch and legalbranch delta',previous_goal_turn_classification='PROGRESS:17pins/39commonreads/960symbolicpoison withmissnegative;16savedtrace4576calls/4400hitphases paid price; exact uninstalledCPdiff; next16source staticentry18roots295calls156branchsites, allunknownnotcoverage.',latest_loop046_completed_review='run105846Z source/symbolicreader/oldtrace/pricePASSNA;loop046PIVOTED;actualNPU/plugin/capture/request gates pending',latest_loop047_completed_review='run114347Z identity/staticentryPASSNA;broaderRun114154Z PLANNED')
            else:d['completeness_established']=False
            s=json.dumps(d,ensure_ascii=False,indent=2)+'\n'
        add(name,s,'Freshpublicbase '+base['base']+' reconciled; concurrenttimeline/coverage/restoration preserved')
    dest='handoff/evidence/common-device-publication-manifest.json';manifest=dict(base_commit=base['base'],paths=[f['path'] for f in files]+[PREFIX+dest],whole_chain_complete=False,new_primitive_admissions=0,new_framework_replacement_admissions=0,isolated_CPU_prototypes_passed=1,every_replacement_bitwise=True,qualification='No proposedCPpatch applied, symbolic960 isnotnativebit. All4576savedtrace phase rows fullypublic across16rankfiles, rootprojectiondeclaredSHA/size; all32local and36requestproxy scenarios and960symbolicrecords full. Currentforward16source/18roots295call156branch is staticentryonly, notlegal/executioncoverage. NewTorch/NPU/model/profile/service/remote-writeactions0.')
    p=ROOT/'common-device-publication-manifest.json';p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');raw(p.name);assert len(files)==45 and len({f['path'] for f in files})==45
    p=Path('/tmp/common-device-publication-payload.json');p.write_text(json.dumps(dict(base_commit=base['base'],files=files),ensure_ascii=False));print(json.dumps(dict(files=len(files),characters=len(p.read_text()),path=str(p))))
if __name__=='__main__':main()
