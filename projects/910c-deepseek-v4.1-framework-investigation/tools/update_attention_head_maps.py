#!/usr/bin/env python3
"""Reconcile head-layout branch/resource findings and next common-frame consumer."""
from pathlib import Path
import json,re

ROOT=Path('audit/execution-search')
TITLE='2026-11-10 head布局分支与共同metadata：继续真实寿命和消费者'
SUMMARY='介绍：贯穿Qb→全局query SMLA→inverseRoPE→TP O投影，并转向同代CP共同metadata交付。到了哪里：17当前head/native源、原标量25packet fixture与有限坐标；两个代表rank各480目标样本拆456CSA/24纯SWA；另9当前metadata源追到真实helper、base共享和SMLA/QLI闭包。结果：CSA的KV逻辑访问扩大8倍、Cube M padding2倍，纯SWA不同；原条件净价须付新增成本。共同plan仍有大范围价值，但31ms提交前边界非全可删或精确函数；首段源码/72条件价格已记。新原语准入0，全链路未扫全，每替换原语逐位一致。'
NEXT='loop040 Run run-20261110T063608Z：共同QSL/CPUlength/device length/positions的全部observer与mutation、闭包attrs，canonical frame在主流覆写前的producer_done+consumer_done栅栏；emptyquery/emptyDP/error/abort/FULL事件集合与顺序。保留每资源slot/ring与原native attrs，沿实际读者补证后继续下一交付。仅只读源码/旧数据，不重采、不操作服务。'
BODY=SUMMARY+'\n\n待确认：贯穿head布局的native位/metadata/NZowner/资源/请求净价；共同frame的alias/双完成栅栏/实际归因与净成本。已确认可实现：本轮0。已停止：cache-only packet借作globalquery、跨物理组整对象或shape-only跨代缓存、CPU上界替device长度；不剪完整动态协议或更大parent。\n\n[head分支与净价](evidence/head布局贯穿SMLA的分支与净价判断.md) · [共同metadata实际生产/价格](evidence/同代CP共同metadata的实际生产与价格判断.md)。\n\n下一条待扫：'+NEXT+'\n'
def insert(path,body):
    p=Path(path);s=p.read_text();heading='## '+TITLE
    if heading in s:return
    i=s.find('\n')+1;p.write_text(s[:i]+'\n'+heading+'\n\n'+body+'\n'+s[i:])
def main():
    local=BODY.replace('(evidence/','(audit/execution-search/')
    insert('HANDOFF.md',local+'\n本轮171统一hostname已核，同owned容器/image/id、最近只读见running，未探测health。仅只读源码/旧数据，本机标量C++与stdlib分析；NPU/模型/请求/新profile/显式远端写入/服务动作0。其它流程e77e8a8/公开2f0d0c3详细账单及恢复回执保留。loop039PIVOTED仅限定review；loop040新Run063608Z仍PLANNED，不计寿命或位验收。公开同步回执另记。')
    for m in ['M00','M01','M03','M11']:insert('audit/modules/'+m+'.md',BODY.replace('(evidence/','(../execution-search/'))
    insert(ROOT/'搜索范围与当前断点.md',BODY.replace('(evidence/','('))
    insert('audit/code-led/当前断点.md',SUMMARY+'\n\n'+NEXT)
    insert('audit/团队开发清单.md',BODY.replace('(evidence/','(execution-search/'))
    insert('ENVIRONMENT.md','适用冻结DeepSeek-V4.1-Flash-1017 W8A8 / CANN9.2 / 910C。171统一入口核hostname；同容器3be44e7b5c29、image ce42d65b，最近读取running只作磁盘读取身份，不作健康证明。17head/native与9metadata磁盘源SHA匹配。此扫描仅python -B -I -S只读源码和本机标量/旧数据，未进口框架，NPU/模型/请求/profile/远端写入/服务动作0。其它流程详细性能账单与服务恢复历史回执保持原样。')
    p=ROOT/'source-frontier.json';d=json.loads(p.read_text());d.update(status='ACTIVE_LOOP040_COMMON_METADATA_FRAME_LIFETIME',region='Head-layout source/CPU/resource bounded; same-generation CP metadata to actual consumers',next_action=NEXT,completeness_established=False,attention_head_layout_review=json.loads((ROOT/'attention-head-layout-frontier.json').read_text()),common_metadata_review=json.loads((ROOT/'common-metadata-frontier.json').read_text()));p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    p=Path('audit/first-pass.json');d=json.loads(p.read_text());d['latest_execution_derived_followup']=dict(status='ACTIVE_NOT_COMPLETE',summary=SUMMARY,next_action=NEXT,head_frontier='audit/execution-search/attention-head-layout-frontier.json',common_metadata='audit/execution-search/common-metadata-frontier.json',whole_chain_complete=False,new_primitive_admissions=0);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    for name in ['171-推理数据流与已查节点.html','171-融合算子全盘排查进度图.html']:
        p=Path(name);s=p.read_text();banner='<aside id="head-common-metadata-followup" style="padding:18px 24px;border:1px solid #c9d9de;margin:18px;border-radius:8px;background:#f5f9fa;color:#203a45;font:14px/1.7 system-ui"><strong>当前覆盖：全链路未扫全 · 每个替换原语逐位一致</strong><p>'+SUMMARY+'</p><p>'+NEXT+'</p><p><a href="audit/execution-search/head布局贯穿SMLA的分支与净价判断.md">head布局</a> · <a href="audit/execution-search/同代CP共同metadata的实际生产与价格判断.md">共同metadata</a> · <a href="HANDOFF.md">断点</a></p></aside>'
        s,n=re.subn(r'<aside id="(?:qkv-native-followup|head-common-metadata-followup)".*?</aside>',lambda m:banner,s,count=1,flags=re.S);assert n==1;p.write_text(s)
if __name__=='__main__':main()
