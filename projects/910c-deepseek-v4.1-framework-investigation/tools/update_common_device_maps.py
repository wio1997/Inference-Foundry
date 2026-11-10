#!/usr/bin/env python3
# coding: utf-8
"""Publish bounded dead-projection evidence; keep whole-forward search open."""
from pathlib import Path
import json,re
ROOT=Path('audit/execution-search')
TITLE='2026-11-10 共同设备投影命中后死工作：保留原首owner，转全模型分支覆盖'
SUMMARY='介绍：设备权威长度经CP投影→原base缓存→C2/native与最终消费者。到了哪里：17当前源/11限定body/18行段、39个common读取点，原base符号哨兵960例与miss反例；16旧trace共4576次目标调用/4400命中段。结果：已有batch命中后的25段新长度值无消费者，原首投影/持久复制保留，独立补丁固定保留各QSL但未证明每份必要；精确补丁可审查但未安装。缩入时钟余量后平均45.234547ms/11周期Free，转移.5/新增.1ms每周期条件净2.098217%独立诊断窗。无原生/框架新准入，全链路未扫全，每替换原语逐位一致。'
NEXT='loop047 Run run-20261110T114154Z：沿当前模型/DSpark/Engram/attention的18个静态入口递归解析callee与合法guard，绑定冻结flag后再与同SHA旧审阅范围相交，分别记录源码已读/分支人口/消费者/原生证据；挑最大未扫交付。入口Run114347Z已PASS：16当前源、295语法callsite/156分支位，均不当唯一算子或执行覆盖。只读源码/配置/旧数据，无Torch/NPU/模型/新profile/服务动作。'
BODY=SUMMARY+'\n\n待确认，值得开发：已有local batch hit的死投影删除；源码边界闭合，仍缺私有clone真实Torch/NPU/原生packet和地址/空尾与capture逐位、无探针成本、全请求与六普通人口/转移。已确认可实现：上轮受限CPU原型1保留，本轮新增框架替换0。已停止：删首投影、CPU上界替设备值、无owner判别删全部QSL、writer-only异步化、跨代cache与空query跳QLI的具体配方；按owner有条件删非首QSL/pin/H2D及CPUscalar/setup的大parent保留，此投影价不当整段上界。\n\n请求另列：同镜像配平C8轻量参考15.967999s，真实metadata frame229/231；人口.5/转移.5/h.1ms的条件整批省1.336721%，人口1则2.817479%，默认未知比例0，非实测，不迁移原不同prompt C8/TTFT/普通。CPU微基准、不同窗口及并发时间线收益不相加。\n\n[完整判断](evidence/共同设备长度投影的死工作与真实消费者判断.md) · [原源码消费者](evidence/common-device-source-review.json) · [精确待审补丁](evidence/common-device-hit-elision.diff) · [局部与请求条件价](evidence/common-device-price.json) · [下一段静态入口](evidence/whole-forward-entry.json)。\n\n下一条待扫：'+NEXT+'\n'
def insert(path,body):
    p=Path(path);s=p.read_text();header='## '+TITLE
    if header in s:
        a=s.index(header);z=s.find('\n## ',a+len(header));z=len(s) if z<0 else z+1
        p.write_text(s[:a]+header+'\n\n'+body.rstrip()+'\n\n'+s[z:]);return
    if p.name in ['HANDOFF.md','ENVIRONMENT.md']:
        i=s.index('## 2026-11-10 共同frame分层key');s=s[:i]+header+'\n\n'+body.rstrip()+'\n\n'+s[i:]
    else:i=s.find('\n')+1;s=s[:i]+'\n'+header+'\n\n'+body.rstrip()+'\n\n'+s[i:]
    p.write_text(s)
def main():
    insert('HANDOFF.md',BODY.replace('(evidence/','(audit/execution-search/')+'\nloop046 Run105846Z PASS/正确性NA，loop046PIVOTED仅限定源码与旧trace；symbolic哨兵不当原生位。loop047入口Run114347Z PASS/NA，完整分支Run114154Z仍PLANNED。远端只读，服务未动。公开同步回执另记。')
    for n in ['M00','M03','M10','M11']:insert('audit/modules/'+n+'.md',BODY.replace('(evidence/','(../execution-search/'))
    insert(ROOT/'搜索范围与当前断点.md',BODY.replace('(evidence/','('));insert('audit/code-led/当前断点.md',SUMMARY+'\n\n'+NEXT);insert('audit/团队开发清单.md',BODY.replace('(evidence/','(execution-search/'))
    insert('ENVIRONMENT.md','适用冻结DeepSeek-V4.1-Flash-1017 W8A8/CANN9.2/910C，2026-11-10核实。171统一hostname正确，同owned容器/image，17device reader源及16全模型静态入口源全SHA匹配，running只作磁盘身份、不代替health。仅stdlib只读旧16份trace与CPU日志、源码；哨兵控制流在本机，不进口Torch/模型。新NPU/请求/profile/服务动作/远端写入0；待审CP diff没有安装到服务。既有恢复回执保留，不冒称本轮重测。')
    review=json.loads((ROOT/'common-device-source-review.json').read_text());front=dict(status=review['status'],classification='待确认，值得开发',source_boundary='Existing local batch hit only; original first main-stream projection/copy and every builderQSL/native output owner retained',sources=review['sources'],evidence=['common-device-source-review.json','common-device-reader-poison.json','common-device-trace-all.json','common-device-price.json','common-device-hit-elision.diff'],pending=review['native_gaps'],stopped=review['negative_recipes'],proposal_applied=False,new_primitive_admissions=0,new_framework_replacement_admissions=0,whole_chain_complete=False)
    (ROOT/'common-device-frontier.json').write_text(json.dumps(front,ensure_ascii=False,indent=2)+'\n')
    entry=json.loads((ROOT/'whole-forward-entry.json').read_text());compactentry=dict(status=entry['status'],sources=[{k:r[k] for k in ['runtime_path','runtime_sha256','matches_local']} for r in entry['sources']],root_count=entry['root_count'],callsite_count=entry['callsite_count'],branch_site_count=entry['branch_site_count'],qualification=entry['qualification'],evidence='whole-forward-entry.json',selected_dispatch_closed=False,whole_chain_complete=False)
    p=ROOT/'source-frontier.json';d=json.loads(p.read_text());d.update(status='ACTIVE_LOOP047_WHOLE_SELECTED_MODEL_CALLSITE_AND_LEGAL_BRANCH_DELTA',region='CP batch-hit deadprojection source/value bounded; wholemodel current-code branch coverage delta',next_action=NEXT,completeness_established=False,common_device_delivery_review=front,whole_forward_entry=compactentry,new_framework_replacement_admissions=0);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    p=Path('audit/first-pass.json');d=json.loads(p.read_text());d['latest_execution_derived_followup']=dict(status='ACTIVE_NOT_COMPLETE',summary=SUMMARY,next_action=NEXT,evidence='audit/execution-search/common-device-frontier.json',wholeForwardEntry='audit/execution-search/whole-forward-entry.json',new_framework_replacement_admissions=0,whole_chain_complete=False);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    for name in ['171-推理数据流与已查节点.html','171-融合算子全盘排查进度图.html']:
        p=Path(name);s=p.read_text();banner='<aside id="common-device-followup" style="padding:18px 24px;border:1px solid #c9d9de;margin:18px;border-radius:8px;background:#f5f9fa;color:#203a45;font:14px/1.7 system-ui"><strong>全链路未扫全 · 死投影25/26值得开发 · 框架新准入0</strong><p>'+SUMMARY+'</p><p>'+NEXT+'</p><p><a href="audit/execution-search/共同设备长度投影的死工作与真实消费者判断.md">消费者和付成本判断</a> · <a href="HANDOFF.md">断点</a></p></aside>';s,n=re.subn(r'<aside id="(?:frame-key-cpu-followup|common-device-followup)".*?</aside>',lambda m:banner,s,count=1,flags=re.S);assert n==1;p.write_text(s)
    p=Path('tasks/execution-derived-fusion-search/index.json');d=json.loads(p.read_text());assert d['active_loop']=='loop-047';d['next_action']=NEXT;p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    p=Path('tasks/execution-derived-fusion-search/loops/loop-047/state.json');d=json.loads(p.read_text());d['next_action']=NEXT;p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');p=p.with_name('status.md');s=p.read_text();s,n=re.subn(r'^- 下一步：.*$',lambda _: '- 下一步：'+NEXT,s,count=1,flags=re.M);assert n==1;p.write_text(s)
if __name__=='__main__':main()
