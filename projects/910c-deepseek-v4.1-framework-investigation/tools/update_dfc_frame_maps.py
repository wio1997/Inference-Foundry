#!/usr/bin/env python3
"""Reconcile current route/reader coverage and advance the measured larger frame."""
from pathlib import Path
import json,re
ROOT=Path('audit/execution-search')
TITLE='2026-11-10 DFC原路由与末读闭合：转有函数归因的共同metadata大交付'
SUMMARY='介绍：DFC原mask/排序/inverse/count发布→远端搬运→GMM与Unpermute末读，再转同代共同metadata/frame。到了哪里：11当前DFC源/10限定body，454完整EP16路由组/7264ranklayer调用、80原生inverse witness、21792纯字节映射；5当前metadata源/4入口body与新16rank函数自身账、144付首producer/新增成本情景。结果：原R与dummy/zero-group flags保留，删gmA写不等于释放activation arena；历史5路由免费输入1.543–6.922us低于旧小预算，原样输入-only维持停止。新metadata仅helper共享的条件净1.995%诊断窗，CP自身25%也共享为3.622%，全base保留；非请求预测。新准入0，全链路未扫全，每替换原语逐位一致。'
NEXT='loop045 Run run-20261110T100641Z：同代frame最小字段key→CP/global/localbase实际caller与闭包→CPUhelper返回值mutation→每资源固定deviceQSL/slot/table/C2/nativeattrs发布；复用原acquire-before-mainwrite及CPU/producer/consumer双退役，价实际保留工作和新增成本。只读源码/已有诊断，不做NPU/框架进口/新profile/服务动作。'
BODY=SUMMARY+'\n\n待确认：改变GMM读法/提前publication/largeM/shared资源的原生位、phase/资源/末读与净价；metadata共同字段的实际caller兼容性、不可变CPU快照、device权威长度及每资源发布、图事件和原语位。已确认可实现：本轮0。已停止：历史固定GMM/flags的M3输入-only，sorted position/topK代原行、删dummy inverse/zero-group flags的具体配方；跨代shape-only或跨物理组整metadata复用仍停。保留不同GMM/大parent，未把旧排序参考改成用户门槛。\n\n[DFC路由与末读](evidence/DFC原路由行映射与GMM末读判断.md) · [共同metadata函数自身与净价](evidence/共同metadata函数自身归因与保留首生产者的净价.md) · [454路由/5原生旧实验](evidence/dfc-row-map-delivery.json) · [144条件价](evidence/common-frame-measured-price.json)。\n\n下一条待扫：'+NEXT+'\n'
def insert(path,body):
    p=Path(path);s=p.read_text();h='## '+TITLE
    if h in s:return
    if p.name in ['HANDOFF.md','ENVIRONMENT.md']:
        marker='## 2026-11-10 Engram全局owner付成本剪枝';i=s.index(marker)
        p.write_text(s[:i]+h+'\n\n'+body+'\n'+s[i:]);return
    i=s.find('\n')+1;p.write_text(s[:i]+'\n'+h+'\n\n'+body+'\n'+s[i:])
def main():
    insert('HANDOFF.md',BODY.replace('(evidence/','(audit/execution-search/')+'\nloop044PIVOTED限定源码/opaque CPU/旧实验净价；loop045入口Run100200Z PASS/正确性NA，新Run100641Z仍PLANNED。只读五路由原始15轮paired差与negative/outlier，旧inverse失败记录未纳入；新profile不是本扫描发起，原receipt归其它已授权流程。1025.503ms新诊断窗与940/930ms旧代表窗、原C8分列，90.347ms标记边界已扣，不重复扣。公开同步回执另记。')
    for n in ['M00','M06','M07','M11']:insert('audit/modules/'+n+'.md',BODY.replace('(evidence/','(../execution-search/'))
    insert(ROOT/'搜索范围与当前断点.md',BODY.replace('(evidence/','('));insert('audit/code-led/当前断点.md',SUMMARY+'\n\n'+NEXT);insert('audit/团队开发清单.md',BODY.replace('(evidence/','(execution-search/'))
    insert('ENVIRONMENT.md','适用冻结DeepSeek-V4.1-Flash-1017 W8A8/CANN9.2/910C，2026-11-10核实。171统一hostname核对成功，同owned container3be44e7b5c29/image ce42d65b；11DFC与5metadata当前源经docker cp stdout tar内存只读全匹配，观察running而非health。本机旧数据/opaque字节分析，无NPU/框架进口/请求/新profile/远端写入/服务动作。5metadata模块SHA与其它已完成诊断adapter一致，其恢复仅按原receipt；保存其90.347ms标记修正/两未匹配边界和独立1025.503ms窗，未把诊断当生产加速。')
    p=ROOT/'source-frontier.json';d=json.loads(p.read_text());d.update(status='ACTIVE_LOOP045_MEASURED_COMMON_FRAME_CALLERS_AND_READERS',region='DFCroute/inverse/count/GMM/Unpermute bounded; larger measured samegeneration commonmetadata/frame',next_action=NEXT,completeness_established=False,dfc_row_map_review=json.loads((ROOT/'dfc-row-map-frontier.json').read_text()),common_frame_measured_entry=json.loads((ROOT/'common-frame-measured-entry-review.json').read_text()))
    for r in d['frontiers']:
        if r['region']=='D-M1':
            r['state']='QUANT_ROWOWNERS_ROUTE_MAP_COUNT_PUBLICATION_FIRST_GMM_RETURN_READERS_BOUNDED'
            r['remaining']='ChangedGMM gather-to-L1/nativeK/Fixpipe/resource/bit gates, earlierpergroup publication and phase costs; other38targetlayers/draftlocalE8/ordinary nativepopulations; actual4/8co-residency/sharedschedule andprefixnet. Oldfixedconsumer input-only remainsstopped; originalR/dummy/flags/arena lastreaders nowbounded.'
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    p=Path('audit/first-pass.json');d=json.loads(p.read_text());d['latest_execution_derived_followup']=dict(status='ACTIVE_NOT_COMPLETE',summary=SUMMARY,next_action=NEXT,DFC='audit/execution-search/dfc-row-map-frontier.json',commonFrame='audit/execution-search/common-frame-measured-entry-review.json',whole_chain_complete=False,new_primitive_admissions=0);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    for name in ['171-推理数据流与已查节点.html','171-融合算子全盘排查进度图.html']:
        p=Path(name);s=p.read_text();banner='<aside id="dfc-common-frame-followup" style="padding:18px 24px;border:1px solid #c9d9de;margin:18px;border-radius:8px;background:#f5f9fa;color:#203a45;font:14px/1.7 system-ui"><strong>当前覆盖：全链路未扫全 · 每替换原语逐位一致</strong><p>'+SUMMARY+'</p><p>'+NEXT+'</p><p><a href="audit/execution-search/DFC原路由行映射与GMM末读判断.md">DFC实际读者</a> · <a href="audit/execution-search/共同metadata函数自身归因与保留首生产者的净价.md">共同frame净价</a> · <a href="HANDOFF.md">断点</a></p></aside>'
        s,n=re.subn(r'<aside id="(?:engram-dfc-row-owner-followup|dfc-common-frame-followup)".*?</aside>',lambda m:banner,s,count=1,flags=re.S);assert n==1;p.write_text(s)
    p=Path('tasks/execution-derived-fusion-search/index.json');d=json.loads(p.read_text());assert d['active_loop']=='loop-045';d['next_action']=NEXT;p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
