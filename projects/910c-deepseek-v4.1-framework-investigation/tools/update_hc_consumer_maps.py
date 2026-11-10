#!/usr/bin/env python3
"""Record HC consumers and move to the larger Engram row/weight delivery."""
from pathlib import Path
import json,re
ROOT=Path('audit/execution-search')
TITLE='2026-11-10 HC双parent跨层与FULL图：转Engram全局行和WKV大交付'
SUMMARY='介绍：HC系数尾部到跨层/Engram/aux/final读者及FULL事件，再转全局Engram rows→WKV→gate→HC。到了哪里：15当前源/11限定body，两入口AST seed，6有限正常DAG/2004顺序对及3反例；144双parent价、16 PMU压力价；另9当前Engram源/6入口body、2权重header，payload0，每rank24条WKV+37gate完整链/54价格。结果：纠正实际首层None解释，target和DSpark首层均有premix；ownPost前整作业join支配后续读者。双parent无争用条件2.672%/2.740%窗口，PMU代理可降至.1%–.4%或负，不能当实测新争用。Engram大段全免1.169%/1.176%窗口，半省h5仅.572%/.575%；旧独立小预算不重开。新准入0，全链路未扫全，每替换原语逐位一致。'
NEXT='loop043 Run run-20261110T084215Z：现存全局complete-K Engram bank→保留原M3 padding/K/Fixpipe的WKV Nowner几何→新row-restore AllToAll→rotated gate和持久HC残差；比较direct-SP-owner的冲突，先用完整原C8/普通/prefill价大parent。只读源码/旧数据，不做NPU/框架进口/服务动作。'
BODY=SUMMARY+'\n\n待确认：HC native位、独立流/FULL捕获/allocator-workspace/真实争用；Engram全局owner/WKV/gate/HC联合原生几何、原语位与付成本净价。已确认可实现：本轮0。已停止：实际首层当None、借无关ExternalEvent作coeffdone、缺producer fence或跳过/异常不join的具体配方；旧WKV-only/gate-only/inactive-passthrough与原样overlap/staged-prefix停止保留。\n\n[HC双支路读者与FULL图](evidence/HC双支路系数跨层读者与FULL图判断.md) · [双parent/PMU价](evidence/hc-both-delivery-price.json) · [9源Engram入口](evidence/engram-joint-entry-review.json) · [完整WKV/gate价](evidence/engram-joint-entry-price.json)。\n\n下一条待扫：'+NEXT+'\n'
def insert(path,body):
 p=Path(path);s=p.read_text();h='## '+TITLE
 if h in s:return
 i=s.find('\n')+1;p.write_text(s[:i]+'\n'+h+'\n\n'+body+'\n'+s[i:])
def main():
 insert('HANDOFF.md',BODY.replace('(evidence/','(audit/execution-search/')+'\nloop042PIVOTED仅限定source/finiteorder/price；loop043入口Run083335Z PASS/正确性NA，新Run084215Z仍PLANNED。PMU N/A转换失败原脚本/原因保留，已改None不填0；EP8目标57/960无PMU，压力价不外推。171统一hostname核对成功，同owned image/id；只读源码/旧数据/header，未health探测或服务动作。公开同步回执另记。')
 for n in ['M01','M02','M04','M05','M07','M08','M10','M11']:insert('audit/modules/'+n+'.md',BODY.replace('(evidence/','(../execution-search/'))
 insert(ROOT/'搜索范围与当前断点.md',BODY.replace('(evidence/','('))
 insert('audit/code-led/当前断点.md',SUMMARY+'\n\n'+NEXT)
 insert('audit/团队开发清单.md',BODY.replace('(evidence/','(execution-search/'))
 insert('ENVIRONMENT.md','适用冻结DeepSeek-V4.1-Flash-1017 W8A8/CANN9.2/910C，2026-11-10核实。171统一入口hostname正确；同owned容器3be44e7b5c29/image ce42d65b运行身份guard匹配，非health。当前15系数/graph源全匹配，另9Engram源8旧match/1新body，两个WKV BF16[25600,6144] header和rotation header/config只读，payload0。EP8旧PMU57目标HC调用N/A留未知，非零工作。本轮新NPU/模型框架进口/请求/profile/显式远端写入/服务动作0。')
 p=ROOT/'source-frontier.json';d=json.loads(p.read_text());d.update(status='ACTIVE_LOOP043_ENGRAM_GLOBAL_ROWS_WKV_GATE_HC',region='HCpremixbothparents source/order/resourceprice bounded; larger Engram globalrowowner WKV rotatedgate HC input',next_action=NEXT,completeness_established=False,hc_coefficient_consumer_review=json.loads((ROOT/'hc-coefficient-consumer-frontier.json').read_text()),engram_joint_entry=json.loads((ROOT/'engram-joint-entry-review.json').read_text()));p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 p=Path('audit/first-pass.json');d=json.loads(p.read_text());d['latest_execution_derived_followup']=dict(status='ACTIVE_NOT_COMPLETE',summary=SUMMARY,next_action=NEXT,HC='audit/execution-search/hc-coefficient-consumer-frontier.json',Engram='audit/execution-search/engram-joint-entry-review.json',whole_chain_complete=False,new_primitive_admissions=0);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 for name in ['171-推理数据流与已查节点.html','171-融合算子全盘排查进度图.html']:
  p=Path(name);s=p.read_text();banner='<aside id="hc-engram-consumer-followup" style="padding:18px 24px;border:1px solid #c9d9de;margin:18px;border-radius:8px;background:#f5f9fa;color:#203a45;font:14px/1.7 system-ui"><strong>当前覆盖：全链路未扫全 · 每替换原语逐位一致</strong><p>'+SUMMARY+'</p><p>'+NEXT+'</p><p><a href="audit/execution-search/HC双支路系数跨层读者与FULL图判断.md">HC跨层与条件价</a> · <a href="HANDOFF.md">当前断点</a></p></aside>'
  s,n=re.subn(r'<aside id="(?:o-hc-premix-followup|hc-engram-consumer-followup)".*?</aside>',lambda m:banner,s,count=1,flags=re.S);assert n==1;p.write_text(s)
 p=Path('tasks/execution-derived-fusion-search/index.json');d=json.loads(p.read_text());assert d['active_loop']=='loop-043';d['next_action']=NEXT;p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
