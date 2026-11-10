#!/usr/bin/env python3
"""Advance branch/dataflow axes and keep the cheap-scope stop cohort-specific."""
from pathlib import Path
import json,re
ROOT=Path('audit/execution-search')
TITLE='2026-11-11 worker状态与合法gate：119源码约束，37动态位未闭合'
REPORT='worker状态协议与早期CPU坐标交付判断.md'
NEXT='loop047 Run170051Z PLANNED：先核5个model/attention guard的初始化与实际读者（MoE skip-padding、Engram rotation、no-metadata/空query），再沿未闭合递归callee走实际_prepare_inputs/_preprocess/lookback/Engram及M03/M06/M08大交付。32个worker动态位已分列源码leg与运行人口，不重扫29个冻结gate。补actual OOT selector patch/use_compress注入与graph backend，影响gid12整段候选合法性。先独立源范围再匹配旧C8/六普通；保留大frame/gid12、全部CPU/producer/consumer租约与native逐位缺口。此Run无新Torch/NPU/模型/profile/请求/服务或远端变更。'
SUMMARY='介绍：沿真实execute→状态更新/CPU copy lease→异步修正→metadata provider→forward/错误→12字段暂态→sample消费协议，排除固定配置不合法gate。到了哪里：24当前源身份；61原worker guard源码leg走读，13原方法opaque协议控制+13字面量/factory控制；新增29受限冻结约束，原156位119绑定/37动态待追。结果：源码leg不等于分支人口或完整callee覆盖，全链路仍未闭合，新增原语/框架准入0。'
DETAIL='已停止：仅同配平C8详细诊断下，保持CPU计数生产/最终记账与GPU metadata原语不变的“删CPU位置影子/结构索引共享/早期等待后移”独立小方案。先定义完整父交付，再放宽到整个execute输入准备self，其prepare-Free仅1.687078ms/11循环（0.164512%诊断窗）；全命中/全转移/零新增的条件整批代理仅35.272ms/0.220889%。50%实删暴露、每frame新增0.02ms，局部净0.623539ms/0.060803%诊断窗；全命中/50%诊断转移的请求代理13.036ms/0.081637%。非实测收益，不迁移原不同prompt C8/TTFT/prefill/六普通；不作为删除CPU计数D2H整个协议、CP共同frame或gid12大段的上界，不与旧候选相加。\n\n待确认：37位仍有动态输入/人口或后续参数缺口；其中32worker位已有条件leg记录，另外5个model/attention guard继续。递归input/preprocess/lookback/MM/DBO/cascade及数值callee尚须追到全部消费者。A3 V1 model-call错误会release metadata；preprocess/Engram准备错误在该try之前，不能把V2 join当V1全错误保证。空scheduler/有效批提前返回与全局活跃但本CP query0不同，后者仍有global cache准备与O交换/补齐；CPU H2D guard也不等于producer_done。gid12整段和大共同frame继续保留图读span、task reset/异常、完整packet/逐位与无探针净价缺口。\n\n已确认可实现：本轮0，旧封闭CPU租约原型1保留。逐位门槛不放宽。控制架两次初版遗漏字段/误数13字段已按原源码修为12并留记录，不是后端错误或native数值失败。'
LINKS='[本段判断](evidence/'+REPORT+') · [119/37逐位与61源码leg](evidence/worker-protocol-source-review.json) · [13原协议控制](evidence/worker-execution-protocol-controls.json) · [先验整段范围](evidence/worker-position-parent-primary.json) · [整段放宽上界/条件价](evidence/worker-position-parent-price.json)。'
BODY=SUMMARY+'\n\n'+DETAIL+'\n\n'+LINKS+'\n\n下一条待扫：'+NEXT+'\n'
PATTERN=r'<aside id="(?:target-draft-default-entry|worker-protocol-default-entry)"[^>]*>.*?</aside>'
def banner(link,default=True):
 ident='worker-protocol-default-entry' if default else 'worker-protocol-followup'
 return '<aside id="'+ident+'" style="padding:16px 22px;margin:14px;border:1px solid #8daeb7;border-radius:8px;background:#f3f9fa;color:#193f4c;font:14px/1.7 system-ui"><strong>当前：119源码约束 / 37动态位未闭合；全链路未扫全</strong><p>24当前源；61 worker guard源码leg与分支人口分列，13 opaque协议+13默认控制。早期CPU影子/等待后移整段放宽预算仅0.165%配平诊断窗，停止该小边界独立开发；大共同frame/gid12继续。新原语/框架准入0，每替换原语逐位一致。</p><a href="'+link+'">真实状态协议、合法分支、整段价与下一断点</a></aside>'
def insert(path,body):
 p=Path(path);s=p.read_text();assert '## '+TITLE not in s;i=s.index('\n')+1;p.write_text(s[:i]+'\n## '+TITLE+'\n\n'+body+'\n'+s[i:])
def frontier():
 r=json.loads((ROOT/'worker-protocol-source-review.json').read_text())
 return dict(status=r['status'],classification='待确认，继续全链路',original_static_sites=156,scoped_source_bindings=119,pending_original_sites=37,current_source_pins=24,new_source_gate_bindings=29,worker_source_leg_records=61,source_leg_records_not_runtime_population=True,opaque_original_worker_cases=13,source_default_factory_controls=13,whole_callgraph_closed=False,whole_chain_complete=False,every_replacement_bitwise=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,isolated_CPU_prototypes_passed=1,stopped_scope=r['stopped'],remaining=r['remaining'],next_action=NEXT,evidence=[REPORT,'worker-protocol-source-review.json','worker-execution-protocol-controls.json','worker-position-parent-primary.json','worker-position-parent-price.json'])
def main():
 f=frontier();(ROOT/'worker-protocol-frontier.json').write_text(json.dumps(f,ensure_ascii=False,indent=2)+'\n')
 (ROOT/REPORT).write_text('# worker状态协议与早期CPU坐标交付判断\n\n'+BODY.replace('(evidence/','(')+'\nRun163605Z仅有界源码/旧预算PASS、正确性NA；Run170051Z PLANNED。C8 detailed17.632761s与light15.967999s分别保留，后者229/231真实metadata frame只用于显式条件代理；默认未知命中/转移0。24身份不当24全文，13协议控制不当真实事件或native数值。原诊断2未匹配FT区间在其它已标记metadata parent内，未补造。新Torch/NPU/模型/profile/请求/服务/远端写入0，未重测live健康。\n')
 # Record each shared working-file identity before additive map updates.
 from hashlib import sha256
 shared=['HANDOFF.md','171-推理数据流与已查节点.html','171-框架Profile热点.html','171-融合算子全盘排查进度图.html'];(ROOT/'worker-protocol-shared-before.json').write_text(json.dumps({n:sha256(Path(n).read_bytes()).hexdigest() for n in shared},ensure_ascii=False,indent=2)+'\n')
 insert('HANDOFF.md',BODY.replace('(evidence/','(audit/execution-search/')+'\nRun163605Z有界PASS/NA，Run170051Z PLANNED。公开同步回执待补；loop047活动，无live任务等待。\n')
 for n in ['M00','M01','M02','M03','M10','M11']:insert('audit/modules/'+n+'.md',BODY.replace('(evidence/','(../execution-search/'))
 insert(ROOT/'搜索范围与当前断点.md',BODY.replace('(evidence/','('));insert('audit/code-led/当前断点.md',SUMMARY+'\n\n'+NEXT+'\n');insert('audit/团队开发清单.md',BODY.replace('(evidence/','(execution-search/'))
 for p in [ROOT/'source-frontier.json',Path('audit/first-pass.json')]:
  d=json.loads(p.read_text());d.update(worker_protocol_review=f,status=f['status'],next_action=NEXT,whole_chain_complete=False,completeness_established=False);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 for name,default in [('171-推理数据流与已查节点.html',False),('171-融合算子全盘排查进度图.html',False),('171-框架Profile热点.html',True)]:
  p=Path(name);old=p.read_text();pattern=PATTERN if default else r'<aside id="(?:target-draft-followup|worker-protocol-followup)"[^>]*>.*?</aside>';s,n=re.subn(pattern,lambda _:banner('audit/execution-search/'+REPORT,default),old,count=1,flags=re.S);assert n==1 and re.sub(pattern,'',old,flags=re.S)==re.sub(pattern,'',s,flags=re.S);p.write_text(s)
 for p in [Path('tasks/execution-derived-fusion-search/index.json'),Path('tasks/execution-derived-fusion-search/loops/loop-047/state.json')]:
  d=json.loads(p.read_text());d['next_action']=NEXT;p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 p=Path('tasks/execution-derived-fusion-search/loops/loop-047/status.md');s=p.read_text();s=re.sub(r'^- 下一步：.*$',lambda _:'- 下一步：'+NEXT,s,flags=re.M);p.write_text(s)
 p=Path('tasks/execution-derived-fusion-search/loops/loop-047/report.md');p.write_text(p.read_text()+'\n\n## Run163605Z · worker协议/合法gate/整段剪枝（有界PASS/NA）\n\n'+SUMMARY+'\n\n'+DETAIL+'\n\n下一Run170051Z PLANNED，5个model guard与递归callee继续。\n')
 print(json.dumps(dict(source_pins=24,bound=119,pending=37,whole_chain_complete=False)))
if __name__=='__main__':main()
