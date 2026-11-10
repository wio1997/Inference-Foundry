#!/usr/bin/env python3
"""Record a material whole delivery without inflating whole-model coverage."""
from pathlib import Path
import json,re
ROOT=Path('audit/execution-search')
TITLE='2026-11-11 target阶段draft metadata整段：值得深入，保留图与状态缺口'
REPORT='target阶段draft_metadata整段交付判断.md'
NEXT='loop047 Run163605Z PLANNED：继续66原静态位及递归callee，从selected execute_model状态/DP空批与异步修正、provider submit/release及KV/EC/offload/Mamba/EPLB/DBO gate走到真实读者。gid12整段候选保留图backend/task注册、捕获读span、覆写前末读、CM/rawphysical、task reset/异常及native位/无探针净价具体缺口，不阻塞剩余路径。优先下一处未扫大交付，先独立源边界再匹配旧C8/六普通；每替换原语逐位一致。无新Torch/NPU/模型/profile/请求/服务或远端变更。'
SUMMARY='介绍：从真实13KV/26builder源计划独立追target阶段最后gid12的整次CP/global/local metadata交付。到了哪里：15当前源码身份、34原Python控制；16旧详细trace/176次last-of26，完整保留两次base与helper成本并去匹配标记边界。结果：墙钟中位1.663035ms，留时钟余量的prepare-Free15.223438ms/11循环，占诊断窗1.484485%，整段值得深入；原156位90受限绑定/66待追未变，不冒充新增全链路覆盖。'
DETAIL='待确认：target只按本层prefix取值且actual draft会fresh build，但仍须核selected图backend/task注册/捕获buffer读span、私有buffer写后读、MM/offload字典读者、两provider task reset及异常恢复、native完整bytes与无探针净价。CM选择/unpad、rawphysical setter和真正draft producer保留。当前legacy FULL update因use_compress绕开，不等于捕获地址无读者。\n\n局部条件净价：实删且暴露50%/新增0.1ms每frame，11循环净省6.511719ms，占该诊断窗0.634978%。同配平轻量15.967999s、229/231 metadata frame，只作条件请求代理：命中50%/诊断转移50%/新增0.1ms每frame，平均56.574ms/0.354297%；命中100%则136.148ms/0.852633%。未知命中/转移默认0；不是实测净增益，不迁移原不同prompt C8、TTFT/prefill或六普通，不与helper/后25长度投影/大frame相加。\n\n已停止：在CM/rawphysical setter之前跳整group；只跳build却保留无reset的take（控制发现会返回旧task；同时删build/take须另证无读/提交）；删除earlier packet首owner；以旧native小岛价剪这段host parent。已确认可实现：本轮原语/框架替换0，旧CPU封闭租约原型1保留。全链路未闭合，每替换原语逐位一致。'
LINKS='[本段判断](evidence/'+REPORT+') · [源/状态与34控制](evidence/target-draft-metadata-review.json) · [完整176调用/区间原件](evidence/target-draft-metadata-price-raw.json) · [局部/请求条件价](evidence/target-draft-metadata-price.json)。'
BODY=SUMMARY+'\n\n'+DETAIL+'\n\n'+LINKS+'\n\n下一条待扫：'+NEXT+'\n'
PATTERN=r'<aside id="(?:input-resource-default-entry|target-draft-default-entry)"[^>]*>.*?</aside>'
def banner(link,default=True):
 ident='target-draft-default-entry' if default else 'target-draft-followup'
 return '<aside id="'+ident+'" style="padding:16px 22px;margin:14px;border:1px solid #8daeb7;border-radius:8px;background:#f3f9fa;color:#193f4c;font:14px/1.7 system-ui"><strong>当前：target阶段draft metadata整段值得深入；全链路未闭合</strong><p>176旧调用整段墙钟中位1.663ms；去标记/时钟余量后15.223ms/11循环prepare-Free，约1.484%诊断窗。尚缺图读span、task reset/错误与原生逐位及无探针净价；保留CM/物理view和actual draft。原90绑定/66待追不变，新原语/框架准入0。</p><a href="'+link+'">整段判断、条件净价与下一断点</a></aside>'
def insert(path,body):
 p=Path(path);s=p.read_text();assert '## '+TITLE not in s;i=s.index('\n')+1;p.write_text(s[:i]+'\n## '+TITLE+'\n\n'+body+'\n'+s[i:])
def frontier():
 return dict(status='ACTIVE_LOOP047_TARGET_DRAFT_METADATA_WHOLE_DELIVERY_WORTH_DEEPENING',classification='待确认，继续全链路',scoped_source_bindings=90,pending_original_static_sites=66,current_delivery_source_pins=15,original_Python_controls=34,saved_rank_traces=16,complete_last_builder_calls=176,body_call_median_ms=1.663035,guarded_prepare_free_mean_11_frame_ms=15.2234375,diagnostic_window_pct=1.4844853153198108,source_facts_not_native_admission=True,whole_callgraph_closed=False,whole_chain_complete=False,every_replacement_bitwise=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,isolated_CPU_prototypes_passed=1,next_action=NEXT,evidence=[REPORT,'target-draft-metadata-review.json','target-draft-metadata-price.json','target-draft-metadata-price-raw.json','target-draft-metadata-source-excerpts.json'])
def main():
 f=frontier();(ROOT/'target-draft-metadata-frontier.json').write_text(json.dumps(f,ensure_ascii=False,indent=2)+'\n')
 report=BODY.replace('(evidence/','(')+'\nRun155707Z仅此有界子段PASS/正确性NA；原计划66位与全部分支未完成，Run163605Z继续。新Torch/NPU/模型导入/profile/请求/服务/远端写入0，未重测live健康。\n\n完整CP body以外的CM/unpad/submit等必要工作不能全删；放宽到parent末端的31.519375ms/3.073553%仅探索包络，不是可删除成本或整段请求上界。已记录2个FT/body未匹配，未填造。\n'
 (ROOT/REPORT).write_text('# target阶段draft metadata整段交付判断\n\n'+report)
 insert('HANDOFF.md',BODY.replace('(evidence/','(audit/execution-search/')+'\nRun155707Z有界PASS/NA，Run163605Z PLANNED。公开同步回执待补；loop047仍EVALUATING，无live任务等待。\n')
 for n in ['M00','M01','M02','M03','M10','M11']:insert('audit/modules/'+n+'.md',BODY.replace('(evidence/','(../execution-search/'))
 insert(ROOT/'搜索范围与当前断点.md',BODY.replace('(evidence/','('));insert('audit/code-led/当前断点.md',SUMMARY+'\n\n'+NEXT+'\n');insert('audit/团队开发清单.md',BODY.replace('(evidence/','(execution-search/'))
 for p in [ROOT/'source-frontier.json',Path('audit/first-pass.json')]:
  d=json.loads(p.read_text());d.update(target_draft_metadata_review=f,status=f['status'],next_action=NEXT,whole_chain_complete=False,completeness_established=False);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 for name,default in [('171-推理数据流与已查节点.html',False),('171-融合算子全盘排查进度图.html',False),('171-框架Profile热点.html',True)]:
  p=Path(name);old=p.read_text();pattern=PATTERN if default else r'<aside id="(?:input-resource-followup|target-draft-followup)"[^>]*>.*?</aside>';s,n=re.subn(pattern,lambda _:banner('audit/execution-search/'+REPORT,default),old,count=1,flags=re.S);assert n==1 and re.sub(pattern,'',old,flags=re.S)==re.sub(pattern,'',s,flags=re.S);p.write_text(s)
 for p in [Path('tasks/execution-derived-fusion-search/index.json'),Path('tasks/execution-derived-fusion-search/loops/loop-047/state.json')]:
  d=json.loads(p.read_text());d['next_action']=NEXT;p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 p=Path('tasks/execution-derived-fusion-search/loops/loop-047/status.md');s=p.read_text();s=re.sub(r'^- 下一步：.*$',lambda _:'- 下一步：'+NEXT,s,flags=re.M);p.write_text(s)
 p=Path('tasks/execution-derived-fusion-search/loops/loop-047/report.md');p.write_text(p.read_text()+'\n\n## Run155707Z · gid12整段交付（仅有界PASS/NA）\n\n'+SUMMARY+'\n\n'+DETAIL+'\n\n下一Run163605Z PLANNED，继续66位与递归callee。\n')
 print(json.dumps(dict(source_pins=15,original_Python_controls=34,bound=90,pending=66,new_admissions=0,whole_chain_complete=False)))
if __name__=='__main__':main()
