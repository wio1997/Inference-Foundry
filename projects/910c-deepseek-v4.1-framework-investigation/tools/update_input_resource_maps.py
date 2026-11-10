#!/usr/bin/env python3
"""Advance current resource/consumer map without erasing prior cohort evidence."""
from pathlib import Path
import json,re
ROOT=Path('audit/execution-search')
TITLE='2026-11-10 真实资源组与坐标退役：90源码约束，66位待追'
REPORT='实际资源组与Engram_DSpark坐标及退役判断.md'
NEXT='loop047 Run155707Z（PLANNED）：先核target阶段gid12 draft builder整段输出的数值/私有cache/first-owner task/FULL及后续draft读者，保留CM与原物理view，用旧逐call整段价判去留；再继续66位及递归callee，走selected execute_model状态协议、DP空批/异步修正、KV/EC/offload/Mamba/EPLB/DBO gate与metadata provider submit/release到真实消费者；区分合法动态源码分支审阅与运行人口。已绑定13KV/26runner/1draft、A3 Engram同步默认；保留actual/FIA/draft坐标域、CPU copy lease、producer/consumer双退役及完整packet位。先用旧C8/六普通，不重开已停小岛；无新Torch/NPU/模型/profile/请求/服务或远端变更。'
SUMMARY='介绍：沿实际Ascend cache patch/原spec与planner→V1实际QSL/物理地址→Engram hash与DSpark context/query读者，再核旧原生任务。到了哪里：30当前源码身份；756文件仅关键词盘点；原源码计划54资源/13KV组/26runner builder/1draft builder，19退役/CPUguard控制、2坐标域反例，3624活跃rank窗各1次扩展；新增13处，原156位90绑定/66待追。结果：停止不存在的多组重复、当前A3基线mask-ready等待后移和固定native小岛独立配方；更大CPU/scalar/coordinate frame继续。原语/框架新准入0，全链路未闭合，每替换原语逐位一致。'
DETAIL='待确认：大frame的actual/FIA/draft正确视图、原physical/packet owner、CPU H2D读租约与producer_done+consumer_done、最早storage覆写、完整4096B packet/尾部和净成本/六普通曝光。V1 finally只release metadata，V2 finish/captured helper join不能当V1保证；当前A3 overlap默认False、savedserve未覆盖，无辅助流竞态观测。新增overlap/提前producer必须另闭合prep/model/错误退役。\n\n已停止：重复draft公共生成（source1组且原C8每活跃窗1kernel）；当前不存在的mask wait后移；盲共享FIA padded req-map给Engram（尾行clamp不同）；原C8固定hash3kernel+唯一expansion及内部gap全免仅0.365136%TTFT/0.524144%整批，50%实删暴露/新增0.02ms每波条件净0.175865%/0.237651%，非实测，不作全host/Engram上界、不迁移普通负载或与大parent相加。已确认可实现：本轮0，旧封闭CPU租约原型1保留。'
LINKS='[本段判断](evidence/'+REPORT+') · [90/66逐位账/消费者](evidence/input-resource-source-review.json) · [13KV/26builder/1draft源计划](evidence/input-resource-source-plan.json) · [原C8逐Task资格/价](evidence/input-kernel-windows-receipt.json) · [条件价格](evidence/input-kernel-island-price.json) · [19原方法控制](evidence/input-retirement-controls.json)。'
BODY=SUMMARY+'\n\n'+DETAIL+'\n\n'+LINKS+'\n\n下一条待扫：'+NEXT+'\n'
PATTERN=r'<aside id="(?:embedding-static-default-entry|input-resource-default-entry)"[^>]*>.*?</aside>'
def banner(link,default=True):
    ident='input-resource-default-entry' if default else 'input-resource-followup'
    return '<aside id="'+ident+'" style="padding:16px 22px;margin:14px;border:1px solid #8daeb7;border-radius:8px;background:#f3f9fa;color:#193f4c;font:14px/1.7 system-ui"><strong>当前源码搜索：90受限绑定 / 66位待追，全链路未闭合</strong><p>真实源计划13KV组/26runner builder/1draft；3624旧活跃窗各1次扩展。A3默认无Engram异步mask等待。固定native两岛全免仅0.52%整批，不是大frame上界；继续CPU/scalar/坐标与完整位/退役。新原语/框架准入0。</p><a href="'+link+'">资源域、末读/覆写、价格与下一断点</a></aside>'
def insert(path,body):
    p=Path(path);s=p.read_text();assert '## '+TITLE not in s;i=s.index('\n')+1;p.write_text(s[:i]+'\n## '+TITLE+'\n\n'+body+'\n'+s[i:])
def frontier():
    r=json.loads((ROOT/'input-resource-source-review.json').read_text())
    return dict(status='ACTIVE_LOOP047_INPUT_RESOURCE_CONSUMER_90_BOUND_66_PENDING',classification='待确认，继续全链路',current_source_pins=30,original_source_retirement_controls=19,coordinate_counterexamples=2,original_static_sites=156,scoped_source_bindings=90,unclosed_input_control_sites=66,source_resource_plan=r['resource_source_plan'],selected_A3_engram_overlap=False,source_plan_is_live_object_dump=False,active_rank_windows=3624,all_rank_windows=3696,native_expansion_calls_per_active_window=1,lexical_package_files=756,lexical_inventory_is_source_coverage=False,stopped_scopes=r['stopped'],worth_deepening=r['worth_deepening'],remaining=r['remaining'],every_replacement_bitwise=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,isolated_CPU_prototypes_passed=1,whole_callgraph_closed=False,whole_chain_complete=False,next_action=NEXT,evidence=[REPORT,'input-resource-source-review.json','input-resource-source-plan.json','input-retirement-controls.json','input-kernel-windows-receipt.json','input-kernel-island-price.json','input-frozen-flags.json'])
def main():
    f=frontier();(ROOT/'input-resource-frontier.json').write_text(json.dumps(f,ensure_ascii=False,indent=2)+'\n')
    insert('HANDOFF.md',BODY.replace('(evidence/','(audit/execution-search/')+'\nRun150110Z有界PASS/正确性NA；Run155707Z PLANNED。loop047保持EVALUATING，无live任务等待；公开同步回执待补。\n')
    for n in ['M00','M01','M02','M03','M10','M11']:insert('audit/modules/'+n+'.md',BODY.replace('(evidence/','(../execution-search/'))
    insert(ROOT/'搜索范围与当前断点.md',BODY.replace('(evidence/','('));insert('audit/code-led/当前断点.md',SUMMARY+'\n\n'+NEXT+'\n');insert('audit/团队开发清单.md',BODY.replace('(evidence/','(execution-search/'))
    for p in [ROOT/'source-frontier.json',Path('audit/first-pass.json')]:
        d=json.loads(p.read_text());d.update(input_resource_consumer_review=f,status=f['status'],next_action=NEXT,whole_chain_complete=False,completeness_established=False);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    for name,default in [('171-推理数据流与已查节点.html',False),('171-融合算子全盘排查进度图.html',False),('171-框架Profile热点.html',True)]:
        p=Path(name);old=p.read_text();pattern=PATTERN if default else r'<aside id="(?:embedding-static-followup|input-resource-followup)"[^>]*>.*?</aside>';s,n=re.subn(pattern,lambda _:banner('audit/execution-search/'+REPORT,default),old,count=1,flags=re.S);assert n==1 and re.sub(pattern,'',old,flags=re.S)==re.sub(pattern,'',s,flags=re.S);p.write_text(s)
    for p in [Path('tasks/execution-derived-fusion-search/index.json'),Path('tasks/execution-derived-fusion-search/loops/loop-047/state.json')]:
        d=json.loads(p.read_text());d['next_action']=NEXT;p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    p=Path('tasks/execution-derived-fusion-search/loops/loop-047/status.md');s=p.read_text();s=re.sub(r'^- 下一步：.*$',lambda _:'- 下一步：'+NEXT,s,flags=re.M);p.write_text(s)
    p=Path('tasks/execution-derived-fusion-search/loops/loop-047/report.md');p.write_text(p.read_text()+'\n\n## Run150110Z · 真实资源组/坐标/退役（有界PASS/NA）\n\n'+SUMMARY+'\n\n'+DETAIL+'\n\n下一Run155707Z仍PLANNED，继续66位和递归callee。\n')
    print(json.dumps(dict(source_pins=30,bound=90,pending=66,new_primitive_admissions=0,whole_chain_complete=False)))
if __name__=='__main__':main()
