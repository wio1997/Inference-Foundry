#!/usr/bin/env python3
"""Record actual guard/consumer selections and preserve larger delivery scopes."""
from pathlib import Path
from hashlib import sha256
import json,re
ROOT=Path('audit/execution-search')
TITLE='2026-11-11 模型guard与实际派发：122源码约束，34动态位未闭合'
REPORT='模型guard与SP行及旋转Engram交付判断.md'
NEXT='loop047 Run174546Z PLANNED：沿真实V1 _prepare_inputs/_preprocess追target/draft ID、position、hidden尾部的所有写者/末读与行映射，到M06 router/dispatch/shared/HC/aux/下一轮消费者。区分actual/FIA/graph/DP/SP，不把is_padding写入当路由已跳过padding；先核大row交付的实际人口/完整输出义务。34动态原位及递归callee未闭合，不重扫122固定约束；保留大metadata/gid12和Engram owner/input候选。source-first整段边界后用旧C8/六普通，单独核费用/曝光，不跨cohort相加；每替换原语逐位一致。此源审Run无新Torch/NPU/模型/profile/请求/服务或远端变更。'
SUMMARY='介绍：从5个模型/attention入口guard追SP mask、实际router、Engram旋转和generic runner/cache两条backend派发。到了哪里：27当前源身份、21进程初始单项env、85原Python selector/布尔行/opaque控制；新增3处，原156位122受限绑定/34动态待追。结果：5处源码条件已说明，不是5处原生人口闭合；全callgraph仍未扫全，新增原语/框架准入0，每替换原语逐位一致。'
DETAIL='已停止：把未选中的分解vision/hash fallback当A3热路径再融合（当前cap已含原生hash/vision）；将A5 unrotated gate替当前nonNone rotation；寻找一个该generic selector源链并不需要的use_compress注入；原样重跑既有独立Engram Wkv/gate/常量小配方。Wkv有quant_config参数但描述仍FLOAT，旧原生任务BF16；初版quantized标签已纠正并保留原件，整段范围未缩小。\n\n旧原C8 first-token匹配：当前6源与旧身份相交、量化描述SHA同2f4020；完整gate全免最大0.489683%TTFT，Wkv0.615615%，常量/重复widening0.037375%。Wkv+完整gate+copy1.115388%、再含紧邻HC Pre1.207905%仍包含必要数学/残差，不是预计净收益，更不是lookup/TP/host上界。50%保留工作情景见原arrival模型，既有有限M3/M15 native fixture按历史资格保留，本轮不重跑或新增准入。\n\n待确认、继续：更大的跨模块row来源→route/dispatch→expert/shared→HC/aux交付。旧EP0/EP8代表rank、localM3形状窗中，global24整DFC全免15.979–17.537%，norm→Post均22.505%；这只是两代表卡旧窗口，非全16卡/请求上界。若真正删掉DFC时间2.5%且每call新增5us，局部均净1.516ms/0.162240%该窗；25%为36.759ms/3.932827%。时间份额不是padding行比例，未知默认0时新增费用为负收益；请求实际eligible calls/曝光和六普通未绑，新增row请求信用默认0，不迁移1025ms诊断或原C8/TTFT。\n\nSP helper只补模型shape对TP的余数；输入mask=None、shape8时不会自动知道actual6。当前Ascend router/BaseRouter所读路径不直接消费is_padding，不能因flag默认True就认padding expert已被跳过，也不能未闭合runner/dispatch/可选reader便删发布。maskNone的profile路径与本卡query0不同，后者保留global cache准备、TP交换、补齐和O投影。generic runner源链选MLA sibling、V41逐层cache backend独立；其updatable predicate不是GQA路径，gid12该源歧义收窄，但捕获地址/全部读span、task reset/异常与native位/净价仍待确认。\n\n已确认可实现：本轮0；旧封闭CPU租约原型1及历史有限native常量fixture分别保留原资格。source、初始env、opaque控制均不当loaded对象/真实事件或数值逐位验收；错误读路径、AST helper扩展、2代表rank误数及量化标签纠正均有记录。'
LINKS='[本段判断](evidence/'+REPORT+') · [122/34与实际消费者](evidence/model-gate-source-review.json) · [85源码控制](evidence/model-gate-source-controls.json) · [先验大row范围](evidence/model-pad-row-parent-primary.json) · [分cohort局部/条件请求价](evidence/model-gate-parent-prices.json)。'
BODY=SUMMARY+'\n\n'+DETAIL+'\n\n'+LINKS+'\n\n下一条待扫：'+NEXT+'\n'
PATTERN=r'<aside id="(?:worker-protocol-default-entry|model-gate-default-entry)"[^>]*>.*?</aside>'
def banner(link,default=True):
 ident='model-gate-default-entry' if default else 'model-gate-followup'
 return '<aside id="'+ident+'" style="padding:16px 22px;margin:14px;border:1px solid #8daeb7;border-radius:8px;background:#f3f9fa;color:#193f4c;font:14px/1.7 system-ui"><strong>当前：122源码约束 / 34动态位未闭合；全链路仍未扫全</strong><p>27当前源、5模型guard源码条件、85 opaque/行几何控制。A3 hash/vision已有原生融合；当前旋转Engram走generic gate，Wkv仍FLOAT/BF16。保留跨模块padding行/路由与大metadata候选，旧小配方不重跑。新原语/框架准入0，每替换原语逐位一致。</p><a href="'+link+'">实际派发、整段价与下一断点</a></aside>'
def insert(path,body):
 p=Path(path);s=p.read_text();assert '## '+TITLE not in s;i=s.index('\n')+1;p.write_text(s[:i]+'\n## '+TITLE+'\n\n'+body+'\n'+s[i:])
def frontier():
 r=json.loads((ROOT/'model-gate-source-review.json').read_text());return dict(status=r['status'],classification='待确认，继续全链路',current_source_pins=27,initial_exec_processes=21,original_static_sites=156,scoped_source_bindings=122,pending_original_sites=34,model_guard_source_legs=5,original_Python_controls=85,whole_callgraph_closed=False,whole_chain_complete=False,every_replacement_bitwise=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,isolated_CPU_prototypes_passed=1,remaining=r['remaining'],stopped=r['stopped'],worth_deepening=r['worth_deepening'],next_action=NEXT,evidence=[REPORT,'model-gate-source-review.json','model-gate-source-controls.json','model-gate-parent-prices.json','model-pad-row-parent-primary.json','model-gate-review-history.json'])
def main():
 f=frontier();(ROOT/'model-gate-frontier.json').write_text(json.dumps(f,ensure_ascii=False,indent=2)+'\n')
 (ROOT/REPORT).write_text('# 模型guard、SP行与旋转Engram交付\n\n'+BODY.replace('(evidence/','(')+'\nRun170051Z仅此有界源码/旧price PASS，正确性NA；Run174546Z PLANNED。6源相交不当旧19身份全部重核；量化描述8.68MB仅读JSON，权重tensor payload读取0。27身份不当27全文；756文件只是term盘点。EP0/8 shape窗不外推全16卡，更不和其它cohort收益相加。新Torch/NPU/模型/profile/请求/服务/远端写入0，未重测live健康。\n')
 shared=['HANDOFF.md','171-推理数据流与已查节点.html','171-框架Profile热点.html','171-融合算子全盘排查进度图.html'];(ROOT/'model-gate-shared-before.json').write_text(json.dumps({n:sha256(Path(n).read_bytes()).hexdigest() for n in shared},ensure_ascii=False,indent=2)+'\n')
 insert('HANDOFF.md',BODY.replace('(evidence/','(audit/execution-search/')+'\nRun170051Z有界PASS/NA，Run174546Z PLANNED。公开同步回执待补；loop047活动，无live任务等待。\n')
 for n in ['M00','M01','M02','M03','M06','M10','M11']:insert('audit/modules/'+n+'.md',BODY.replace('(evidence/','(../execution-search/'))
 insert(ROOT/'搜索范围与当前断点.md',BODY.replace('(evidence/','('));insert('audit/code-led/当前断点.md',SUMMARY+'\n\n'+NEXT+'\n');insert('audit/团队开发清单.md',BODY.replace('(evidence/','(execution-search/'))
 for p in [ROOT/'source-frontier.json',Path('audit/first-pass.json')]:
  d=json.loads(p.read_text());d.update(model_gate_review=f,status=f['status'],next_action=NEXT,whole_chain_complete=False,completeness_established=False);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 for name,default in [('171-推理数据流与已查节点.html',False),('171-融合算子全盘排查进度图.html',False),('171-框架Profile热点.html',True)]:
  p=Path(name);old=p.read_text();pattern=PATTERN if default else r'<aside id="(?:worker-protocol-followup|model-gate-followup)"[^>]*>.*?</aside>';s,n=re.subn(pattern,lambda _:banner('audit/execution-search/'+REPORT,default),old,count=1,flags=re.S);assert n==1 and re.sub(pattern,'',old,flags=re.S)==re.sub(pattern,'',s,flags=re.S);p.write_text(s)
 for p in [Path('tasks/execution-derived-fusion-search/index.json'),Path('tasks/execution-derived-fusion-search/loops/loop-047/state.json')]:
  d=json.loads(p.read_text());d['next_action']=NEXT;p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 p=Path('tasks/execution-derived-fusion-search/loops/loop-047/status.md');s=p.read_text();s=re.sub(r'^- 下一步：.*$',lambda _:'- 下一步：'+NEXT,s,flags=re.M);p.write_text(s)
 p=Path('tasks/execution-derived-fusion-search/loops/loop-047/report.md');p.write_text(p.read_text()+'\n\n## Run170051Z · 模型guard/实际派发/大row交付（有界PASS/NA）\n\n'+SUMMARY+'\n\n'+DETAIL+'\n\n下一Run174546Z PLANNED，尾部写者/实际行域与递归callee继续。\n')
 print(json.dumps(dict(sources=27,controls=85,bound=122,pending=34,new_admissions=0)))
if __name__=='__main__':main()
