#!/usr/bin/env python3
"""Record constrained CPU feasibility and advance the larger device delivery."""
from pathlib import Path
import json,re
ROOT=Path('audit/execution-search')
TITLE='2026-11-10 共同frame分层key与26caller：CPU原语受限通过，继续device读者'
SUMMARY='介绍：共同frame按rawCPU/postCPU/device/每资源交付分层，追真实caller、变异和图地址。到了哪里：12当前源/11限定body/29行段；16worker旧CPUparent日志确认每frame目标26+draft1；原CPU AST标量1728例，实际TorchCPU1728例/44928字节比较，guard变体3模式/78比较。结果：封闭不可变租约内CPU复用原型已确认可实现，服务框架替换准入0；v1 inference Tensor版本读取失败原件保留，模式guard通过但不证明不可变。CPU v1六组26call配对净省2.226–2.445ms/frame，非请求收益；真实caller/保留首body的新条件价1.964%/3.275%诊断窗。全链路未扫全，每替换原语逐位一致。'
NEXT='loop046 Run run-20261110T105846Z：一次原主流CP suffix/clamp/where→base首batchcache与全部common字段/原生closure读者→稳定QSL和capture/event firstowner。保留各builder QSL复制与物理输出，查是否能只删重复device长度投影；只读源码/已有profile，不新增NPU/模型/服务动作。CPU模式门槛Run110329Z已PASS，不能以guard省略输入租约。'
BODY=SUMMARY+'\n\n已确认可实现：仅封闭CPU输入租约下的隔离原语原型，含optional-out旁路和inference-aware key；未安装到服务。待确认：框架实际集成/全部保护负载、较大公共device投影的所有读者/地址/原生位与关键路径净价。已停止：全局helper cache漏out写、CPU双mirror合并、shape/版本号当不可变证明、遗漏TP/causal/dtype/actualTokens，以及假定V41draft继承Legacy缓存或有26次重复。旧大frame异步writer-only配方仍停，不剪原主流完整方案。\n\n[分层key、caller与CPU原语](evidence/共同frame分层键与真实caller及CPU原语判断.md) · [CPU模式资格补充](evidence/共同frame_CPU模式资格补充.md) · [真实caller与付成本价](evidence/common-frame-caller-price.json) · [实际CPU字节/局部成本](evidence/common-frame-torch-cpu.json) · [inference反例与guard](evidence/common-frame-cpu-modes.json)。\n\n下一条待扫：'+NEXT+'\n'
def insert(path,body):
    p=Path(path);s=p.read_text();h='## '+TITLE
    if h in s:return
    if p.name in ['HANDOFF.md','ENVIRONMENT.md']:
        i=s.index('## 2026-11-10 DFC原路由与末读闭合');p.write_text(s[:i]+h+'\n\n'+body+'\n'+s[i:]);return
    i=s.find('\n')+1;p.write_text(s[:i]+'\n'+h+'\n\n'+body+'\n'+s[i:])
def main():
    insert('HANDOFF.md',BODY.replace('(evidence/','(audit/execution-search/')+'\nloop045PIVOTED限定CPU/caller/价格，CPU Run104319Z PASS/CPU正确性PASS；loop046模式门槛Run110329Z PASS，原source Run105846Z仍PLANNED。首次SSH banner timeout留件，后续12源读取成功，无重启。普通CPU v1、inference guard v2和服务集成分列；v2三个单轮cost不是v1完整paired实验。90.347ms标记边界已扣，2个FT缺口保持，新144caller价取代旧mean27排序口径但不改旧原件。公开同步回执另记。')
    for n in ['M00','M03','M10','M11']:insert('audit/modules/'+n+'.md',BODY.replace('(evidence/','(../execution-search/'))
    insert(ROOT/'搜索范围与当前断点.md',BODY.replace('(evidence/','('));insert('audit/code-led/当前断点.md',SUMMARY+'\n\n'+NEXT);insert('audit/团队开发清单.md',BODY.replace('(evidence/','(execution-search/'))
    insert('ENVIRONMENT.md','适用冻结DeepSeek-V4.1-Flash-1017 W8A8/CANN9.2/910C，2026-11-10核实。171统一hostname已核，同owned容器3be44e7b5c29/image ce42d65b，12当前源全SHA匹配。首次SSH banner timeout留原件后仅重取只读source。新CPU进程用已安装Torch2.10.0+cpu/Python3.12.13，TORCH_DEVICE_BACKEND_AUTOLOAD=0、-S显式site与1CPU线程；前后无torch_npu/vllm和libtorch_npu/libascendcl/libhccl。两个私有映射目录runtime/cpu-frame-run-20261110T104319Z与T110329Z分别12.60s/1.78s完成，仅脚本/spec/tmp/cache/回执写入/data/tiankuan/wio；未改框架源/服务/全局环境，NPU/模型/推理/新profile动作0。CPU字节原型不代替服务插件状态下的完整回归，旧NPU/模型恢复证明仍按各自原回执。')
    p=ROOT/'source-frontier.json';d=json.loads(p.read_text());d.update(status='ACTIVE_LOOP046_ORIGINAL_MAINSTREAM_COMMON_DEVICE_READERS',region='Tieredcommonframe CPUkey/caller/prototype bounded; once originalmainstream deviceprojection to persistentreaders',next_action=NEXT,completeness_established=False,common_frame_key_review=json.loads((ROOT/'common-frame-key-frontier.json').read_text()),common_frame_CPU_mode_overlay=json.loads((ROOT/'common-frame-cpu-mode-frontier.json').read_text()),isolated_CPU_primitive_prototypes_passed=1,new_framework_replacement_admissions=0);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    p=Path('audit/first-pass.json');d=json.loads(p.read_text());d['latest_execution_derived_followup']=dict(status='ACTIVE_NOT_COMPLETE',summary=SUMMARY,next_action=NEXT,commonFrame='audit/execution-search/common-frame-key-frontier.json',CPUmode='audit/execution-search/common-frame-cpu-mode-frontier.json',isolated_CPU_prototypes_passed=1,new_framework_replacement_admissions=0,whole_chain_complete=False);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    for name in ['171-推理数据流与已查节点.html','171-融合算子全盘排查进度图.html']:
        p=Path(name);s=p.read_text();banner='<aside id="frame-key-cpu-followup" style="padding:18px 24px;border:1px solid #c9d9de;margin:18px;border-radius:8px;background:#f5f9fa;color:#203a45;font:14px/1.7 system-ui"><strong>全链路未扫全 · CPU原型受限通过1 · 框架替换准入0</strong><p>'+SUMMARY+'</p><p>'+NEXT+'</p><p><a href="audit/execution-search/共同frame分层键与真实caller及CPU原语判断.md">分层key与caller</a> · <a href="audit/execution-search/共同frame_CPU模式资格补充.md">inference资格</a> · <a href="HANDOFF.md">断点</a></p></aside>'
        s,n=re.subn(r'<aside id="(?:dfc-common-frame-followup|frame-key-cpu-followup)".*?</aside>',lambda m:banner,s,count=1,flags=re.S);assert n==1;p.write_text(s)
    p=Path('tasks/execution-derived-fusion-search/index.json');d=json.loads(p.read_text());assert d['active_loop']=='loop-046';d['next_action']=NEXT;p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    p=Path('tasks/execution-derived-fusion-search/loops/loop-046/state.json');d=json.loads(p.read_text());d['next_action']=NEXT;p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    p=p.with_name('status.md');s=p.read_text();s,n=re.subn(r'^- 下一步：.*$',lambda _: '- 下一步：'+NEXT,s,count=1,flags=re.M);assert n==1;p.write_text(s)
if __name__=='__main__':main()
