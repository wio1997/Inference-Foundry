#!/usr/bin/env python3
# coding: utf-8
"""Keep selected-dispatch facts and unclosed control/dataflow sites separate."""
from pathlib import Path
from collections import Counter
import json,re
ROOT=Path('audit/execution-search')
TITLE='2026-11-10 全模型注册与继承绑定：49源码约束，107输入控制位仍待扫'
SUMMARY='介绍：从真实注册、C3/别名、冻结配置与逐层职责建立执行入口，再与旧范围相交。到了哪里：40当前源码、config43d3/serve c297未变；16方法解析与Python C3对照、40target角色/3draft与私有128专家TopK3标量控制；156原静态分支/循环/上下文位逐项记录。结果：49受限源码/角色/排除绑定，107仍未闭合，不是107算子或已完成路径。DSpark继承/别名与实际_propose已补，rawMTPbank存在但固定DSpark数值路径不消费；同预算旧terminal小方案不重开。新框架/原语准入0，全链路未扫全，每替换原语逐位一致。'
NEXT='loop047 Run run-20261110T124045Z：继续实际V1 worker metadata/input的107剩余位，核输入Tensor来源、selected builder/资源组、canonical QSL与每个first owner→全部base/native/model graph消费者和CPU/producer/consumer退役及覆写；区分V2/A5和固定/动态条件。40源派发Run114154Z已PASS/NA；新Run仍PLANNED，配置两形态标量补充不当真实对象或原生位。只读源码/旧数据，无Torch/NPU/模型/新profile/服务动作。'
BODY=SUMMARY+'\n\n待确认、值得深入：较大共同CP frame的CPU/scalar/canonical QSL/设备长度交付；冻结V1 executor不capture producer，A3 builder不启用metadata_in_graph，producer输入地址与model graph地址分开审查。不能据此跳过native位、全部读者/事件或输入租约，增加共享读者可延长末读；旧device-only2.098217%不是整parent上界。107处还须继续callsite/人口/消费者，不把源码行交集当搜索完成。\n\n配置补充：原HF类体展平控制保留vision_n_layers32，而draft arch-info的is_mm_prefix_lm=False；两者不是同一字段。纯文本/DSpark不能推出vision-router bias不存在，实际HF/text/gate与token谓词仍待绑定。已确认可实现：仅上轮受限CPU原型1保留，本轮新增融合0。已停止：静态18入口当全覆盖、flag1当CANN MegaMoe、raw bank不存在、凭text-only删除vision-bias与同预算独立terminal copy小方案；更大parent保留。\n\n[完整判断](evidence/全模型派发继承与合法分支覆盖判断.md) · [注册/C3/40角色](evidence/whole-forward-dispatch.json) · [156位与旧范围交集](evidence/whole-forward-coverage-delta.json) · [配置形态对照](evidence/whole-forward-config-shape-controls.json) · [源行段](evidence/whole-forward-binding-source-excerpts.json)。\n\n下一条待扫：'+NEXT+'\n'
def default_banner(link):
    return '<aside id="whole-forward-default-entry" style="padding:14px 22px;margin:14px;border:1px solid #8daeb7;border-radius:8px;background:#f3f9fa;color:#193f4c;font:14px/1.7 system-ui"><strong>当前源码搜索：全链路未闭合，49受限绑定 / 107位待追</strong><p>40当前源、16方法派发与40层角色已核；静态入口、历史候选和时间线记录数量分别保留。新框架准入0。下一段追V1输入/metadata至实际消费者与退役。</p><a href="'+link+'">注册、继承、分支与开发前沿</a></aside>'
def insert(path,body):
    p=Path(path);s=p.read_text();h='## '+TITLE
    if h in s:
        a=s.index(h);z=s.find('\n## ',a+len(h));z=len(s) if z<0 else z+1;p.write_text(s[:a]+h+'\n\n'+body.rstrip()+'\n\n'+s[z:]);return
    if p.name in ['HANDOFF.md','ENVIRONMENT.md']:i=s.index('## 2026-11-10 共同设备投影命中后死工作')
    else:i=s.find('\n')+1;body='\n'+h+'\n\n'+body;p.write_text(s[:i]+body.rstrip()+'\n\n'+s[i:]);return
    p.write_text(s[:i]+h+'\n\n'+body.rstrip()+'\n\n'+s[i:])
def main():
    d=json.loads((ROOT/'whole-forward-dispatch.json').read_text());delta=json.loads((ROOT/'whole-forward-coverage-delta.json').read_text());controls=json.loads((ROOT/'whole-forward-config-shape-controls.json').read_text());unknown=Counter(b['root_owner']+'.'+b['root_method'] for b in delta['branches'] if b['condition_state']=='UNRESOLVED_INPUT_OR_RUNTIME')
    frontier=dict(status='ACTIVE_LOOP047_SELECTED_DISPATCH_BOUND107_CONTROL_AND_CONSUMER_GAPS',classification='待确认，继续全链路',sources_current=40,method_resolutions=16,target_roles=40,draft_layers=3,branch_sites=156,scoped_source_bindings=49,unclosed_input_or_control_sites=107,unclosed_groups=dict(unknown),branch_states=delta['branch_states'],whole_callgraph_closed=False,source_range_intersection_is_execution_coverage=False,config_shape_controls=2,config_shape_qualification=controls['qualification'],worth_deepening=delta['next_deliveries'],reused_terminal_budget=delta['reused_terminal_budget'],remaining=delta['remaining_selection_gaps'],next_action=NEXT,evidence=['whole-forward-dispatch.json','whole-forward-coverage-delta.json','whole-forward-config-shape-controls.json','whole-forward-binding-source-excerpts.json'],new_framework_replacement_admissions=0,new_primitive_admissions=0,isolated_CPU_prototypes_passed=1,whole_chain_complete=False)
    (ROOT/'whole-forward-frontier.json').write_text(json.dumps(frontier,ensure_ascii=False,indent=2)+'\n')
    insert('HANDOFF.md',BODY.replace('(evidence/','(audit/execution-search/')+'\nloop047保持EVALUATING；Run114154Z仅源码派发/标量/旧交集PASS，源自原代码的两配置形态控制单列，不覆盖真实Transformers初始化。新Run124045Z总体仍PLANNED，无已确认live任务等待。40 current源外另复用同SHA device_metadata引用、原aux selector父源证明；不是全部对象/loaded binary重核。公开同步回执另记。')
    for n in ['M00','M01','M03','M06','M08','M10','M11']:insert('audit/modules/'+n+'.md',BODY.replace('(evidence/','(../execution-search/'))
    insert(ROOT/'搜索范围与当前断点.md',BODY.replace('(evidence/','('));insert('audit/code-led/当前断点.md',SUMMARY+'\n\n'+NEXT);insert('audit/团队开发清单.md',BODY.replace('(evidence/','(execution-search/'))
    insert('ENVIRONMENT.md','适用冻结DeepSeek-V4.1-Flash-1017 W8A8/CANN9.2/910C，2026-11-10核实。171统一hostname成功，同owned容器/image；40份当前source identities、模型config43d3与serve c297一致。仅stdlib docker cp stdout tar/host白名单配置读取，未展开shell、未打印凭据/原环境；观察running不作health或loaded对象证明。原纯标量定义、C3及配置stub控制只在本机执行，无Torch/Transformers/模型/NPU进口，新请求/profile/服务动作/远端写入0。前述CPU/NPU旧回执按各自适用范围保留。')
    p=ROOT/'source-frontier.json';x=json.loads(p.read_text());x.update(status=frontier['status'],region='Wholemodel registered/MRO/scalar role bindings;107remaining selected input/control/dataflow sites',next_action=NEXT,completeness_established=False,whole_forward_dispatch_review=frontier,new_framework_replacement_admissions=0);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
    p=Path('audit/first-pass.json');x=json.loads(p.read_text());x['latest_execution_derived_followup']=dict(status='ACTIVE_NOT_COMPLETE',summary=SUMMARY,next_action=NEXT,evidence='audit/execution-search/whole-forward-frontier.json',new_framework_replacement_admissions=0,whole_chain_complete=False);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
    for name in ['171-推理数据流与已查节点.html','171-融合算子全盘排查进度图.html']:
        p=Path(name);s=p.read_text();banner='<aside id="whole-forward-followup" style="padding:18px 24px;border:1px solid #c9d9de;margin:18px;border-radius:8px;background:#f5f9fa;color:#203a45;font:14px/1.7 system-ui"><strong>全链路未扫全 · 49源码绑定 / 107位未闭合 · 新框架准入0</strong><p>'+SUMMARY+'</p><p>'+NEXT+'</p><p><a href="audit/execution-search/全模型派发继承与合法分支覆盖判断.md">真实派发与覆盖缺口</a> · <a href="HANDOFF.md">断点</a></p></aside>';s,n=re.subn(r'<aside id="(?:common-device-followup|whole-forward-followup)".*?</aside>',lambda _:banner,s,count=1,flags=re.S);assert n==1;p.write_text(s)
    p=Path('171-框架Profile热点.html');s=p.read_text();banner=default_banner('audit/execution-search/全模型派发继承与合法分支覆盖判断.md')
    if 'id="whole-forward-default-entry"' in s:s,n=re.subn(r'<aside id="whole-forward-default-entry".*?</aside>',lambda _:banner,s,count=1,flags=re.S)
    else:s,n=re.subn(r'(<body[^>]*>)',lambda m:m.group(1)+banner,s,count=1)
    assert n==1;p.write_text(s)
    p=Path('tasks/execution-derived-fusion-search/index.json');x=json.loads(p.read_text());assert x['active_loop']=='loop-047';x['next_action']=NEXT;p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
    p=Path('tasks/execution-derived-fusion-search/loops/loop-047/state.json');x=json.loads(p.read_text());x.update(status='EVALUATING',next_action=NEXT);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');p=p.with_name('status.md');s=p.read_text();s,n=re.subn(r'^- 下一步：.*$',lambda _: '- 下一步：'+NEXT,s,count=1,flags=re.M);assert n==1;p.write_text(s)
if __name__=='__main__':main()
