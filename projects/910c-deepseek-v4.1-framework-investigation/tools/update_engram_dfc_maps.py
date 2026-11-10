#!/usr/bin/env python3
"""Record paid Engram scope and DFC row ownership without claiming completion."""
from pathlib import Path
import json,re
ROOT=Path('audit/execution-search')
TITLE='2026-11-10 Engram全局owner付成本剪枝与DFC输入行共享：继续真实路由读者'
SUMMARY='介绍：全局Engram rows→WKV/gate/HC付成本判断，再沿较大DFC的BF16输入/量化/路由/跨EP搬运追到首GMM。到了哪里：Engram9源、56几何/4字节oracle、108C8/216短窗价、544比例与六普通代理；DFC9源18限定行段、原host DoTiling CPU12形状、1032匹配父段。结果：Engram仅明确小预算子情景停止，大parent待确认；DFC单核已共享但跨核仍重复，localM3有18次量化/3输入行，prefill416仅多20次，不能统一取6倍。targetDFC占自身两窗15.979%/17.537%，删整段10%另付h5条件净1.343%/1.496%窗口，量化内部占比未知。新准入0，全链路未扫全，每替换原语逐位一致。'
NEXT='loop044 Run run-20261110T094003Z：原expandedRowIdx sort/active mask/dummy/count生产→unique sender row map→远端gather→receiver连续gmA与GMM首读；保留原flags/容量/K/Fixpipe/舍入，追bank末读，付新map/gather/并行度成本。先用旧profile，不做NPU/框架进口/新profile/服务动作。'
BODY=SUMMARY+'\n\n待确认：更大Engram lookup/globalowner/WKV/gate/HC的原生位/资源/传输/末读与请求曝光；DFC稳定行映射、mask/dummy/zero-work完整分支、peer/gather或GMM A loader、原量化位与内部phase和新增成本。已确认可实现：本轮0。已停止：Engram已明确低于旧排序预算的独立子情景、N512/1024单AIC八panel同驻、flag-only分片及错误8倍MAC/B读取信用；DFC全形状unique-once/每路由独立重算的错误信用、把整个DFC当量化phase。只停止这些范围，不剪更大parent。\n\n[Engram全局owner与付成本价](evidence/Engram全局行Nowner几何与付成本请求判断.md) · [DFC输入行与首GMM判断](evidence/DFC输入行量化共享与首GMM交付判断.md) · [原host回放](evidence/dfc-input-host-cpu.json) · [DFC完整父段价](evidence/dfc-input-delivery-price.json)。\n\n下一条待扫：'+NEXT+'\n'
def insert(path,body):
    p=Path(path);s=p.read_text();h='## '+TITLE
    if h in s:return
    i=s.find('\n')+1
    # Preserve the other authorized collector's live operation notice first.
    if p.name=='HANDOFF.md' and '## 正在采集 · 2026-11-10：完整请求与逐函数时间账单' in s:
        i=s.index('## 2026-11-10 HC双parent跨层与FULL图')
        p.write_text(s[:i]+h+'\n\n'+body+'\n'+s[i:]);return
    p.write_text(s[:i]+'\n'+h+'\n\n'+body+'\n'+s[i:])
def main():
    insert('HANDOFF.md',BODY.replace('(evidence/','(audit/execution-search/')+'\nloop043PIVOTED；loop044入口Run092257Z PASS/正确性NA，后续Run094003Z仍PLANNED。Engram初次runningguard失败/两时点container观察/价格A2A缺失解析失败原件保留；16处无原A2A用明示其它parent proxy，不造零成本。普通完整响应代理校正为.824710–1.418414%，WKV-only在旧代理与预算内全free至多.9988565%，不停止input/lookup/gate/HC大parent。公开同步回执另记。')
    for n in ['M01','M06','M07','M11']:insert('audit/modules/'+n+'.md',BODY.replace('(evidence/','(../execution-search/'))
    insert(ROOT/'搜索范围与当前断点.md',BODY.replace('(evidence/','('));insert('audit/code-led/当前断点.md',SUMMARY+'\n\n'+NEXT);insert('audit/团队开发清单.md',BODY.replace('(evidence/','(execution-search/'))
    insert('ENVIRONMENT.md','适用冻结DeepSeek-V4.1-Flash-1017 W8A8/CANN9.2/910C，2026-11-10核实。171统一入口hostname已核，同owned容器3be44e7b5c29/image ce42d65b。Engram时观察exited/exit0，runningguard失败留件；随后逐文件docker cp stdouttar内存只读核9源。后续DFC六入口/三host header在同id/image观察running，均全SHA匹配；两个时点不等于health或其它采集恢复回执。本扫描没有重启、服务/健康/模型请求、新NPU或profile、远端写入；未进口Torch/NPU。原DFC host头仅本机CPU隔离编译执行12形状。其它97函数计时流程的临时hook/采集与服务恢复由其原回执说明，不改其文件。')
    p=ROOT/'source-frontier.json';d=json.loads(p.read_text());d.update(status='ACTIVE_LOOP044_DFC_INPUT_ROW_OWNERS_AND_FIRST_GMM',region='Engramboundedpaidrowowner; DFCdynamicquant/TopKrouting/dispatch to actualGMMconsumer',next_action=NEXT,completeness_established=False,engram_row_owner_review=json.loads((ROOT/'engram-row-owner-frontier.json').read_text()),dfc_input_delivery_review=json.loads((ROOT/'dfc-input-delivery-frontier.json').read_text()));p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    p=Path('audit/first-pass.json');d=json.loads(p.read_text());d['latest_execution_derived_followup']=dict(status='ACTIVE_NOT_COMPLETE',summary=SUMMARY,next_action=NEXT,Engram='audit/execution-search/engram-row-owner-frontier.json',DFC='audit/execution-search/dfc-input-delivery-frontier.json',whole_chain_complete=False,new_primitive_admissions=0);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    for name in ['171-推理数据流与已查节点.html','171-融合算子全盘排查进度图.html']:
        p=Path(name);s=p.read_text();banner='<aside id="engram-dfc-row-owner-followup" style="padding:18px 24px;border:1px solid #c9d9de;margin:18px;border-radius:8px;background:#f5f9fa;color:#203a45;font:14px/1.7 system-ui"><strong>当前覆盖：全链路未扫全 · 每替换原语逐位一致</strong><p>'+SUMMARY+'</p><p>'+NEXT+'</p><p><a href="audit/execution-search/DFC输入行量化共享与首GMM交付判断.md">DFC行共享与实际读者</a> · <a href="HANDOFF.md">当前断点</a></p></aside>'
        s,n=re.subn(r'<aside id="(?:hc-engram-consumer-followup|engram-dfc-row-owner-followup)".*?</aside>',lambda m:banner,s,count=1,flags=re.S);assert n==1;p.write_text(s)
    p=Path('tasks/execution-derived-fusion-search/index.json');d=json.loads(p.read_text());assert d['active_loop']=='loop-044';d['next_action']=NEXT;p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
