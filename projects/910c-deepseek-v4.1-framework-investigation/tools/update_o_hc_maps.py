#!/usr/bin/env python3
"""Reconcile new source/price coverage without touching concurrent profile ledgers."""
from pathlib import Path
import json,re
ROOT=Path('audit/execution-search')
TITLE='2026-11-10 O投影资源剪枝与HC premix提前交付：继续真实系数和跨层读者'
SUMMARY='介绍：OA/OB→SP→HC/FFN大交付，再沿实际HCPost/HCPreV3/RMS走到MLP和FFNPost。到了哪里：37O投影源/23限定body、20HC源/17限定body，两rank各516条源一致链，144 parent价、108资源价、144 fork/join价；原profile六条1800MHz记录。结果：当前M24单owner81.636us下界高于全部旧OA+OB，停止该schedule及擦掉OA舍入的预合成；多owner/大prefill待确认。premix存在时collapse不依赖本次新系数，提前交付e5/h5、无争用条件净1.340%/1.368%窗口，40us MLP争用会翻负；原BF16残差、collapse与RMS舍入/FP32扩宽均保留。新原语准入0，全链路未扫全，每替换原语逐位一致。'
NEXT='loop042 Run run-20261110T080543Z：HCPre stage2 post/comb/Sinkhorn/newpre到下一decoder、Engram与final读者及实际多流FULL图split/join；保留原coeff完整工作、原舍入/归约，查能改变去留的事件/资源/争用边界。仅只读源码/旧profile，不进口框架、不做NPU/服务/采样。'
BODY=SUMMARY+'\n\n待确认：多owner OA/OB的活跃key、原语位、聚合服务与净价；premix提前交付的原系数尾部/跨层代次/独立事件/FULL图/实际MLP争用；大prefill和普通请求曝光。已确认可实现：本轮0。已停止：擦掉OA舍入的普遍预合成、当前M24单owner全部OA/OB、原kernel间gap-only小范围；舍入前FP32直接喂router、删除后续FFNPost残差、无premix原样提前或跳过活跃coeff。只停止具体变换，不剪大parent。\n\n[O投影原生资源与大段价](evidence/O投影原生资源与SP残差FFN交付判断.md) · [HC原生与提前交付](evidence/HC原生舍入与premix提前交付判断.md) · [144 fork/join情景](evidence/hc-early-delivery-price.json)。\n\n下一条待扫：'+NEXT+'\n'
def insert(path,body):
 p=Path(path);s=p.read_text();h='## '+TITLE
 if h in s:return
 i=s.find('\n')+1;p.write_text(s[:i]+'\n'+h+'\n\n'+body+'\n'+s[i:])
def main():
 insert('HANDOFF.md',BODY.replace('(evidence/','(audit/execution-search/')+'\nloop041PIVOTED仅限定source/resource/price；loop042 Run075729Z PASS/正确性NA，Run080543Z仍PLANNED。首次clock路径误读INVALID原件保留，修正后主机旧DB读取成功。171统一hostname同owned image/id，仅只读磁盘/旧数据，running不代替health。其它流程性能核准与恢复回执保留；公开同步回执另记。')
 for m in ['M00','M04','M05','M10','M11']:insert('audit/modules/'+m+'.md',BODY.replace('(evidence/','(../execution-search/'))
 insert(ROOT/'搜索范围与当前断点.md',BODY.replace('(evidence/','('))
 insert('audit/code-led/当前断点.md',SUMMARY+'\n\n'+NEXT)
 insert('audit/团队开发清单.md',BODY.replace('(evidence/','(execution-search/'))
 insert('ENVIRONMENT.md','适用冻结DeepSeek-V4.1-Flash-1017 W8A8/CANN9.2/910C，2026-11-10核实。171统一hostname正确，owned容器/image/id守卫匹配；仅当前源码/142checkpoint header（payload0）/已有主机profile DB readonly immutable读取，未新采样或live查频。两代表旧DB六条AICORE_FREQ=1800MHz，初次容器路径误读记录INVALID保留；不推广到所有卡连续频率/当前服务健康。本轮新NPU、框架/模型进口、请求、profile、显式远端写入、服务动作0。')
 p=ROOT/'source-frontier.json';d=json.loads(p.read_text());d.update(status='ACTIVE_LOOP042_HC_PREMIX_NATIVE_EARLY_DELIVERY',region='Original O projection source/resource bounded; larger HC premix delivery and later coefficient/nextdecoder consumers',next_action=NEXT,completeness_established=False,o_projection_native_review=json.loads((ROOT/'o-projection-native-frontier.json').read_text()),hc_delivery_review=json.loads((ROOT/'hc-delivery-frontier.json').read_text()));p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 p=Path('audit/first-pass.json');d=json.loads(p.read_text());d['latest_execution_derived_followup']=dict(status='ACTIVE_NOT_COMPLETE',summary=SUMMARY,next_action=NEXT,OA='audit/execution-search/o-projection-native-frontier.json',HC='audit/execution-search/hc-delivery-frontier.json',whole_chain_complete=False,new_primitive_admissions=0);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 for name in ['171-推理数据流与已查节点.html','171-融合算子全盘排查进度图.html']:
  p=Path(name);s=p.read_text();banner='<aside id="o-hc-premix-followup" style="padding:18px 24px;border:1px solid #c9d9de;margin:18px;border-radius:8px;background:#f5f9fa;color:#203a45;font:14px/1.7 system-ui"><strong>当前覆盖：全链路未扫全 · 每替换原语逐位一致</strong><p>'+SUMMARY+'</p><p>'+NEXT+'</p><p><a href="audit/execution-search/HC原生舍入与premix提前交付判断.md">HC实际读者与条件净价</a> · <a href="HANDOFF.md">当前断点</a></p></aside>'
  s,n=re.subn(r'<aside id="(?:frame-o-projection-followup|o-hc-premix-followup)".*?</aside>',lambda m:banner,s,count=1,flags=re.S);assert n==1;p.write_text(s)
 p=Path('tasks/execution-derived-fusion-search/index.json');d=json.loads(p.read_text());assert d['active_loop']=='loop-042';d['next_action']=NEXT;p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
