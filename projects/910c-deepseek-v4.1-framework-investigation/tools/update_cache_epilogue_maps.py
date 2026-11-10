#!/usr/bin/env python3
"""Reconcile bounded native and later join review without rewriting finished runs."""
from pathlib import Path
from hashlib import sha256
import json,re

ROOT=Path('audit/execution-search')
TITLE='2026-11-10 原生缓存发布与SWA主辅流join：局部3%不等于净省'
SUMMARY='介绍：真实OOT/native的indexK/longKV双发布扩大到SWA整段与Qb重叠。到了哪里：39当前source pins/29非空ranges；两个代表rank各48条十算子链、516条SWA三算子链与Qb/queryRoPE/SMLA一一关联，已有完整native CSV各35664行。结果：十算子链全免仅.427%–.432%窗口，独立旧小预算继续停止。SWA局部全免3.065%–3.080%，但固定Qb到达的join最多仅提前.648774%–.660879%窗口；不把局部3%当收益。保留RMS BF16与RoPE BF16边界、FP32三角表、原乘乘加及stride/skip；SK已融合地址计算。更大WKV/Q/input-ready资源与数学共享待确认，本轮准入0；窗口不是请求收益，全链路未扫全，每个替换原语逐位一致。'
NEXT='loop038 Run run-20261110T050354Z：沿CVLinearWrapper/实际层方法与量化守卫→TorchNPU npu_quant_matmul→CANN9.2 QuantBatchMatmulV3 tiling/epilogue→FP32-trig query RoPE→SMLA实际读者。7当前磁盘source pins/5匹配，CV13–189与CP/量化入口已读；CP local-query/global-cache metadata不同，不能盲共享全张量。继续较大Q/KV整段价格和精度/布局/资源。owned容器最近读到exited，同image/id；用docker cp stdout只读源码成功，不启动/干扰其它采集，不冒称当前live服务健康。'

def insert(path,heading,body,before_current=False):
 p=Path(path);s=p.read_text()
 if heading in s:return
 i=s.find('## 当前断点') if before_current else s.find('\n')+1
 assert i>=0;p.write_text(s[:i]+'\n'+heading+'\n\n'+body+'\n\n'+s[i:])

def main():
 f=json.loads((ROOT/'cache-epilogue-native-frontier.json').read_text())
 j=json.loads((ROOT/'swa-publication-join-price.json').read_text())
 join=dict(status=j['status'],price='swa-publication-join-price.json',report='SWA主辅流join与较大输入交付判断.md',
  profiles=[{k:v for k,v in p.items() if k!='binds'} for p in j['profiles']],
  classification='SWA-only fixed-Qb-arrival child not developed from local3% envelope; larger Q/KV resource parent pending, not a fullrequest stop',pending=j['limits'],whole_chain_complete=False)
 entry=json.loads((ROOT/'swa-query-producer-source-review.json').read_text())
 links='[原生与局部价格](audit/execution-search/原生缓存发布与SWA整段融合判断.md) · [join与较大交付](audit/execution-search/SWA主辅流join与较大输入交付判断.md)'
 body='**Goal ACTIVE；loop037 PIVOTED限定源码/价格；loop038 RUNNING。**\n\n- **介绍、到了哪里、结果**：'+SUMMARY+'\n- **证据**：'+links+'；cache-epilogue-native-{frontier,source-review}、cache-epilogue-balanced-sites、swa-publication-join-price。\n- **待确认**：实际RMS/DQ/QuantMatmul tiling和loadedbinary、逐原语位对拍、slot唯一/skip/stride代次、UB/flags和真实跨卡/完整请求曝光。已确认可实现本轮0；不再收独立小index预算的NPU参数。\n- **环境**：本轮远端只读source与旧CSV，NPU/模型/推理请求/新profile/显式远端写入/服务动作0。稍后读取owned容器见exited，同image/id；失败running guard保留，用docker cp stdout内存tar继续只读成功。上方其它采集流程和文件保持原样，旧恢复回执为历史，不声称当前live状态。\n- **下一条待扫**：'+NEXT+'\n- **公开同步**：限定路径回执另记；保留并发首页/采集记录。'
 insert('HANDOFF.md','## 当前断点 · '+TITLE,body,True)
 for n in ['M00','M01','M03','M11']:
  insert(f'audit/modules/{n}.md','## '+TITLE,SUMMARY+'\n\n'+NEXT+'\n\n[原生判断](../execution-search/原生缓存发布与SWA整段融合判断.md) · [join判断](../execution-search/SWA主辅流join与较大输入交付判断.md)。')
 insert(ROOT/'搜索范围与当前断点.md','## '+TITLE,SUMMARY+'\n\n'+NEXT+'\n\n[原生判断](原生缓存发布与SWA整段融合判断.md) · [join判断](SWA主辅流join与较大输入交付判断.md)。')
 insert('audit/code-led/当前断点.md','## '+TITLE,SUMMARY+'\n\n'+NEXT)
 insert('audit/团队开发清单.md','## '+TITLE,SUMMARY+'\n\n已停止：旧独立index尾段；固定Qb的SWA-only不以3%局部包络立项。待确认：更大Q/KV producer及资源竞争、原生逐位/代次/实际请求净价。已确认可实现：本轮0。\n\n'+NEXT)
 insert('ENVIRONMENT.md','## 2026-11-10 loop037/038只读源码与容器状态观察','前段39source pin/29审阅范围与EP0/8 savedCSV只读完成；随后running guard断言失败。只读docker inspect见owned容器wio-vllm-v41-repro171-20261102为exited，image ce42d65b…、id3be44e7b…匹配；不推断退出原因或当前整体服务健康。改为docker cp individual source到stdout tar并在内存读，7source/5 frozen匹配、无容器进程/解包/写入或启动。上方并行阶段采集记录保留，本扫描不干预。证据audit/execution-search/swa-query-producer-{entry,retrieval-attempts,source-review}.json。历史CAPTURED_RESTORED回执不等于此时live状态。')
 p=ROOT/'source-frontier.json';d=json.loads(p.read_text());d.update(status='ACTIVE_LOOP038_LARGER_QKV_PRODUCER_AFTER_SWA_JOIN',region='Actual native publication bounded; smaller index stopped; fixed Qb SWA exposure low, larger Q/KV native producer pending',next_action=NEXT,cache_epilogue_review=f,swa_publication_join_review=join,swa_query_producer_entry=entry)
 d['completeness_established']=False;p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 p=Path('audit/first-pass.json');d=json.loads(p.read_text());d['latest_execution_derived_followup']=dict(status='ACTIVE_NOT_COMPLETE',summary=SUMMARY,next_action=NEXT,source_frontier='audit/execution-search/source-frontier.json',native_frontier='audit/execution-search/cache-epilogue-native-frontier.json',join_price='audit/execution-search/swa-publication-join-price.json',whole_chain_complete=False,new_primitive_admissions=0);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 for name in ['171-推理数据流与已查节点.html','171-融合算子全盘排查进度图.html']:
  p=Path(name);s=p.read_text()
  if 'id="cache-publication-followup"' in s:continue
  banner='<aside id="cache-publication-followup" style="padding:18px 24px;border:1px solid #c9d9de;margin:18px;border-radius:8px;background:#f5f9fa;color:#203a45;font:14px/1.7 system-ui"><strong>当前覆盖：全链路未扫全 · 每个替换原语逐位一致</strong><p>'+SUMMARY+'</p><p>下一处：CVLinearWrapper → QuantBatchMatmulV3 → query RoPE → SMLA；精度、代次、资源和请求净价待确认。<a href="audit/execution-search/原生缓存发布与SWA整段融合判断.md">原生判断</a> · <a href="audit/execution-search/SWA主辅流join与较大输入交付判断.md">join判断</a> · <a href="HANDOFF.md">当前断点</a></p></aside>'
  s,n=re.subn(r'(<body[^>]*>)',lambda m:m[0]+banner,s,count=1)
  if n==0:s,n=re.subn(r'(<main[^>]*>)',lambda m:m[0]+banner,s,count=1)
  assert n==1;p.write_text(s)
 print(json.dumps(dict(status=d['latest_execution_derived_followup']['status'],whole_chain_complete=False,new_primitive_admissions=0)))
if __name__=='__main__':main()
