#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reconcile real QKV/CANN value and next actual head-layout consumer entry."""
from pathlib import Path
import json,re
from analyze_qkv_native_frontier import SUMMARY

ROOT=Path('audit/execution-search')
TITLE='2026-11-10 QKV原生次序与head布局贯穿Attention：继续较大consumer'
NEXT='loop039 Run run-20261110T054436Z：GSize8全局query host/AICPU packet→arch22 QK/PV行/head地址、原MMAD/softmax→inverseRoPE→原TP O投影布局；全局selected-ID的96次producer收集/同代复用、common TP24资格、zero-query/draft20/ordinary回退与图collective顺序。8当前SMLA/CP源SHA匹配，两处head guard允许8只属入口条件，不是完整合法性/原语位准入。继续完整付成本范围，不把新短窗或35MiB读范围兑换成请求/容量收益。'
def insert(path,heading,body,before_current=False):
 p=Path(path);s=p.read_text()
 if heading in s:return
 i=s.find('## 当前断点') if before_current else s.find('\n')+1
 assert i>=0;p.write_text(s[:i]+'\n'+heading+'\n\n'+body+'\n\n'+s[i:])
def main():
 frontier=json.loads((ROOT/'qkv-native-frontier.json').read_text());entry=json.loads((ROOT/'attention-head-layout-entry-review.json').read_text())
 body='**Goal ACTIVE；loop038 PIVOTED限定原生/价格；loop039 RUNNING。**\n\n- **介绍、到了哪里、结果**：'+SUMMARY+'\n- **精度**：CPU有限公式反例不是NPU/模型结果；INT32安全仍须原BF16 scale和原反量化次序。运行Qb raw22/22不是tilingKey22；compiled option列表不证明activekey。\n- **证据**：[QKV与贯穿Attention判断](audit/execution-search/QKV原生反量化与贯穿Attention的head布局判断.md)、qkv-native-{source-review,frontier,catalog,dependencies,extra-dependencies,route-identity}、qkv-joint-delivery-price、qkv-dequant-order-cpu。\n- **状态**：待确认——贯穿SMLA/TP O投影的共同布局及精度/metadata/资源/请求净价；已确认可实现——本轮0；已停止——两个当前窗固定SWA到达的Qb-only，以及普遍scale重结合配方。不是停止任意资源重排/全部请求或更大parent。\n- **环境**：本轮171 hostname核对成功，同owned image/id且观察running；仅只读源码/配置/header与旧profile，本机标量公式反例，NPU/模型/推理请求/新profile/显式远端写入/服务动作0。上方并行设备空档采集原件/恢复回执保持原样，不将旧exited或running代替health证明。\n- **下一条待扫**：'+NEXT+' [8源入口](audit/execution-search/attention-head-layout-entry-review.json)。新Run054436Z仍PLANNED。\n- **公开同步**：限定路径回执另记，保留并发设备空档新报告。'
 insert('HANDOFF.md','## 当前断点 · '+TITLE,body,True)
 for n in ['M00','M01','M03','M11']:insert(f'audit/modules/{n}.md','## '+TITLE,SUMMARY+'\n\n'+NEXT+'\n\n[当前判断](../execution-search/QKV原生反量化与贯穿Attention的head布局判断.md)。')
 insert(ROOT/'搜索范围与当前断点.md','## '+TITLE,SUMMARY+'\n\n'+NEXT+'\n\n[当前判断](QKV原生反量化与贯穿Attention的head布局判断.md)。')
 insert('audit/code-led/当前断点.md','## '+TITLE,SUMMARY+'\n\n'+NEXT)
 insert('audit/团队开发清单.md','## '+TITLE,SUMMARY+'\n\n已确认可实现：本轮0；待确认：头布局保持到SMLA/O投影，需完整逐位/metadata/资源门槛；已停止：固定其它到达的Qb-only和普遍scale重结合。新范围与原SWA、QLI、SMLA候选有重叠，不另加收益。\n\n'+NEXT)
 insert('ENVIRONMENT.md','## 2026-11-10 loop038 QKV原生/量化元数据与039入口','171统一hostname核对成功，同owned容器/image/id且本轮见running；无health/模型请求或服务动作。当前7Python源SHA重核相同；CANN9.2的QuantBatchMatmulV3源/910_93 compiled JSON、text_config/quant_description和142个safetensors header/247相关projection metadata只读（payload0），本机CPU IEEE公式反例不当NPU bitgate。native Qb22/22不是tilingKey22；仍缺activekey/已加载binary。另8SMLA/CP源SHA匹配，head8只通过已读两处头数guard。并行scoped-profile的当前完成/恢复证据见其自有记录，不由本源码扫描产生或改动。')
 p=ROOT/'source-frontier.json';d=json.loads(p.read_text())
 compact_frontier={k:v for k,v in frontier.items() if k not in ['same_scope_prices','larger_layout_prices']}
 compact_frontier['full_price_scenarios']='qkv-joint-delivery-price.json;324 cost sensitivities and all1032 delivery records retained'
 d.update(status='ACTIVE_LOOP039_HEAD_LAYOUT_THROUGH_ATTENTION',region='QKV integer/dequant bounded; continue larger headlayout through SMLA and TP output projection',next_action=NEXT,qkv_native_review=compact_frontier,attention_head_layout_entry=entry,completeness_established=False);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 p=Path('audit/first-pass.json');d=json.loads(p.read_text());d['latest_execution_derived_followup']=dict(status='ACTIVE_NOT_COMPLETE',summary=SUMMARY,next_action=NEXT,source_frontier='audit/execution-search/source-frontier.json',native_frontier='audit/execution-search/qkv-native-frontier.json',whole_chain_complete=False,new_primitive_admissions=0);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 for name in ['171-推理数据流与已查节点.html','171-融合算子全盘排查进度图.html']:
  p=Path(name);s=p.read_text();banner='<aside id="qkv-native-followup" style="padding:18px 24px;border:1px solid #c9d9de;margin:18px;border-radius:8px;background:#f5f9fa;color:#203a45;font:14px/1.7 system-ui"><strong>当前覆盖：全链路未扫全 · 每个替换原语逐位一致</strong><p>'+SUMMARY+'</p><p>下一处：head8全局query metadata → SMLA原MMAD/softmax/PV → 原TP O投影。<a href="audit/execution-search/QKV原生反量化与贯穿Attention的head布局判断.md">范围与付成本判断</a> · <a href="HANDOFF.md">当前断点</a></p></aside>'
  if 'id="qkv-native-followup"' in s:s,n=re.subn(r'<aside id="qkv-native-followup".*?</aside>',lambda m:banner,s,count=1,flags=re.S)
  elif 'id="cache-publication-followup"' in s:s,n=re.subn(r'<aside id="cache-publication-followup".*?</aside>',lambda m:banner,s,count=1,flags=re.S)
  else:s,n=re.subn(r'(<main[^>]*>)',lambda m:m[0]+banner,s,count=1)
  assert n==1;p.write_text(s)
 print(json.dumps(dict(status='ACTIVE_LOOP039',whole_chain_complete=False,new_primitive_admissions=0)))
if __name__=='__main__':main()
