#!/usr/bin/env python3
"""Reconcile actual SMLA packet coverage, branch correction and paid plan frontier."""
from pathlib import Path
import json

OUT=Path('audit/execution-search')
SUMMARY='介绍：沿长度/掩码/平台/确定性→AICPU分核包→arch22 FA/FD与QK/PV实际读者，判selected-KV大交付。到了哪里：21实际源pin/18限定ranges；原标量323–1631的7个本机CPUfixture；1840原C8 native/2560窗/30typed组，6旧值精确复现、54付plan/协调成本情景。结果：packet已按batch ratio共享，900词已定义/124尾词未写；默认整行，q3 level3可生成6FA/3FD/48vec。更正旧“未传FLASH_DECODE”原因：入口会传，但host/key选择FD0；未继续消费的是BATCH_CONSISTENCY算术。停止flag-only安全跨S2和重复38层metadata信用；不是服务故障/NPU精度结论。12B计划/.75稀疏参考/h5us，K256条件最有利请求.727%–.875%，K512为1.101%–1.395%；后者继续深入。13负请求信用保留，均非实测/P90。新增原语准入0，全链路未扫全，每替换原语逐位一致。'
NEXT='loop035 Run run-20261110T023853Z：原序selected-KV64位地址+transfer计划→Cube QK/PV精确L1布局。已读Vector pair/large-stride fallback、Cube panel ND2NZ和DataCopyPA片段；继续vendor Nd2NzParams字段/stride/padding、PV布局/LoadData/flags、原ring与源/page/index代次，先找匹配旧搬运数据，再只补影响去留的门槛。保留原metadata/逻辑TopK及所有MMA/softmax/FP16-BF16边界。'

def prepend(path,title,body):
 p=Path(path);s=p.read_text();assert title not in s;i=s.find('\n');p.write_text(s[:i+1]+'\n'+title+'\n\n'+body+'\n'+s[i+1:])

def main():
 prepend('HANDOFF.md','## 当前断点 · 2026-11-10：SMLA分核包已判，继续selected-KV到L1', '**Goal ACTIVE；loop034 PIVOTED限定source/CPU控制与条件价格，loop035 RUNNING；全链路未扫全，每替换原语逐位一致。**\n\n- **介绍、到了哪里、结果**：'+SUMMARY+'\n- **关键边界**：AICPU无selectedID/page-table输入，metadata512不能证明真实K512；CTA已用3-ring，host也是3，末尾死offset加4不是第四bank可删。packet早读不退役循环仍读的length/TopK/page-table。CPU900词相同不代表1024词/NPU原语逐位通过。\n- **证据**：[判断](audit/execution-search/SMLA分核包与原顺序交付判断.md)、smla-schedule-{frontier,source-review,dependencies,catalog,exploration-notes}、smla-schedule-cpu-a1/{result,driver,stdout}、smla-address-plan-paid-price。原报告错误原因以本轮更正为准，原Run不改。\n- **环境**：171 hostname及owned容器/镜像核对成功；仅只读源/catalog，remote python -B -I -S；本机clang++标量fixture，NPU/模型/请求/新profile/显式远端写入/服务变化0，不操作其它controller。\n- **下一条待扫**：'+NEXT+' [入口](audit/execution-search/smla-direct-kv-entry.json)。新Run PLANNED不作已完成。\n- **公开同步**：本段回执另记；此前d65d1e0只覆盖workspace与metadata入口。')
 for n in ['M00','M03','M11']:
  prepend(f'audit/modules/{n}.md','## 2026-11-10 SMLA分核包实际消费者与付成本原序计划',SUMMARY+'\n\n'+NEXT+' 见[报告](../execution-search/SMLA分核包与原顺序交付判断.md)。')
 prepend(OUT/'搜索范围与当前断点.md','## 2026-11-10 SMLA分核包源级资格与原序计划净价',SUMMARY+'\n\n'+NEXT+'\n\n[报告](SMLA分核包与原顺序交付判断.md)。')
 prepend('audit/code-led/当前断点.md','## 2026-11-10 SMLA分核包到原序L1前沿',SUMMARY+'\n\n'+NEXT+'\n\n[当前判断](../execution-search/SMLA分核包与原顺序交付判断.md)。')
 prepend('audit/团队开发清单.md','## 2026-11-10 SMLA原序计划继续深入',SUMMARY+'\n\n待确认开发：保留原metadata，在QLI或首真实reader发布原实际顺序地址/transfer计划，给同源代次QK/PV使用；必须付计划/随机DMA/协调成本并过完整原语位门槛，不直接删softmax/BF16边界。\n\n'+NEXT+' [报告](execution-search/SMLA分核包与原顺序交付判断.md)。')
 prepend('ENVIRONMENT.md','## 2026-11-10 SMLA metadata原生源与本机CPU核实（loop034）','171统一hostname核对成功，同owned容器/镜像；21source pins，新增13依赖9原SHA匹配/4实际op-api新取。平台核数/Soc与deterministic level3是ACLNN代码中的运行时输入，未从服务对象读取实时值。仅本机clang++编译原标量排程323–1631并用mock Tensor/entry ABI运行7fixture，不是NPU数值/性能验收。CPU q3 level3给6FA/3FD/48vec，默认3FA；不得据此声称线上运行level3或故障。source packet常量36/72非实际launch；当前host/key固定FD0事实保留，纠正先前“不向SMLAType传FLASH_DECODE”的错误原因。源、fixture与范围见[smla-schedule-source-review](audit/execution-search/smla-schedule-source-review.json)和[CPU结果](audit/execution-search/smla-schedule-cpu-a1/result.json)。新NPU/模型/请求/profile/显式远端写入/服务变化均0，其它controller未动。')
 p=OUT/'source-frontier.json';d=json.loads(p.read_text());front=json.loads((OUT/'smla-schedule-frontier.json').read_text());entry=json.loads((OUT/'smla-direct-kv-entry.json').read_text())
 d.update(status='ACTIVE_LOOP035_ORIGINAL_ORDER_SELECTED_KV_TO_L1_ENTRY',region='Actual SMLA metadata scheduling/FA reader boundary and paid plan bounded; deepen original-order selectedKV QK/PV L1 delivery',next_action=NEXT,smla_schedule_review=dict(status=front['status'],report=front['report'],frontier='smla-schedule-frontier.json',source_review=front['source_review'],source_pins=21,reviewed_nonempty=18,CPU_fixture=front['CPU_fixture'],paid_parent=front['paid_parent'],correction=front['correction'],stopped=front['stopped'],pending=front['pending'],confirmed_implementable=[],whole_chain_complete=False),smla_direct_kv_entry=entry)
 d['SMLA_review']['correction']=front['correction'];d['SMLA_review']['latest_native_metadata_review']='smla-schedule-frontier.json';p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(updated_maps=9,sources=21,new_primitive_admissions=0),ensure_ascii=False))

if __name__=='__main__':main()
