#!/usr/bin/env python3
"""Update only owned maps, preserving concurrent capture handoff and artifacts."""
from pathlib import Path
import json
OUT=Path('audit/execution-search')
SUMMARY='介绍：selected IDs/可见长度→原V0物理pair/半区压紧→三槽GM panel→QK/PV真实L1交付。到了哪里：11实际source/限定ranges，333原V0 CPUfixture/46184行地址；12旧nativeQLI×PA情景与1239552活跃NZ坐标；54旧价复现、108联合与20指令成本情景。结果：910C ND2NZ srcDValue为uint16元素，物理行距128会溢出，保留64位base并按原顺序拆分；朴素逻辑序104例不符，7例活跃洞回退。发现两块L1保留末128行、第三块work的较大联合方向，需跨QK(i)/PV(i-1)任务及新flags。K512/.75参考/h5us/12B factor1、全尾合资格时最有利请求条件1.6663%，非P90/实测，负值保留；高指令数可转负。原生ND2NZ padding/LoadData/MMAD逐位、真实K/页代次/分组、预取与净价仍待确认。停止逻辑序直替、A5宽stride套910C、原7块全驻K512与旧stockGM prepack的具体配方。本轮新增原语0，全链路未扫全，每个替换原语逐位一致。'
NEXT='loop036 Run run-20261110T032750Z：Engram紧凑scale checkpoint→host/sharedUVA注册owner→device pointer bank→lookup/dequant→末读/close/reload。3当前source SHA匹配；已读storage/scale类型分配、I8与MXFP8加载分支和lookup入口。继续SharedUvaBuffer注册/失败回滚/各rank close、成对code/scale bank寿命和图稳定地址；保留原INT8×FP32→BF16与255 NaN位合同。只为会改变18.4325GB容量/大交付净价去留的参数补证；不干扰进行中的配平profile控制器。'
def prepend(path,title,body,keep_capture=False):
 p=Path(path);s=p.read_text();assert title not in s
 i=s.find('## 当前断点') if keep_capture else s.find('\n')+1
 if i<0:i=s.find('\n')+1
 p.write_text(s[:i]+'\n'+title+'\n\n'+body+'\n\n'+s[i:])
def main():
 identity=json.loads((OUT/'engram-storage-identity.json').read_text());reviewed={'embedding.py':[[54,80],[140,169],[266,331]],'npu.py':[[14,27]],'engram_lookup.py':[[1,120]]};sources=[]
 for r in identity['sources']:
  p=Path(r['local_path']);sources.append(dict(runtime_path=r['runtime_path'],path=str(p),sha256=r['runtime_sha256'],reviewed_ranges=reviewed[p.name],scope='Entry ranges only. Registration/paired loading/close/reload and native exactness/value not completed.'))
 entry=dict(status='ENGRAM_COMPACT_STORAGE_ENTRY_ONLY',whole_chain_complete=False,every_replacement_bitwise=True,sources=sources,earlier_CPU_scale_gate='engram-scale-representation-cpu.json',capacity_reference_bytes=18432548400,current_findings=['INT8 storage currently allocatesFP32 scale; shared/localUVA shapes derive dtype. Newcompact must independently choose scale representation.','I8 load copies checkpoint scale intoFP32 without asserting E8M0; compact path must guard actualscale dtype/shape and preserve non-E8M0 fallback. MXFP8/generalfloat load branches can requantize and are separate legal paths.','Current UVA lookup chooses scale pointer type by MXFP8 code-type flag. NewINT8+E8M0 requires independent decoder selector, not enablingMXFP8 (which would also change code pointer type).'],next_action=NEXT,new_NPU_work=0,remote_writes=[])
 (OUT/'engram-storage-entry.json').write_text(json.dumps(entry,ensure_ascii=False,indent=2)+'\n')
 body='**Goal ACTIVE；loop035 PIVOTED（限定source/CPU/条件价格）；loop036 RUNNING。全链路未扫全，每个替换原语逐位一致。**\n\n- **介绍、到了哪里、结果**：'+SUMMARY+'\n- **证据**：[本轮判断](audit/execution-search/SMLA原序直达L1与尾块复用判断.md)、smla-direct-l1-{frontier,source-review,price,exploration-notes}、smla-transfer-plan-cpu-a2与smla-l1-layout-cpu-a1。a1本机mock编译失败保留；所有CPU PASS均不当NPU bit gate。\n- **环境**：owned171容器只读source/安装header；本机CPU和旧数据复算。此扫描无NPU/模型/推理请求/新profile/显式远端写入/服务更改。上方另一个已授权配平profile控制器及其文件保留，不声称全机无并行工作。\n- **下一条待扫**：'+NEXT+' [入口](audit/execution-search/engram-storage-entry.json)；Run仍PLANNED，不当已完成。\n- **公开同步**：本轮回执另记；此前c589af1仅覆盖loop034与035入口。'
 prepend('HANDOFF.md','## 当前断点 · 2026-11-10：SMLA原序L1与尾块已判，转Engram注册交付',body,True)
 for n in ['M00','M01','M03','M11']:prepend(f'audit/modules/{n}.md','## 2026-11-10 原序L1、尾块复用与Engram下一交付',SUMMARY+'\n\n'+NEXT+' 见[判断](../execution-search/SMLA原序直达L1与尾块复用判断.md)。')
 prepend(OUT/'搜索范围与当前断点.md','## 2026-11-10 原序直达L1与尾块联合前沿',SUMMARY+'\n\n'+NEXT+' [判断](SMLA原序直达L1与尾块复用判断.md)。')
 prepend('audit/code-led/当前断点.md','## 2026-11-10 SMLA交付已判，Engram成对注册入口',SUMMARY+'\n\n'+NEXT)
 prepend('audit/团队开发清单.md','## 2026-11-10 directL1与尾块联合待确认',SUMMARY+'\n\n待确认：先按实际K/原序分组和字段资格筛包，保留旧fallback；原语位门槛与新flags/预取净价决定开发准入。\n\n'+NEXT)
 prepend('ENVIRONMENT.md','## 2026-11-10 SMLA L1字段与CPU地址核实（loop035）','171统一入口hostname及owned容器/镜像已核对；4安装CANN9.2.0头文件和7模型/SMLA源匹配。910C2201 Nd2NzParams的srcDValue为uint16元素，A53510另为uint64，不能混用；本机仅抽取struct编译，不是设备测试。333原V0地址fixtures、12旧QLI PA情景和活跃NZ坐标CPU检查见[本轮报告](audit/execution-search/SMLA原序直达L1与尾块复用判断.md)。尾pin使用原KV三块中的两块，异步事件/原生位值尚未验收。此扫描只读远端source，所有remotePython -B -I -S，无NPU/模型/profile/request/服务动作或显式远端写入；HANDOFF另一个已授权配平profile控制器及环境变化由其状态文件说明，本扫描不操作它。Engram下一入口3当前SHA匹配，尚非成对加载/注册/图地址/末读完成。')
 p=OUT/'source-frontier.json';d=json.loads(p.read_text());front=json.loads((OUT/'smla-direct-l1-frontier.json').read_text());d.update(status='ACTIVE_LOOP036_ENGRAM_STORAGE_OWNER_ENTRY',region='Bounded selectedKV original-order L1 layout and joint tail price; next Engram paired storage owner lifetime',next_action=NEXT,smla_direct_l1_review=dict(status=front['status'],report=front['report'],frontier='smla-direct-l1-frontier.json',source_review=front['source_review'],source_pins=11,CPU_original_V0=front['CPU_original_V0'],CPU_fields_NZ=front['CPU_fields_NZ'],price=front['price'],confirmed_implementable=[],stopped=front['stopped'],pending=front['pending'],whole_chain_complete=False),engram_storage_entry=entry)
 d['smla_direct_kv_entry']['status']='HISTORICAL_ENTRY_SUPERSEDED_BY_LOOP035_BOUNDED_REVIEW';d['SMLA_review']['latest_direct_l1_review']='smla-direct-l1-frontier.json';d['Engram_review']['next_storage_owner_entry']='engram-storage-entry.json'
 for f in d['frontiers']:
  if f['region'].startswith('SMLA internal'):f.update(state='ORIGINAL_ORDER_L1_LAYOUT_CPU_PROVENANCE_AND_JOINT_PRICE_BOUNDED',remaining='Actual primitive ND2NZ/padding/LoadData/MMAD bitwise; trueK/page/group distribution, role/event schedule and paid prefetch; other legal branches remain open.')
 p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(updated_maps=10,next_source_pins=3,whole_chain_complete=False,new_primitive_admissions=0)))
if __name__=='__main__':main()
