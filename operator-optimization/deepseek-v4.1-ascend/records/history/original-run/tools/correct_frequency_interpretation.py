# -*- coding: utf-8 -*-
from pathlib import Path
import datetime,hashlib,json
T=Path(__file__).resolve().parents[1]
def load(p):return json.loads((T/p).read_text())
def save(p,x):(T/p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
for p,nested in [('evidence/conditional-nominal-capacity.json',False),('evidence/conditional-performance-comparisons.json',True)]:
 x=load(p);cap=x['capacity'] if nested else x
 assert cap['observed_frequency_MHz']=='1800';cap['frequency_field_interpretation']='Value recorded in profiler device metadata; not established as runtime time-series frequency or as a maximum physical frequency.';cap['metadata_frequency_is_live_measurement_proven']=False
 cap['conditions']=[v.replace('observed1800MHz alone','profiler-metadata 1800MHz alone') for v in cap['conditions']];save(p,x)
p=T/'evidence/IDEAL-MODEL.md';s=p.read_text();s=s.replace('观测到 1800 MHz 不证明它是物理最高频率','profiler 元数据记载 1800 MHz；尚未证明该字段是运行时实时频率，也不证明它是物理最高频率');p.write_text(s)
p=T/'REPORT.md';s=p.read_text();s=s.replace('官方标称公式与实际设备信息给出条件分母','官方标称公式与 profiler 设备元数据给出条件分母');p.write_text(s)
o={'timestamp':now,'status':'FREQUENCY_FIELD_NOT_PROVEN_LIVE_OR_HARD_MAX','profiler_metadata_frequency_mhz':1800,'interpretation':'The saved info.json value is a profiler metadata field, and no source here proves it is a live frequency measurement or the highest possible frequency for 9382.','primary_sources':[{'url':'https://mindstudio-profiler-docs.readthedocs.io/zh-cn/latest/profiling-cases/operator_mfu_guide/operator_mfu_guide/','claim':'The MFU tool combines device info ai_core_num and aic_frequency with dtype to estimate theoretical peak; no rigorous 9382 physical ceiling is established.','retrieved_date':now[:10]},{'url':'https://www.hiascend.com/doc_center/source/zh/mindx-dl/60rc2/faultdiag/fdug/MindX%20DL%206.0.RC2%20%E6%95%85%E9%9A%9C%E8%AF%8A%E6%96%AD%E7%94%A8%E6%88%B7%E6%8C%87%E5%8D%97.pdf','claim':'This product documentation distinguishes NPU nominal Aicore Freq and current Aicore curFreq; these are npu-smi fields and their mapping to profiler aic_frequency on 9382 is not proven.','retrieved_date':now[:10]},{'path':'sources/primary/msprof-analyze/chip_peak_flops.py','sha256':hashlib.sha256((T/'sources/primary/msprof-analyze/chip_peak_flops.py').read_bytes()).hexdigest(),'claim':'Versioned frozen estimator formula; no 9382 physical ceiling certificate.'}],'unresolved':['Actual per-time frequency and frequency limits for this board and execution context','Applicability and hard maximum of per-cycle nominal INT8 operations on 9382','Mixed-resource operations and necessary mathematical work'],'no_npu_work':True}
save('evidence/frequency-field-review.json',o)
s=load('STATE.json');s.update(last_update=now,checkpoint_status='FREQUENCY_INTERPRETATION_CORRECTED');save('STATE.json',s)
with (T/'REPORT.md').open('a') as f:f.write('''\n\n## 频率证据修正\n\n前述 1800 MHz 是 profiler 的设备元数据字段；尚未证明它是在基线运行时逐时采样的实际频率，也未证明它是 9382 的物理最高频率。华为文档说明其 MFU 工具使用设备信息和数据类型估算理论峰值；另一份华为文档区分标称频率与当前频率，但不能据此直接确定本设备 profiler 字段的对应关系。有关来源和适用范围见 `evidence/frequency-field-review.json`。现有数值仍是有条件模型，不是不可超越的已证物理上界。\n''')
print(json.dumps({'status':o['status'],'profiler_metadata_frequency_mhz':1800,'hard_cap':'UNKNOWN'}))
