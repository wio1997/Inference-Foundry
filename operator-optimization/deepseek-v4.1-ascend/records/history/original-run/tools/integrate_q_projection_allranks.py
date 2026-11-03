# -*- coding: utf-8 -*-
from pathlib import Path
from fractions import Fraction
import datetime,hashlib,json
T=Path(__file__).resolve().parents[1]
def read(n):return json.loads((T/n).read_text())
def write(n,x):(T/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
s=read('q-projection-allranks/summary.json');assert s['rank_count']==16
x=next(x for x in s['groups'] if x['scope']=='CAPTURED_TARGET_INTERIOR' and x['m']==3)
assert x['calls']==1920 and x['rank_count']==16 and x['median_duration_us']==42.0
cap=Fraction(353894400000000); ideal_us=Fraction(3*41943040*1000000,1)/cap
ratio=Fraction(str(x['median_duration_us']))/ideal_us
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
e={'status':'CONDITIONAL_SAME_SHAPE_16RANK_SENSITIVITY','timestamp':now,'scope':'16 ranks, source-invariant captured target interior layer windows, matching stream/model ID; one profiled kernel sample rather than request throughput', 'm':3,'calls':x['calls'],'ranks':x['rank_count'],'median_task_us':x['median_duration_us'],'p05_task_us':x['p05_duration_us'],'p95_task_us':x['p95_duration_us'],'per_rank_median_us_range':[x['rank_median_us_min'],x['rank_median_us_max']], 'reported_cycle_weighted_mac_ratio_proxy':x['aic_cycle_weighted_mac_proxy'],'reported_cycle_weighted_mte2_ratio_proxy':x['aic_cycle_weighted_mte2_proxy'], 'conditional_dense_cube_compute_ideal_us':float(ideal_us),'conditional_observed_median_over_ideal':float(ratio),'assumptions':'Existing conditional nominal capacity per logical device, independent dense full-output algorithm and modeled Cube-only work. Same shape and source-bracket association do not prove every possible implementation belongs to the mathematical model.', 'not_proven':['Unconditional physical ceiling','Measured ratio equals MFU or full-chip utilization','MTE2 ratio establishes HBM saturation','Pipes sum as exclusive time','Profiler overhead negligible','Conditional gap is achievable acceleration','Whole-framework largest performance gap'], 'summary_sha256':hashlib.sha256((T/'q-projection-allranks/summary.json').read_bytes()).hexdigest(),'raw_source_hash_index':'evidence/TRACE-EXPORT-SHA256.json'}
write('evidence/q-projection-16rank-diagnosis.json',e)
a=read('module-analysis.json');a['supplementary_q_projection_16rank']={'path':'evidence/q-projection-16rank-diagnosis.json','scope':'captured target interior, matched model/stream windows','conditional_not_universal':True};write('module-analysis.json',a)
s2=read('STATE.json');s2.update(last_update=now,checkpoint_status='Q_PROJECTION_16RANK_REVIEWED',next_read_only_action='Resolve or document eager target and Engram attribution and independently review the G4 fusion graph; hardware bounds remain conditional.');write('STATE.json',s2)
with (T/'REPORT.md').open('a') as f:f.write('''\n\n## 16 rank 的同形状 INT8 矩阵算子\n\n用已验证的 target 层内部窗口、相同 model ID 和 stream 关联，在 16 个 rank 找到 1920 个 M=3、INT8×INT8 输入、BF16 输出的同形状矩阵算子样本。整体中位耗时 42.000 微秒，各 rank 的中位值范围约 41.090–43.259 微秒；分组原始聚合见 `q-projection-allranks/`。严格 draft 样本和不能归属的同形状调用分开保留。\n\n沿用先前明确列出的标称计算容量和独立 dense、全输出、Cube-only 条件，计算项的理想时间约 0.356 微秒，中位样本与该条件模型的比值约 118.1。它是采样算子与一个宽松条件模型的差距，不是可达到的加速比，也不是框架整体上界。该组 AIC MAC 和 MTE2 计数器代理约为 10.6% 和 72.9%；不同流水可以重叠，不能相加或单凭 MTE2 值判定 HBM 带宽已饱和。见 `evidence/q-projection-16rank-diagnosis.json`。\n''')
print(json.dumps({'status':'INTEGRATED','target_calls':x['calls'],'median_us':x['median_duration_us'],'conditional_ratio':float(ratio),'ranks':x['rank_count']}))
