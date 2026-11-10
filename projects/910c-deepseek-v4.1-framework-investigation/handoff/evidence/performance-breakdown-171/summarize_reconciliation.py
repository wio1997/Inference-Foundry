"""Generate an additive Computing activity tree and the ranked-time bridge."""
from pathlib import Path
from collections import defaultdict
import csv,json,statistics as st

P=Path(__file__).resolve().parent
rows=json.loads((P/'computing-reconciliation.json').read_text())['ranks']
detail=json.loads((P/'findings.json').read_text())
shown={'DFC_target_top6','QuantBatchMatmulV3','MatMulV2','HcPre','VllmAscendQuantLightningIndexerV2','TransposeBatchMatMul','SparseFlashMla','ScatterNdUpdateSk','RmsNorm','ForeachAddList','MatMulV3'}
def aggregate(rs,group):
    result={'rank_count':len(rs),'mean_window_ms':st.mean(r['window_ms'] for r in rs)}
    for key in ['sum_type_ms','union_all_type_ms','multi_type_extra_count_ms','union_outside_computing_ms','computing_not_covered_by_csv_ms','computing_ms']:
        result[key]=st.mean(r[key] for r in rs)
        result[key.replace('_ms','_pct')]=st.mean(r[key]/r['window_ms']*100 for r in rs)
    single=defaultdict(float);parallel=defaultdict(float)
    for r in rs:
        for row in r['computing_partition']:
            key=tuple(row['types']);target=single if row['kind']=='single' else parallel
            target[key]+=row['computing_ms']/len(rs)
    result['single']=[{'types':list(k),'mean_ms':v} for k,v in sorted(single.items(),key=lambda p:-p[1])]
    result['parallel']=[{'types':list(k),'mean_ms':v} for k,v in sorted(parallel.items(),key=lambda p:-p[1])]
    result['single_total_ms']=sum(single.values());result['parallel_total_ms']=sum(parallel.values())
    result['unmatched_total_ms']=result['computing_not_covered_by_csv_ms']
    assert abs(result['single_total_ms']+result['parallel_total_ms']+result['unmatched_total_ms']-result['computing_ms'])<1e-5
    ops=[r for r in detail[group]['operators'] if r['kind']=='operator']
    result['previously_shown_11_pct']=sum(r['mean_window_pct'] for r in ops if r['name'] in shown)
    result['remaining_68_pct']=sum(r['mean_window_pct'] for r in ops if r['name'] not in shown)
    result['notes']=['Single and parallel rows are disjoint pieces within CANN Computing. All rows add to Computing.',
                     'A parallel interval is counted once, with all active type names preserved. It is not assigned arbitrarily to one operator.',
                     'The sum of per-type ranked times counts parallel regions multiple times. It is a different metric.',
                     'Exclusive activity time is an accounting allocation, not a predicted optimization saving or critical-path gain.']
    return result
out={g:aggregate([r for r in rows if g=='all' or r['ep']//8==int(g[-1])],g) for g in ['all','dp0','dp1']}
(P/'reconciliation-summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
with (P/'computing-time-tree.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f,lineterminator='\n');w.writerow(['parent','category','types','mean_ms_per_card'])
    for kind in ['single','parallel']:
        for r in out['all'][kind]:w.writerow(['Computing',kind,' + '.join(r['types']),r['mean_ms']])
    w.writerow(['Computing','unmatched','Computing without CSV match',out['all']['unmatched_total_ms']])
a=out['all']
report=f'''# 算子为何没有直接加到54%：已经逐区间核准

统一使用同一次16卡、每卡11个闭合循环的设备窗口。之前展示的头部11项仅占{a['previously_shown_11_pct']:.2f}%；其余68类占{a['remaining_68_pct']:.2f}%。两者相加是所有79类各自任务区间的累计，不是互斥时间账。

| 从排行榜回到设备总账 | 平均ms/卡 | 占设备窗口 |
|---|---:|---:|
| 79类各自的区间时长相加 | {a['sum_type_ms']:.5f} | {a['sum_type_pct']:.5f}% |
| 减：不同类型并行造成的重复计时 | -{a['multi_type_extra_count_ms']:.5f} | -{a['multi_type_extra_count_pct']:.5f}% |
| 全部79类任务的时间并集 | {a['union_all_type_ms']:.5f} | {a['union_all_type_pct']:.5f}% |
| 减：并集在CANN Computing之外的边界部分 | -{a['union_outside_computing_ms']:.5f} | -{a['union_outside_computing_pct']:.5f}% |
| 加：Computing中未被CSV任务覆盖的边界部分 | {a['computing_not_covered_by_csv_ms']:.5f} | {a['computing_not_covered_by_csv_pct']:.5f}% |
| CANN Computing | **{a['computing_ms']:.5f}** | **{a['computing_pct']:.5f}%** |

前后边界抵消后的差额仅{a['union_all_type_ms']-a['computing_ms']:.7f}ms，不是新的性能瓶颈。主要区别是重复计算并行。

## 可加总的计算活动树

```text
Computing {a['computing_ms']:.5f}ms
├─ 只有一个算子类型活跃的区间 {a['single_total_ms']:.5f}ms
├─ 多个算子类型并行的共享区间 {a['parallel_total_ms']:.5f}ms
└─ CSV未匹配边界区间 {a['unmatched_total_ms']:.5f}ms
```

子项互斥、可直接相加。全部单算子项和并行组合见[computing-time-tree.csv](computing-time-tree.csv)。并行组合保留所有算子名，不随意把共享时长分给其中一个。三项及以上同时活跃时，重复计时可以大于该并行区域自身的时长。

这棵树用于时间对账，不能把独占时长直接当作优化收益，也不能说并行项没有优化价值；净收益仍要沿依赖与资源重排判断。Host各阶段另建同线程父子树，CPU与设备按同一时间轴对齐，不把跨线程/跨卡时间直接相加。
'''
(P/'RECONCILIATION.md').write_text(report)
print('Reconciled Computing',round(a['computing_ms'],5),'ms; single',round(a['single_total_ms'],5),'parallel',round(a['parallel_total_ms'],5))
