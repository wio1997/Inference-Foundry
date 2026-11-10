from pathlib import Path
import json,statistics as st
root=Path(__file__).resolve().parent
analysis=json.loads((root/'balanced-analysis.json').read_text())
pmu=json.loads((root/'kernel-pmu-analysis.json').read_text())
receipt=json.loads((root/'receipt.json').read_text())
refs=[json.loads((root/f'balanced-reference-{i}.json').read_text()) for i in range(3)]
summary={'date':'2026-11-10 Asia/Shanghai','request_assignment':receipt['request_assignment'],'load':{'concurrency':8,'per_dp':4,'input_tokens_per_request':32768,'output_tokens_per_request':1024,'total_output_tokens':8192},'reference_rounds':[{'seconds':r['batch_seconds'],'tokens_per_second':r['output_tokens_per_second'],'checks':r['balance_checks']} for r in refs],'reference_median_seconds':st.median(r['batch_seconds'] for r in refs),'reference_median_tokens_per_second':st.median(r['output_tokens_per_second'] for r in refs),'phases':{},'kernel_pmu':{},'capture_windows':receipt['capture_windows'],'restoration':json.loads((root/'restored-smoke.json').read_text())}
for phase in ['early','steady']:
    summary['phases'][phase]={}
    for dp in [0,1]:
        rows=[r for r in analysis['files'] if r['phase']==phase and r['dp']==dp]
        summary['phases'][phase][f'dp{dp}']={'ranks':len(rows),'window_ms':st.median(r['device_window_ms'] for r in rows),'union_ms':{key:st.median(r['union_ms'].get(key,0) for r in rows) for key in ['gloo','free','compute','communication','event_sync','DispatchFFNCombine','SparseFlashMla','HcPre','HcPost','VllmAscendQuantLightningIndexerV2']},'pct':{key:st.median(r['device_window_pct'].get(key,0) for r in rows) for key in ['gloo','free','compute','communication','event_sync','DispatchFFNCombine','SparseFlashMla','HcPre','HcPost','VllmAscendQuantLightningIndexerV2']},'gloo_free_intersection_ms':st.median(r['gloo_and_free_ms'] for r in rows),'event_free_intersection_ms':st.median(r['event_and_free_ms'] for r in rows)}
for dp in [0,1]:
    files=[r for r in pmu['files'] if r['phase']=='steady' and r['dp']==dp]
    summary['kernel_pmu'][f'dp{dp}']={}
    for key in ['DFC_target_top6','DFC_draft_top3','QuantBatchMatmulV3','MatMulV2','SparseFlashMla','HcPre','VllmAscendQuantLightningIndexerV2']:
        rows=[next(r for r in f['operators'] if r['operator_role']==key) for f in files]
        summary['kernel_pmu'][f'dp{dp}'][key]={name:st.median(r[name] for r in rows) for name in ['wall_sum_ms','calls','call_median_us','aic_mac_active_pct','aic_mte2_active_pct']}
summary['twice_throughput_target']={'seconds':summary['reference_median_seconds']/2,'tokens_per_second':summary['reference_median_tokens_per_second']*2}
examples=next(r for r in analysis['files'] if r['phase']=='steady' and r['ep']==0)['host_device_examples']
summary['same_call_example']=next(r for r in examples if r['operator']=='SparseFlashMla' and r.get('host_flow_source'))
summary['limits']=['Inputs are the first four frozen prompts mirrored across DP0/DP1, changing the request mix relative to the previous eight distinct prompts. Do not attribute the old/new throughput difference solely to routing.','Early and steady windows are different execution phases. The old DP1-heavy trace was early, so its percentages are not a same-phase A/B comparator.','Per-device window fractions are per-rank then group medians. They are not full-request time fractions and cannot be added.','Free is the export overlap-analysis compute/communication gap; full attribution still requires producer/consumer linkage.','MAC/MTE2 activity percentages use measured per-task active time over measured AIC time; they are not achieved fractions of theoretical peak bandwidth or FLOPs.','HostToDevice start-to-kernel latency includes prior queued work and dependencies; it is not all avoidable host overhead.']
(root/'findings.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
a=summary['phases']['steady']['dp0'];b=summary['phases']['steady']['dp1'];e=summary['same_call_example']
report=f'''# 171 配平测试与框架/算子联合 profile

2026-11-10。用户授权后实际运行，原模型、算子源码、启动参数沿用冻结版本。临时实例仅增加采集和开发接口，结束后原服务恢复health200，小请求完成，serve/model/Attention源码SHA与冻结记录一致。

[全局性能图](../../171-框架Profile热点.html)覆盖全链路、16卡共同时间线、全部88类设备算子（steady84类）、shape与PMU、同次Host→Device关联。支持模块跳转、时间排序、搜索及CSV导出。灰色模块独立耗时及EngineCore内部仍待归因，不代表全模型每段已完成测量。离线复建使用build_global_view.py；交互校验见view-validation.json。

**配平通过。** 两组各显式路由4条请求，使用原冻结集合前4条输入的镜像。每条输入32,768、输出1,024，4对输出文本SHA一致。三轮参考及两轮profile均核对每DP实际4成功请求、输入131,072、输出4,096、前缀命中118,016（90.039%）、无抢占，并采到同时4+4 running。草稿接受计数存在小幅差异，保留原计数。

| 无profile参考轮次 | 整批秒 | 输出token/s |
|---|---:|---:|
'''
for i,r in enumerate(refs):
    report+=f'| {i+1} | {r["batch_seconds"]:.3f} | {r["output_tokens_per_second"]:.2f} |\n'
report+=f'''
中位{summary['reference_median_seconds']:.3f}s、{summary['reference_median_tokens_per_second']:.2f}输出token/s。两倍目标为{summary['twice_throughput_target']['seconds']:.3f}s、{summary['twice_throughput_target']['tokens_per_second']:.2f}输出token/s。本轮是诊断测试，尚无优化实现或两倍收益。

32个worker trace和2个前端trace已离线导出。采集分early与steady，每窗最多12个worker迭代。steady在8请求均出现首个可见文本后再等2秒开始，开始时两组均有4个运行请求。

| 稳态观测（各组8卡中位） | DP0 | DP1 | 含义 |
|---|---:|---:|---|
| 设备分析窗口 | {a['window_ms']:.2f}ms | {b['window_ms']:.2f}ms | 此窗口的分母 |
| CPU Gloo等待 | {a['union_ms']['gloo']:.2f}ms / {a['pct']['gloo']:.2f}% | {b['union_ms']['gloo']:.2f}ms / {b['pct']['gloo']:.2f}% | 稳态两组差异小，优先级下降 |
| 计算/通信空档Free | {a['union_ms']['free']:.2f}ms / {a['pct']['free']:.2f}% | {b['union_ms']['free']:.2f}ms / {b['pct']['free']:.2f}% | 两组共有的主要排查空间 |
| Free与Gloo交集 | {a['gloo_free_intersection_ms']:.2f}ms | {b['gloo_free_intersection_ms']:.2f}ms | 大部分空档需另找生产者 |
| DFC目标+草稿 | {a['union_ms']['DispatchFFNCombine']:.2f}ms / {a['pct']['DispatchFFNCombine']:.2f}% | {b['union_ms']['DispatchFFNCombine']:.2f}ms / {b['pct']['DispatchFFNCombine']:.2f}% | 最大设备算子单项 |
| Event同步等待 | {a['union_ms']['event_sync']:.2f}ms | {b['union_ms']['event_sync']:.2f}ms | 与Free交集仅{a['event_free_intersection_ms']:.2f}/{b['event_free_intersection_ms']:.2f}ms，多数期间设备仍工作 |

首段仍有暂态差异：本轮early大等待出现在DP0（Gloo约831ms），DP1约93ms；稳态变成14/20ms。旧首段DP1约982ms不能外推为固定DP组的稳态同步成本。相同并发与输入配平后，剩余重点是两组共有的空档及目标MoE。

**把同一次调用接起来。** 稳态EP0的SMLA通过HostToDevice flow定位，Task ID {e['task_args']['Task Id']}、stream {e['stream']}。CANN Node@launch调用{e['connection_api_candidates'][0]['dur']:.2f}µs，launch事件到kernel开始{e['submission_to_kernel_us']/1000:.3f}ms，kernel本体{e['kernel_us']:.2f}µs。排队差包含前序工作与依赖；优化需要追到对应生产者，不能把这24ms给每个kernel单独相加。该例实际展示框架提交和算子执行之间的区别。[逐调用关联](balanced-analysis.json)。

**空档中的Host活动。** EP0/EP8是代表卡，稳态tensor类CPU事件与Free交集约174.51/171.91ms，main-thread cpu_op交集约187.20/183.40ms。草稿Attention和MoE的Host包装各36次，其与Free交集仅约1.90/1.93ms、1.10/1.07ms。剩余时间不能笼统归因于草稿kernel；应继续追输入、metadata、张量索引/拷贝和工作队列。cpu_op墙钟时长包含底层调用，交集只说明同时发生。[代表卡区间](steady-host-gaps.json)。

| 稳态算子内部PMU（8卡中位） | DP0 | DP1 |
|---|---:|---:|
'''
for label,key,field,fmt in [('目标DFC累计','DFC_target_top6','wall_sum_ms','ms'),('草稿DFC累计','DFC_draft_top3','wall_sum_ms','ms'),('目标DFC单次中位','DFC_target_top6','call_median_us','µs'),('目标DFC MAC活跃比例','DFC_target_top6','aic_mac_active_pct','%'),('目标DFC MTE2活跃比例','DFC_target_top6','aic_mte2_active_pct','%')]:
    x=summary['kernel_pmu']['dp0'][key][field];y=summary['kernel_pmu']['dp1'][key][field];report+=f'| {label} | {x:.2f}{fmt} | {y:.2f}{fmt} |\n'
report+='''
目标/草稿DFC由Input Shapes中路由TopK6/3区分。目标单次M3约309µs，MTE2活动占比明显高于MAC，应优先核对权重读取、数据复用、小M效率和内部等待；需要有效字节及资源分析才能认定带宽上界。PMU活动比不是峰值带宽利用率，管线时长可重叠。[逐算子及shape证据](kernel-pmu-analysis.json)。

后续顺序：先把稳态Free逐段绑定到实际CPU生产者、提交与依赖，选一个可删的大范围；并行按代码核对目标DFC权重读取/复用与小M方案。每项用完整交付边界估净省，再回到相同配平负载测整批吞吐。保留同输入/精度/状态约束。当前无源码优化改动。

来源在launch.json记录的171自有目录。原始大trace留远端；公开摘要保留每卡每阶段数据与复算脚本。与旧测试相比，本轮请求集合为4条镜像，而旧为8条不同请求，吞吐差异不能单独归为DP路由收益。所有窗口百分比均不直接代表整请求占比。
'''
(root/'REPORT.md').write_text(report)
print('Balanced report and findings written.')
