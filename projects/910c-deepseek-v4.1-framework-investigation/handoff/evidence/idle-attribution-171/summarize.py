"""Build a human/AI summary from original timing links and the scoped capture."""
from pathlib import Path
import json,statistics as st

P=Path(__file__).resolve().parent
load=lambda name:json.loads((P/name).read_text())
scope=load('scoped-capture/scope-analysis.json')['ranks']
original=load('submission-summary.json')
reference=load('scoped-capture/balanced-reference-0.json')
profiled=load('scoped-capture/balanced-steady.json')
restored=load('scoped-capture/restored-smoke.json')
boundaries=load('scoped-capture/prepare-boundary-analysis.json')
descriptions={
 'prepare input':('准备下一轮输入','更新请求状态、token位置和长度、缓存映射；协调batch并构造各缓存组的注意力metadata。','model_runner_v1.py:2287–2530；_prepare_inputs:1281；_build_attention_metadata:3449'),
 'forward':('目标整图启动','将已准备的输入交给目标整图；Host范围与实际整图设备执行范围分开计量。','model_runner_v1.py:2553–2598'),
 'post process':('模型读出与结果整理','整理目标输出、选择采样位置和读出所需数据。','model_runner_v1.py:2610–2678'),
 'sample_token':('校验与采样','校验草稿并确定本轮接受的token。','model_runner_v1.py:2812–2813'),
 'draft_token':('生成下一轮草稿','准备DSpark输入、context/slot/metadata，运行草稿并交付后续候选。','model_runner_v1.py:2890–2908；llm_base_proposer.py:1013–1398；dspark_proposer.py:226–357'),
 'outside_recorded_worker_scopes':('其余未归类','没有落入上述worker阶段的空档；不直接归为EngineCore调度。','EngineCore自身未直接采集')}

def summarize(rows,window):
    values=[r['ledgers'][window] for r in rows]
    result={'ranks':len(rows),'window':window,
            'mean_device_window_ms':st.mean(r['device_window_ms'] for r in values),
            'mean_free_ms':st.mean(r['free_ms'] for r in values),
            'mean_free_pct_of_device':st.mean(r['free_pct_of_device'] for r in values),'rows':[]}
    for key,(label,work,code) in descriptions.items():
        pct=st.mean(r['exclusive_pct_of_free'].get(key,0) for r in values)
        time=st.mean(r['exclusive_free_ms'].get(key,0) for r in values)
        result['rows'].append({'key':key,'label':label,'work':work,'code':code,
            'mean_pct_of_free':pct,'mean_free_ms':time,
            'mean_pct_of_device':st.mean(r['exclusive_free_ms'].get(key,0)/r['device_window_ms']*100 for r in values)})
    assert abs(sum(r['mean_pct_of_free'] for r in result['rows'])-100)<1e-6
    result['rows'].sort(key=lambda r:-r['mean_pct_of_free'])
    return result

out={'date':'2026-11-10 Asia/Shanghai','status':'MEASURED_SCOPES_AND_RESTORED',
     'scope':'one new targeted steady capture; 16 ranks; 11 closed target-graph cycles each',
     'scoped':summarize(scope,'closed_cycles'),'full_window':summarize(scope,'all'),
     'groups':{'dp'+str(dp):summarize([r for r in scope if r['dp']==dp],'closed_cycles') for dp in [0,1]},
     'original_submission':original,'prepare_boundaries':[],
     'test':{'reference_seconds':reference['batch_seconds'],'reference_output_tokens_s':reference['output_tokens_per_second'],
             'profile_seconds':profiled['batch_seconds'],'profile_output_tokens_s':profiled['output_tokens_per_second'],
             'reference_checks':reference['balance_checks'],'profile_checks':profiled['balance_checks'],
             'restored':restored},
     'priorities':[
       {'title':'先查输入/metadata的共同构造','reason':'约96%的空档发生在prepare input。代表卡的约31ms协调后参数构造窗口贡献约27ms空档，并出现243次where。先消除兼容同代视图/长度/掩码的重复构造，复用缓冲并合并交付。','code':'AscendDSAV41CPMetadataBuilder.build → _build_local_token_metadata；model_runner_v1._build_attention_metadata','status':'待确认；热点已测到，具体替换及净收益未验收'},
       {'title':'将Host准备和设备融合一起计价','reason':'现有slot mapping已经有fused-groups路径；新增方案应减少剩余Host派发与输入生产，而不是重复实现已有融合。保留所有缓存组、精度、同步和代次合同。','code':'ascend-block_table.py:469；metadata-host / BatchFrame既有方向','status':'待确认；复用历史，不新增重复信用'},
       {'title':'草稿和DP等待按本次实际预算排序','reason':'草稿Host阶段仅约2%的空档；DP协调和事件等待中有很多时间与设备工作重叠。优先完成输入准备大段的净价判断。','code':'draft_token；_sync_metadata_across_dp','status':'当前优先级降低；不等于这些阶段完全无优化空间'}],
     'limits':[
       'Scope overlap identifies which worker phase is active, not a proof that every instruction in it causes idle or can be removed.',
       'Closed cycles use first MODEL_EXECUTE through last MODEL_EXECUTE, excluding incomplete trace edges; full-window result is also retained.',
       'EngineCore scheduler process is not directly profiled. Uncovered idle is not its CPU execution time or a universal upper bound.',
       'The 31ms subwindow uses DP all_reduce end and first native metadata wrapper start. It is not an exact Python function boundary.',
       'The legacy CP metadata builder is a source-supported lead. No full worker Python stack is available to bind each where to one function.',
       'The profiler adds overhead. No optimization or 2x throughput improvement was implemented; do not extrapolate window ratios to all 16 seconds.']}
for row in boundaries:
    records=row['records'];parts=[]
    for i in range(3):
        values=[r['parts'][i] for r in records]
        parts.append({'key':values[0]['key'],'mean_wall_ms':st.mean(r['wall_ms'] for r in values),
                      'mean_free_ms':st.mean(r['free_ms'] for r in values),
                      'where_calls_per_prepare':st.mean(r['where_calls'] for r in values)})
    out['prepare_boundaries'].append({'ep':row['ep'],'calls':len(records),
        'mean_prepare_wall_ms':st.mean(r['wall_ms'] for r in records),
        'mean_prepare_free_ms':st.mean(r['free_ms'] for r in records),'parts':parts})
(P/'findings.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))

table='\n'.join('| '+r['label']+' | '+f'{r["mean_pct_of_free"]:.2f}%'+' | '+f'{r["mean_pct_of_device"]:.2f}%'+' | '+r['work']+' |' for r in out['scoped']['rows'])
report=f'''# 171设备空档：主要发生在准备下一轮输入

**结论：补采的16卡稳态闭合循环中，约96%的设备空档与worker的“prepare input”阶段重叠。** 它主要包含输入和注意力参数准备，不能笼统称为“调度器在挑请求”。草稿阶段约2%，其余很少。原服务已经恢复health200，8-token小请求成功，冻结源码SHA不变。

本轮只补一个带框架原生阶段标签的steady窗口，沿用DP各4条镜像请求、每条32768输入/1024输出。参考和profile两轮的10项配平检查全部通过，8条输出SHA也与此前参考完全相同。参考{reference['batch_seconds']:.3f}秒、{reference['output_tokens_per_second']:.2f}输出token/s；采集轮{profiled['batch_seconds']:.3f}秒。没有优化实现或新增性能收益。

## 空档期间Host在哪个阶段

为排除采样首尾不完整迭代，按每卡第一个MODEL_EXECUTE到最后一个MODEL_EXECUTE裁出11个完整循环。每卡设备窗口平均{out['scoped']['mean_device_window_ms']:.2f}ms，Free平均{out['scoped']['mean_free_ms']:.2f}ms，Free占设备时间平均{out['scoped']['mean_free_pct_of_device']:.2f}%。先逐卡计算占比，再取16卡平均；分项互斥、合计100%。原完整采样窗口也保留，prepare input仍占约94%的Free，结论一致。

| Host阶段 | 占全部空档 | 占设备窗口 | 具体工作 |
|---|---:|---:|---|
{table}

DP0/DP1的prepare input分别占空档{next(r for r in out['groups']['dp0']['rows'] if r['key']=='prepare input')['mean_pct_of_free']:.2f}%/{next(r for r in out['groups']['dp1']['rows'] if r['key']=='prepare input')['mean_pct_of_free']:.2f}%。这是阶段定位，不等于这些时间全部可省；prepare input内也可能有必要等待。与设备工作重叠的CPU阶段部分没有算进Free。

## 输入准备里，重点又在哪里

代表EP0/EP8各11轮，prepare input墙钟平均45.60/45.73ms，其中与Free相交28.80/28.08ms。按实际调用边界进一步分为：

| 代表卡每轮子窗口 | EP0墙钟 / Free | EP8墙钟 / Free | where次数 |
|---|---:|---:|---:|
| 输入状态、位置/缓存映射至DP协调结束 | 12.00 / 0.37ms | 12.85 / 0.39ms | 2 |
| 协调结束至首次原生metadata算子提交前 | 31.79 / 27.72ms | 30.98 / 26.93ms | 243 |
| metadata提交及入图前收尾 | 1.81 / 0.72ms | 1.90 / 0.76ms | 0 |

中间窗口是明确的调用边界，不冒充某个Python子函数的精确耗时。源码显示_build_attention_metadata遍历缓存组和Attention组；legacy CP builder调用_build_local_token_metadata构造本卡query区间、前缀长度、suffix、掩码和local seq_lens，再做pin/copy/where等交付。CPU输入会走torch fallback。243次where的时序与这些源码相符，但主线程没有完整Python栈，尚不能逐条声明都来自某一个函数。

## 值得推进的方向

先沿已有metadata-host / BatchFrame共同生产方向，核对同代、同语义的CP区间/长度/掩码能否只算一次，复用缓冲并合并交付。图外Host构造和后端提交应与设备kernel一起计价。现有slot mapping已有fused-groups路径，不能将它重新计为新融合。每个缓存组的block table、slot、压缩状态和末读/发布合同保留，旧CPU-plan、25次H2D等历史信用不重复相加。

本次确认热点位置，候选仍为“待确认”；没有直接改优化代码。草稿Host阶段和Gloo不作为当前这笔空档的首要大项，但保留各自局部开发价值。具体净省需对替代成本、重叠和精度验证后判断。

## 原样本时序交叉验证

旧16卡steady中约97%的空档在目标整图执行区间外；按Free为分母，75.61%在下一图外计算任务提交API开始前，4.39%在调用中，14.93%在调用返回后，3.10%图内，剩余1.97%未精确关联。这张提交账和新增阶段账分开记，不相加，也不混合两次样本绝对时间。它支持优先追Host向下一步交付输入，而不是将所有空档归为设备算子内部慢。

## 数据和复算

- [结构化结论](findings.json)、[原样本时序账](submission-summary.json)、[原16卡逐调用关联](idle-attribution.json)。
- [新增16卡阶段分析](scoped-capture/scope-analysis.json)、[代表卡细分](scoped-capture/prepare-boundary-analysis.json)、[恢复验证](scoped-capture/restored-smoke.json)。
- [冻结代码索引](source-map.json)、[追加只读源码](additional-source-map.json)，正文只承认实际审阅范围，不声明全部源码已穷尽。
- 原始trace留在scoped-capture/launch.json所列171自有目录。summarize.py离线复算本结论；其余脚本和采集回执完整保留。

EngineCore调度进程本身没有直接采集，不能给它编造独立耗时，也不能把未覆盖的0.08%当作它的总CPU时间上限。所有百分比均属设备短采样，不能直接按16秒整批请求放大。profiler有开销，最终收益回到无profile、同负载验证。
'''
(P/'REPORT.md').write_text(report)
print('Summarized',[(r['label'],round(r['mean_pct_of_free'],3)) for r in out['scoped']['rows']])
