"""Readable, same-sample rankings; preserve overlap and attribution limits."""
from pathlib import Path
from collections import defaultdict
import csv,json,statistics as st

P=Path(__file__).resolve().parent
raw=json.loads((P/'rank-data.json').read_text())['ranks']
idle=json.loads((P.parent/'idle-attribution-171/findings.json').read_text())

# Specific-module labels are limited to dedicated primitives and reviewed code.
INFO={
 'DFC_target_top6':('目标MoE路由专家','MoE','Dispatch→专家两次矩阵乘与激活→Combine；内部含通信、搬运和等待。','先拆内部权重读取、复用、小M效率与同步；不能将整段时长都当可省。'),
 'DFC_draft_top3':('草稿MoE路由专家','DSpark / MoE','草稿模型的路由专家计算与汇合。','与目标DFC分开计价，单独预算较小。'),
 'QuantBatchMatmulV3':('量化矩阵乘','多模块 / 调用点待绑定','W8A8等量化矩阵乘；同类型覆盖多个投影或专家位置。','按完整shape和读者拆分，核对小M、权重读取和前后交付。'),
 'MatMulV2':('矩阵乘','多模块 / 调用点待绑定','多个非同一调用点的矩阵乘。','先按shape与上下游绑定，避免把不同位置当成一个开发任务。'),
 'MatMulV3':('矩阵乘V3','多模块 / 调用点待绑定','另一类矩阵乘实现。','与V2/量化矩阵乘区分后按调用点计价。'),
 'HcPre':('mHC输入混合','mHC','Attention/FFN前及草稿中的残差混合。','检查公共归约与Post→Pre交付，保留精度和所有读者。'),
 'HcPost':('mHC输出合并','mHC','将子层输出交回残差流。','与下一Pre按大段联合判断。'),
 'VllmAscendQuantLightningIndexerV2':('稀疏注意力索引选择','Attention / QLI','量化索引计算及候选选择。','按输入生产→索引→实际KV读者考察数学共享与交付。'),
 'SparseFlashMla':('稀疏注意力','Attention / SMLA','消费选中位置与KV，执行注意力。','先核选中KV搬运、QK/PV、精度和后续O投影交付。'),
 'TransposeBatchMatMul':('转置批量矩阵乘','Attention O投影（源码支持）','Attention输出投影中的批量矩阵乘；各次调用仍按shape核对。','与SMLA输出布局和TP O投影整体计价。'),
 'ScatterNdUpdateSk':('索引散写','缓存等 / 多调用点','按索引把结果写入目标张量。','核对缓存位置、代次与末读，保留原写入语义。'),
 'RmsNorm':('RMS归一化','多模块','计算归一化，目标与草稿有多处调用。','与实际相邻投影/量化按交付边界合并判断。'),
 'RmsNormCast':('RMS归一化与转换','多模块','归一化并完成类型交付。','数值边界与所有消费者保留。'),
 'ForeachAddList':('批量张量加法','多模块','对一组张量执行加法。','作为完整残差交付的一部分核算。'),
 'InplacePartialRotaryMul':('原地局部RoPE旋转','Attention','对指定维度执行位置旋转。','保留原舍入与末读顺序，联合Q/KV生产判断。'),
 'SparseFlashMlaMetadata':('SMLA参数生成','Attention metadata','生成注意力运行所需的设备参数。','与前面的Host构造区间分开，不将二者重复相加。'),
 'VllmAscendQuantLightningIndexerV2Metadata':('QLI参数生成','Attention metadata','生成索引计算的设备参数。','检查共同输入与批量交付，单独kernel预算有限。'),
 'Transpose':('张量转置','多模块','调整维度顺序。','按实际布局读者判断能否消除，不只看单点。'),
 'ViewCopy':('视图数据拷贝','多模块','完成视图对应的数据拷贝。','优先核可否由上下游直接生产/消费。'),
 'Cast':('精度转换','多模块','将张量转换到另一数据类型。','独立收益预算小，优先作为有价值大段的附属优化。'),
 'DynamicQuant':('动态量化','多模块','生成量化值与比例因子。','保留原scale、舍入和消费者要求，联合投影判断。'),
 'DequantSwigluQuant':('反量化与SwiGLU再量化','MoE / FFN','衔接专家矩阵乘的激活与量化。','与GMM前后阶段共同计价。'),
 'hcom_allGather_':('跨卡全收集','通信','从参与卡收集数据。','先判断哪些字节与交付必要，再看布局折叠和重叠。'),
 'hcom_allReduce_':('跨卡归约并广播','通信','跨卡汇总并发布结果。','按真实依赖与暴露段核算，不能直接相加所有通信时长。'),
 'hcom_alltoall_':('跨卡全交换','通信','各卡交换分片。','核对数据交付布局与通信重组。'),
 'hcom_reduceScatter_':('跨卡归约并分片','通信','归约结果按卡分片。','结合上下游布局与重叠计价。')}
SIMPLE={'Add':'逐元素加法','Sub':'逐元素减法','Mul':'逐元素乘法','SelectV2':'按条件选值','Fill':'填充','Index':'张量索引','IndexCheck':'索引检查','GatherV2':'索引取值','GatherElementsV2':'逐元素索引取值','Slice':'张量切片','ConcatD':'张量拼接','Abs':'绝对值','ArgMaxV2':'最大值索引','BroadcastTo':'广播','ClipByValueV2':'裁剪','Equal':'相等判断','FloorDiv':'整除','FloorMod':'取模','Greater':'大于判断','GreaterEqual':'大于等于判断','Less':'小于判断','LogicalAnd':'逻辑与','LogicalNot':'逻辑非','LogicalOr':'逻辑或','MaskedFill':'按掩码填充','Neg':'取负','NotEqual':'不等判断','Pack':'张量打包','PadV3':'张量填充扩展','Pows':'幂运算','Range':'生成序列','ReduceMean':'归约平均','ReduceSum':'归约求和','RotaryPositionEmbedding':'位置旋转','Rsqrt':'平方根倒数','ScatterElementsV2':'按索引散写','Sigmoid':'Sigmoid激活','Sqrt':'平方根','TensorMove':'数据搬运','Tile':'张量重复','ZerosLike':'创建零张量','MemSet':'内存填充'}

def describe(n):
    if n in INFO:a,b,c,d=INFO[n]
    elif 'AicpuKernel' in n:a,b,c,d='通信辅助任务','通信',n+'对应的设备侧通信辅助任务。','时长可能与HCCL及计算重叠，先核关键依赖。'
    elif 'engram' in n:a,b,c,d='Engram专用操作','Engram',n,'按查表到残差的完整交付计价。'
    elif 'dspark' in n or 'dflash' in n:a,b,c,d='草稿输入/状态操作','DSpark',n,'按草稿整段输入与代次合同计价。'
    elif 'slot_mapping' in n:a,b,c,d='缓存位置映射','输入 / cache',n,'已有融合路径先复用，避免重复开发。'
    elif 'indexer' in n:a,b,c,d='注意力索引辅助操作','Attention',n,'与QLI输入/输出共同计价。'
    elif 'rejection' in n:a,b,c,d='草稿校验与采样','采样',n,'保留原接受和采样语义。'
    else:a,b,c,d=SIMPLE.get(n,n),'通用 / 调用点待绑定',n+'类型任务。','单点预算小的操作优先并入较大交付方案，仍需绑定调用点。'
    return {'label':a,'module':b,'work':c,'next':d,'optimization_status':'待确认'}

def aggregate(ranks):
    n=len(ranks);windows={r['ep']:r['window_ms'] for r in ranks}
    overall={'rank_count':n,'cycles_per_rank':11,'mean_window_ms':st.mean(windows.values()),
             'device':{},'operators':[],'host_groups':[],'host_ops':[]}
    for key in ranks[0]['device_union_ms']:
        overall['device'][key]={'mean_ms':st.mean(r['device_union_ms'].get(key,0) for r in ranks),
                                'mean_window_pct':st.mean(r['device_union_ms'].get(key,0)/r['window_ms']*100 for r in ranks)}
    names=sorted({o['name'] for r in ranks for o in r['operators']})
    for name in names:
        values=[(r,next((o for o in r['operators'] if o['name']==name),None)) for r in ranks]
        exists=[v for _,v in values if v]
        mean=lambda k:st.mean(v.get(k,0) if v else 0 for _,v in values)
        shapes={}
        for rank,v in values:
            if not v:continue
            for case in v['shape_cases']:
                signature=json.dumps(case['fields'],sort_keys=True)
                if signature not in shapes:shapes[signature]={'fields':case['fields'],'calls_sum_across_ranks':0,'wall_sum_across_ranks_ms':0,'rank_examples':[]}
                shapes[signature]['calls_sum_across_ranks']+=case['calls'];shapes[signature]['wall_sum_across_ranks_ms']+=case['wall_sum_ms']
                shapes[signature]['rank_examples'].append(rank['ep'])
        cases=[]
        for c in shapes.values():
            cases.append({'fields':c['fields'],'mean_calls_per_rank':c['calls_sum_across_ranks']/n,'mean_task_ms_per_rank':c['wall_sum_across_ranks_ms']/n,'ranks':c['rank_examples']})
        op={'name':name,'kind':exists[0]['kind'],**describe(name),'mean_union_ms':mean('union_ms'),
            'mean_window_pct':mean('window_pct'),'mean_calls':mean('calls'),'mean_calls_per_cycle':mean('calls')/11,
            'calls_range':[min(v['calls'] if v else 0 for _,v in values),max(v['calls'] if v else 0 for _,v in values)],
            'call_median_us':st.median(v['call_median_us'] for v in exists),
            'call_p90_us':st.median(v['call_p90_us'] for v in exists),
            'mean_target_graph_union_ms':mean('target_graph_union_ms'),
            'aic_mac_active_pct':st.median(v['aic_mac_active_pct'] for v in exists if v['aic_mac_active_pct'] is not None) if any(v['aic_mac_active_pct'] is not None for v in exists) else None,
            'aic_mte2_active_pct':st.median(v['aic_mte2_active_pct'] for v in exists if v['aic_mte2_active_pct'] is not None) if any(v['aic_mte2_active_pct'] is not None for v in exists) else None,
            'shape_cases':sorted(cases,key=lambda c:-c['mean_task_ms_per_rank'])}
        overall['operators'].append(op)
    overall['operators'].sort(key=lambda o:-o['mean_union_ms'])
    groups={k for r in ranks for k in r['host_self_groups_ms']}
    overall['mean_prepare_free_ms']=st.mean(r['prepare_free_ms'] for r in ranks)
    for k in groups:
        overall['host_groups'].append({'name':k,'mean_free_ms':st.mean(r['host_self_groups_ms'].get(k,0) for r in ranks),
            'mean_pct_of_prepare_free':st.mean(r['host_self_groups_ms'].get(k,0)/r['prepare_free_ms']*100 for r in ranks),
            'mean_pct_of_device':st.mean(r['host_self_groups_ms'].get(k,0)/r['window_ms']*100 for r in ranks)})
    overall['host_groups'].sort(key=lambda r:-r['mean_free_ms'])
    for name in sorted({o['name'] for r in ranks for o in r['host_ops']}):
        values=[next((o for o in r['host_ops'] if o['name']==name),{}) for r in ranks]
        overall['host_ops'].append({'name':name,'category':next(v['category'] for v in values if v),
            'mean_calls':st.mean(v.get('calls',0) for v in values),
            'mean_calls_per_cycle':st.mean(v.get('calls',0) for v in values)/11,
            'mean_self_host_ms_in_prepare':st.mean(v.get('self_host_ms_in_prepare',0) for v in values),
            'mean_self_host_ms_during_free':st.mean(v.get('self_host_ms_during_free',0) for v in values)})
    overall['host_ops'].sort(key=lambda o:-o['mean_self_host_ms_during_free'])
    assert abs(sum(r['mean_free_ms'] for r in overall['host_groups'])-overall['mean_prepare_free_ms'])<1e-5
    return overall

out={'date':'2026-11-10 Asia/Shanghai','sample':'idle-scopes-profile-20261110T045717Z',
     'scope':'16 ranks; first through last MODEL_EXECUTE; 11 closed cycles; one sample for host and kernel ledgers',
     'all':aggregate(raw),'dp0':aggregate([r for r in raw if r['dp']==0]),'dp1':aggregate([r for r in raw if r['dp']==1]),
     'idle_stages':idle['scoped']['rows'],'prepare_boundaries':idle['prepare_boundaries'],
     'limits':['All times are per-card means over a short profiled window, not all 16 seconds or cross-card sums.',
               'Different device operator types and communication tasks can overlap. They are rankings, not an additive request breakdown.',
               'Host self wall time removes nested cpu_op intervals. It is not CPU instruction time; underlying library activity remains included.',
               'Uncovered host intervals may contain Python, runtime code or thread pauses. No exact-function attribution is fabricated.',
               'General matmuls, norm, copy and cast types aggregate multiple call sites. Shape and target-graph split are shown; some module bindings remain unverified.',
               'Pipe activity is not peak compute or bandwidth utilization. Optimization candidates remain pending.']}
assert abs(out['all']['mean_prepare_free_ms']-idle['scoped']['rows'][0]['mean_free_ms'])<1e-5
(P/'findings.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
def export_csv(filename,rows,keys):
    with (P/filename).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)
export_csv('operator-ranking.csv',out['all']['operators'],['name','label','module','kind','mean_union_ms','mean_window_pct','mean_calls','mean_calls_per_cycle','call_median_us','call_p90_us','mean_target_graph_union_ms','aic_mac_active_pct','aic_mte2_active_pct','work','next','optimization_status'])
export_csv('host-breakdown.csv',out['all']['host_groups'],['name','mean_free_ms','mean_pct_of_prepare_free','mean_pct_of_device'])
export_csv('host-operations.csv',out['all']['host_ops'],['name','category','mean_calls','mean_calls_per_cycle','mean_self_host_ms_in_prepare','mean_self_host_ms_during_free'])

def table(ops):
    return '\n'.join(f'| {i+1} | {o["label"]} | {o["name"]} | {o["mean_union_ms"]:.3f} | {o["mean_window_pct"]:.3f}% | {o["mean_calls_per_cycle"]:.2f} | {o["call_median_us"]:.2f} | {o["module"]} |' for i,o in enumerate(ops))
a=out['all'];operators=[o for o in a['operators'] if o['kind']=='operator'];communications=[o for o in a['operators'] if o['kind']=='communication']
report=f'''# 171详细性能账单：空档、CPU小操作与全部设备算子

本轮只复算已经导出的同一次阶段标记采样；模型请求/NPU测试/新profile/服务改动均0。16卡各取11个闭合循环，设备窗口平均{a['mean_window_ms']:.3f}ms；表格时间是每卡平均，不能乘16作为请求收益。整批无profile参考仍为此前三轮中位16.091秒、509.11输出token/s。

## 同一窗口的一级分类

| 类别 | 平均ms/卡 | 平均窗口占比 |
|---|---:|---:|
'''
for key,label in [('Computing','设备算子执行'),('Free','没有被记录的计算/通信覆盖'),('Communication(Not Overlapped)','未被计算覆盖的通信')]:
    r=a['device'][key];report+=f'| {label} | {r["mean_ms"]:.3f} | {r["mean_window_pct"]:.3f}% |\n'
report+='''
Overlap分类加总与闭合窗口有不到0.1%的边界差，保留原始分类，不将它编成新瓶颈。算子执行范围包含任务内部搬运和等待，不等于所有算术单元一直工作。

## 空档展开到具体Host阶段

| 空档时Host在哪个阶段 | 平均空档ms/卡 | 占设备窗口 | 具体内容 |
|---|---:|---:|---|
'''
for r in out['idle_stages']:report+=f'| {r["label"]} | {r["mean_free_ms"]:.3f} | {r["mean_pct_of_device"]:.3f}% | {r["work"]} |\n'
report+='''
输入准备再分：代表EP0/EP8每轮从输入状态处理到DP协调结束，墙钟12.00/12.85ms、与空档交集仅0.37/0.39ms；协调结束至首个原生metadata提交前，墙钟31.79/30.98ms、空档27.72/26.93ms、where事件243次；提交和收尾墙钟1.81/1.90ms、空档0.72/0.76ms。这是提交边界定位，不是假造的Python子函数耗时。

## 输入准备的空档：去除CPU-op嵌套后在做什么

下列是prepare input与Free相交时间的内部互斥分解。每段只归给最深的已记录cpu_op，未覆盖区间单列。cpu_op自身的墙钟仍可包含底层库调用；不能把它叫纯Python执行或全部可省。

| Host活动 | 平均空档ms/卡 | 占输入准备空档 | 占设备窗口 |
|---|---:|---:|---:|
'''
for r in a['host_groups']:report+=f'| {r["name"]} | {r["mean_free_ms"]:.3f} | {r["mean_pct_of_prepare_free"]:.3f}% | {r["mean_pct_of_device"]:.3f}% |\n'
report+='''
“无CPU-op范围覆盖”仍位于已确认的prepare input范围里，但可能是Python代码、运行时调用或线程停顿；主线程没有完整Python栈，暂不能进一步分配给具体函数，不能把它统一标成“调度”。

### Host具体操作排行

| Host事件 | 每轮平均次数 | 输入准备内去嵌套墙钟ms | 其中与空档重叠ms |
|---|---:|---:|---:|
'''
for r in a['host_ops'][:25]:report+=f'| {r["name"]} | {r["mean_calls_per_cycle"]:.2f} | {r["mean_self_host_ms_in_prepare"]:.3f} | {r["mean_self_host_ms_during_free"]:.3f} |\n'
report+='''
## 设备算子完整排行

同类任务内部区间取并集，不把同类型重叠重复加一次。不同类型仍可互相重叠。单次中位数先按卡计算，再取16卡中位；每轮次数由每卡窗口调用数除11再平均。通用矩阵乘等多调用点未完全绑定时明确标注，不能直接当一个融合开发项目。

| 序号 | 算子说明 | Type | 平均区间ms/卡 | 窗口占比 | 每轮平均调用 | 单次中位µs | 所属位置 |
|---|---|---|---:|---:|---:|---:|---|
'''+table(operators)+'''

## 通信任务完整排行

这些是任务自身区间时长，可与计算、其它通信或辅助任务重叠，不等于一级分类中的“未重叠通信”。不能直接相加来宣称通信瓶颈。

| 序号 | 说明 | Type | 平均区间ms/卡 | 窗口占比 | 每轮平均调用 | 单次中位µs | 所属位置 |
|---|---|---|---:|---:|---:|---:|---|
'''+table(communications)+'''

## 当前开发优先级

1. 输入/metadata共同构造与批量交付：大空档已有阶段和提交边界证据；先核CP区间、长度、掩码、缓冲和发布的同代共享，精确Python热点仍缺证据，不重复旧CPU-plan/slot mapping信用。
2. 目标DFC整段：最大设备单项，先拆内部权重读取、专家GMM和同步成本；已有大融合不能直接假设再融合就省全段。
3. 量化矩阵乘、普通矩阵乘、mHC和Attention大段：按shape与真实消费者拆分开发任务；多个候选共享预算时只计增量。
4. 小Cast/Fill/单次量化：独立预算小，优先作为上述大段的附属项；不因单点时间小而剪掉能额外消除工作的大parent。

所有优化方向保持待确认；本轮未新增优化实现或已兑现收益。findings.json包含全部类型、全shape签名、目标图内时间及PMU，rank-data.json保留逐卡原件，analyze_detailed.py和summarize.py可复算。输入阶段和算子表属于同一批数据；其子表是展开关系，不层层相加。仍不能将此约0.84秒窗口直接放大为整批16秒的精确分解。
'''
report+='\n时间总账已另按逐区间去重核准：[54%核准](RECONCILIATION.md) · [可加总时间树CSV](computing-time-tree.csv) · [逐函数计时和框架优化推进方案](TIMING-PLAN.md)。排行与互斥分账是不同口径，不能直接相加。\n'
(P/'REPORT.md').write_text(report)
print('Same-sample report:',len(operators),'operator types,',len(communications),'communication types; window',round(a['mean_window_ms'],3))
