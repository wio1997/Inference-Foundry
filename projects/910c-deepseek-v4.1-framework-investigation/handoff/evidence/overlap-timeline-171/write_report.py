from pathlib import Path
import json,subprocess
P=Path(__file__).resolve().parent;D=json.load(open(P/'view-data.json'));G=D['global'];B=D['cpu_bound']
js="const M=require('../balanced-profile-171/overlap-model.js'),D=require('./view-data.json');console.log(JSON.stringify(D.representatives.map(r=>({ep:r.ep,half:M.simulate(r,{early:.5,planner:true,buffers:true,overhead:.2}).total,all:M.simulate(r,{early:1,planner:true,buffers:true}).total,late:M.simulate(r,{early:1,planner:true,buffers:true,release:40}).total}))))"
scenarios=json.loads(subprocess.check_output(['node','-e',js],cwd=P));(P/'scenario-examples.json').write_text(json.dumps(scenarios,indent=2))
lines=['# 空档能被计算覆盖多少：时间线与全量融合地图','',
'**结论：有一部分正在被覆盖；还能多覆盖多少，取决于可提前字段、结果释放、CPU资源与缓冲退役。当前数据能给容量上限和条件预算，尚不能给已经证明可实现的吞吐收益。**','',
'## 原样本里已经发生什么','',
f"16卡各11闭合循环，平均窗口{G['window_ms']:.3f}ms，即{G['window_ms']/11:.3f}ms/周期。Computing {G['compute_ms']:.3f}ms、Free {G['free_ms']:.3f}ms、暴露通信{G['comm_ms']:.3f}ms。",'',
f"每周期输入准备墙钟约{G['prepare_wall_ms']/11:.3f}ms，其中与设备计算相交{G['compute_during_prepare_ms']/11:.3f}ms，已经存在重叠；仍与Free相交{G['prepare_free_ms']/11:.3f}ms。这是Host阶段墙钟，不等同CPU指令执行量。",'',
'当前Host提交草稿/结果链随后准备下一轮；同一线程不能同时执行两份CPU工作。前后迭代还通过接受数、草稿ID、seq_lens、缓存和复用bank连接。','',
'## 三个数字不能混成同一个预测','',
'|判断|计算|含义|','|---|---|---|',
f"|设备时间容量上限|{G['window_ms']:.3f}/({G['window_ms']:.3f}−{G['free_ms']:.3f}) = {G['capacity_speedup']:.6f}×|把全部Free抹去、其它工作与成本不变；不是可实现预测|",
f"|固定CPU账本条件界|{B['batch_ms']:.3f}/{B['max_worker_cpu_ms']:.3f} = {B['fixed_cpu_demand_speedup_cap']:.6f}×|另一轮完整轻量请求；假设所有记录CPU耗时都必须保留在原worker线程上|",
'|依赖/资源条件预算|删除的暴露准备 + 可提前且能安放的部分 + 设备净缩短 − 新增成本|参数未实测，不能用历史候选的百分比直接填入|','',
'CPU账本包含运行时、探针和可能轮询成本。消除这些CPU工作、用更高效实现、增加CPU并行资源或换执行协议会改变此约束；1.047×不是融合总收益上限。两轮样本绝对时间不混合；本轮没有实测CPU争用、字段可提前比例或最终请求收益。','',
'## 图中条件例子','',
'下面是假设，不能当作已确认方案：所选暴露准备中50%可提前，周期起点已满足其数据依赖，独立planner资源和私有缓冲安全成立，每周期新增关键成本0.2ms。','',
'|代表段|原3周期ms|条件最多减少ms|容量系数|数据到40ms才释放时的新增覆盖|','|---|---:|---:|---:|---:|']
for s in scenarios:lines.append(f"|EP{s['ep']}|{s['half']['base_ms']:.3f}|{s['half']['saved_ms']:.3f}|{s['half']['speedup']:.4f}×|{s['late']['covered_ms']:.3f}ms|")
lines+=['',
'释放延迟0不表示接受数或草稿ID真的已知，只是该组被选字段的条件设定。额外planner也可能争CPU/NUMA/内存带宽；双缓冲不能只保护consumer，CPU H2D源与producer/device读者的末读均需退役。','',
'## 可提前与必须晚交付','',
'|工作|释放条件|图中处理|','|---|---|---|',
'|配置常量/稳定batch结构/兼容公共字段|结构已确定且与当前数据代次相容|作为候选提前计划，不凭shape跨代缓存|',
'|接受数、draft token、CPU/GPU长度校正|对应采样/校验/草稿生产者真实完成|保留依赖，不拿Host提交结束替代device结果就绪|',
'|block table、slot、C2/native attrs|对应物理资源和代次就绪|保留逐资源差异及必要发布|',
'|live CPU/device输入bank覆写|CPU源、producer与consumer都满足复用条件|提前阶段写私有快照，晚阶段才绑定发布|',
'|主worker上的另一份CPU准备|有可调度CPU时间或独立资源|未证明资源时不自动计新增覆盖|','',
'源码依据：gpu_model_runner.py synchronize_input_prep约3841行；Ascend runner _update_states约961行、_prepare_inputs约1281行、GPU权威length校正约1604行、draft与异步输出约2887–2970行；同代BatchFrame协议与双端退役既有证据复用。完整source路径见旧framework/idle证据包。','',
'## 数学共享、融合、megakernel如何放在一起','',
'- 数学共享/计算消除：删除原来必须付的CPU/设备工作，缩短条形；保留原语位、精度和状态合同。',
'- kernel融合/批量发布：减少中间写读、重复索引、提交及工作区，按实际可删除部分计价。',
'- megakernel/tile流水：统一生产消费和资源调度。比如K03是路由Cube完成后重叠shared Up与Vector尾部，MAC不自动减少；内部尾长和争用尚待证。',
'- 通信与布局重组：改变交付payload、collective或重复owner工作；需要匹配prefill/decode和真正的发送/接收读者。',
'- 跨迭代流水：把可以提前的准备移到上轮计算下；结果相关晚交付和bank保护边界保留。','',
f"地图完整映射主台账72条记录，加本轮P01共73条；另映射执行重审{len(D['research'])}条研究记录。主台账快照为46条旧候选、15条代码交付链、11条K切口。保留全部编号、条件、原收益口径、模块、机制、进度与重复关系，不合计为独立开发项目或总收益。",'',
'## 对账与局限','',
'区间模型对删除与覆盖互斥记账。已有prepare/compute重叠不再次认领；设备计算缩短会减小可用计算覆盖窗口。分类量化边界按通信→Free→Computing保守互斥化，原始值保留；白色边界不编造成调度。',
'条件图是活动容量压缩与重排示意，并未把每个Host函数/CPU核、设备task、队列和跨卡依赖建成可执行DAG。只加速设备不能自动证明Host提交也会提前；仍须同时检查CPU账本和真实数据释放。通信等未建模项不自动消失。',
'任务位置保留原始Type/time/Stream ID/Task ID；通用Type的高亮不等于调用点已唯一绑定。K01/K02等prefill记录不会把旧TTFT收益加到当前decode。K00与F01/F02/F05等已做项不能重复算新增收益。',
'图中默认未确认提前比例为0。所有新性能方案仍待确认，没有性能实现或已兑现吞吐提升。','',
'## 复用与验证','',
'独立新入口仍为根目录171-框架Profile热点.html，默认“空档覆盖与融合地图”；旧函数账单/算子明细/16卡时间线保留。',
'复算：build_data.py→Node verify_model.cjs→balanced-profile-171/build_global_view.py；UI检验覆盖73候选、45研究条目、12模块、选择/过滤、EP0/EP8与周期放大。原始精确抽取以measured-timeline.json.gz保存；展开JSON留本机，raw trace仍在171旧目录。',
'本轮新NPU采集/模型请求/服务操作均0；只读已完成样本和既有源码。一次SSH255后按同入口重试成功，无凭据复制或替代认证。','']
(P/'REPORT.md').write_text('\n'.join(lines))
status='''# 当前状态

介绍：已建立实际重叠→释放/资源约束→条件覆盖→全部融合记录的同一张交互时间线。

到了哪里：复用16卡旧稳态11周期；EP0/EP8显示连续3周期并可放大，保留全部同窗task与stream ID。73条候选记录和45条执行重审记录都归到12模块，可按进度、机制、编号搜索。

结果：设备Free容量上限约1.623×；另一轮完整轻量记录的固定CPU账本条件界约1.047×。可提前比例与争用尚未确认，不能将两者混成请求预测。默认新增覆盖0；滑块与预设展示假设敏感性，删除与覆盖不重复计。

下一步：优先核共同metadata/frame中可删CPU工作和最小字段释放点，再判独立计划资源与双端退役；只有会改变去留的缺口才补证。prefill、K03内部资源和更大megakernel按各自case继续，不把旧百分比相加。

新NPU测试/请求/服务动作0。原服务没有被本轮操作，健康状态沿用上轮恢复回执而非本轮重测。数据、代码、源台账SHA和全部候选范围均保留。
'''
(P/'STATUS.md').write_text(status)
print('Wrote report and scenario examples')
