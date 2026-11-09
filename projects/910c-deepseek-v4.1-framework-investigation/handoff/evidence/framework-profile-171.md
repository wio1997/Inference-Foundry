# 171 推理框架 profile · 2026-11-10

已实际采集并导出。冻结 C8、32,768 输入、1,024 输出、90%前缀预热，DP2/TP8/EP16，DeepSeek-V4.1-Flash W8A8、DSpark、Engram。8条参考请求及8条profile请求均验证实际token长度。采集由已有ZCode执行器启动，使用vLLM官方CPU+NPU Level1 profiler及其CANN/msprof数据导出，保留前端CPU、worker算子/提交、CANN API和设备时间线。

**当前判断为待确认。** 最值得深入的框架问题是DP组间进度与CPU协调失衡。DP1的Gloo等待与设备空档明显重叠。设备侧最大单项仍是DFC。下表给出本轮新证据。

| 环节 | DP0的8卡中位数 | DP1的8卡中位数 | 判断 |
|---|---|---|---|
| 设备分析窗口 | 1939.27 ms | 2801.30 ms | 各卡实际采到的窗口长度不同 |
| CPU Gloo all-reduce等待 | 76.25 ms，窗口3.93% | 981.65 ms，窗口34.67% | 优先查DP协调与到达差异 |
| 设备Free空档 | 330.39 ms，窗口17.04% | 1082.38 ms，窗口38.64% | DP1空档更多 |
| Gloo与Free的交集 | 0.79 ms | 710.28 ms | 同卡同时间线相交；仍需分辨慢侧上游与协调协议的贡献 |
| Event同步等待 | 230.69 ms，窗口11.87% | 255.26 ms，窗口9.10% | 与Free交集仅2.52/3.04 ms，多数等待与设备工作重叠 |
| DFC设备区间 | 679.91 ms，窗口35.06% | 842.09 ms，窗口30.07% | 最大设备热点，内部含通信/等待 |

前端8次tokenization的Python追踪自身时间合计466.21ms，8192次DecodeStream输出解码合计103.00ms，HTTP发送合计约43.27ms。这些值跨线程/请求累计，可重叠。长时间SimpleQueue.get及poll主要表现为等待下游，用各自的壁钟时长排序会高估可消除CPU工作。

无profile整批16.428s，带profile整批17.481s，单轮差异6.41%。本轮采集开销和运行波动都在这个差异里。worker覆盖前12个迭代，前端和请求起止覆盖完整请求。上述百分比的分母是各卡设备分析窗口，不能直接转成整批吞吐提升。EngineCore内Python scheduler没有直接采到，不能给调度器自身CPU耗时比例。

下一步优先将Gloo的30次左右协调与target/draft迭代、两DP就绪点对齐，区分较慢组的必要工作、Host提交与可合并协调。冻结源码model_runner_v1.py的_sync_metadata_across_dp使用CPU组all-reduce，属于核对入口；本轮尚未把全部Gloo事件逐一绑定到该函数。关联地图C04/C13及相邻生产者，继续保持待确认。Event等待先沿对应未完成设备工作追踪。DFC内部继续区分必要GMM、EP和可重叠的shared GateUp。

原始数据及16张设备trace留在171自有目录，约3GB导出timeline另加原始PROF。运行位置见[运行与指标](framework-profile-171.json)，指标及口径见[结构化结果](framework-profile-171.json)，复算输入见[逐卡交集](framework-profile-171-overlap.json)。[复算脚本](framework-profile-171-overlap.py)按每卡合并区间再求Gloo/Free交集。

采集经历一次停服前预检错误，以及一次缓存重置接口404后的重试。修正仅在新采集脚本与临时实例中启用开发接口和profiler。原serve.sh、model.py及dsa_v41.py哈希与冻结记录一致，模型/算子实现未修改。原容器已恢复、health200，且8输出token的小请求完成。[恢复状态与请求](framework-profile-171.json)
