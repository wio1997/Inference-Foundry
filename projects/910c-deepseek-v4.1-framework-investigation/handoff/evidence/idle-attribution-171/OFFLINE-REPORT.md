# 原16卡稳态trace：设备空档先定位到哪里

本页复用已有balanced-profile-171数据，不新增推理。阶段归属另用scoped-capture的定点补采确认，不混用两次采样的绝对时间。

将每张卡的设备空档单独当作100%，先求各卡占比，再对16卡取平均。

| 空档期间的时序位置 | 占全部Free |
|---|---:|
| 下一项图外计算任务的提交API尚未开始 | 75.61% |
| 下一项提交API正在调用 | 4.39% |
| 下一项提交调用已返回，设备尚未执行该任务 | 14.93% |
| 目标模型整图已开始、尚未结束 | 3.10% |
| 下一活动不是已绑定的普通计算任务，或无精确关联 | 1.97% |

第一项说明应重点追Host在下一次提交前做什么；它不等于75.61%都能删除。第三项包含设备流顺序、依赖、运行时调度等，不能一律算Host准备。目标图内空档按设备MODEL_EXECUTE到同connection_id的MODEL_WAIT_COMPLETE结束界定。这里的图外仍包括草稿、采样、缓存及跨轮准备，不等于纯请求调度。

16卡的图内空档占Free均为2.71%–3.44%，因此“约97%在图外”在两DP组都成立。原稳态图的Free为DP0/DP1卡内中位356.07/353.62ms；图内仅约11ms。分项中位数与总量中位数不保证直接相加。

Host的同时活动是另一张账，不与上表相加。按优先级去重后，图外空档中与张量CPU事件相交的组中位约158/159ms；后台Dequeue独占剩余区间约24ms；约148ms未被这些CPU事件和API覆盖。原worker主线程没有完整Python栈，因而不能从这148ms推断EngineCore在选请求。

代表EP0与EP8在12迭代中的事件数量一致：aten::slice 14004次，aten::where 3228次，aten::copy_ 5928次，empty_tensor 11604次，aten::to 6300次。这些是Host层的PyTorch事件，部分互相嵌套，并不等于同数量独立kernel或都可消除。它们和Free的交集，指向图外输入、metadata和草稿交付中的碎片化张量处理。

队列按Enqueue/Dequeue的correlation_id精确匹配；常见InplaceCopy、Where、Fill入队到开始处理的中位为约3–5µs，部分事件管理P90为数十µs。不能把所有后台队列墙钟时长相加为设备空档，也不能凭当前中位数排除少数长尾。

证据：idle-attribution.json为16卡原件，submission-summary.json为上述比例；analyze_idle.py可在原trace目录复算。source-map.json及source/保留3份匹配原冻结SHA的源码。阶段补采使用现有record_function_or_nullcontext开关，补prepare input / draft_token等范围；不改变模型及算子实现。
