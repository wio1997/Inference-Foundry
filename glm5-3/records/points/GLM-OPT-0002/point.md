# GLM-OPT-0002 — 完整请求的资源部署与动态placement

类型：deployment/runtime；状态PROTOTYPE，尚无性能Run或KEEP。

问题：当前1P1D两台各持完整模型，长输出主要由D16芯执行，P仅产生首token。旧c3和新短E2E证明PD路径有效但未稳定达SLO；并发1短TPOT也超过18ms。不能只在D调并发后宣布两台硬件极限。两个完整模型实例已合法驻留，研究完整请求分配给两个节点的可行性与容量；未证明收益。

来源：当前安装0.27 Mooncake scheduler的get_num_new_matched_tokens在kv_transfer_params=None时返回(0,False)，request_finished返回(False,None)，支持本地请求的源码路径。引擎attention/role、prefill budget与MTP/Graph完整本地执行仍需真实E2E核验。P MTP1/eager与D MTP3/FULL_DECODE_ONLY，未经配置对齐不能将两实例差异当代码收益。TP8承载方式按EP世界判断：源码支持DP2×TP8×EP16专家跨DP/TP分片，不能用checkpoint/8直接否定；非专家、KV/activation/Graph适配仍需实际验证。

实现：placement.py以唯一服务进程管理请求lease、未知/已校准成本、drain和代际，释放恰好一次；未校准用active计数，不混用秒和请求数。replica_gateway.py透明转发完整API payload/native status/raw body，保留sampling/tool/多模态字段，流开始后不重试；断开、header错误和cancel shield资源回收。每个engine仍拥有KV/MTP/Graph状态，router不发明或跨engine复制状态。默认一个调度进程，当前未宣称多worker共享lease。

另外GLM-OPT-0001的glm_gateway.py以source hash锁定完整stock PD proxy，提供可关闭的请求阶段观测，保留native routing/retry/状态处理与所有PD接口。与完整请求路由是不同部署候选，不能把trace收益算优化。

验收：先模拟验证payload/status/SSE、错误取消、lease/idempotence/drain/代际与动态更新；正式Run0003释放共享资源后，用当前两机小真实请求核验本地全路径。可行再选择对齐配置和同工作量真实E2E，动态到达/混合长度容量是目标，不以固定shape手工清单缩小功能。若本地consumer路径不支持，就依据证据选择必要重启/可行配置。

裁决：PROTOTYPE/INCONCLUSIVE。无新增实际请求或服务部署；不提升Current。未知：本地完整prefill/生成合法性、observer成本、两实例配置对齐、动态压力成本、达标稳定容量及全可行域上界。

合成合同：CONTRACTS-20261001T1010Z经真实zcode --prompt/Result/bridge通过，23 tests，stderr3594B SHA256 d6038dc4269f4b5d07f1f178ea7567f77bb7daa02706f9717fcffdae9bec2861。只证明所列CPU合同，不是GLM E2E。历史ROUTER-TEST-0937的九测试原始报告通过，但Result格式/十测试数量声明无效；保持INVALID、不修改raw。Zcode命令模式现在明确包装为Bash执行，避免绝对路径被当CLI slash命令。

并行配置依据见[source_notes](source_notes.md)。默认PP39/39不能无条件沿用：第二stage从共享索引层39开始，依赖stage0层38；当前IntermediateTensors不传索引buffer。38/40从完整indexer层38切入，是待验证候选，非运行正确性结论。DP/EP和PP是可行域研究，不预置固定路线或阶段。

发布前NATIVE-CONTRACTS-20261001T1013Z：23/23再次通过，完整stock source SHA校验与route list加载成功，无HTTP/NPU调用。test.stderr3594B SHA48afc3aff3bbf975da09d6fad5d9a63c11fdb9e4f506cdf98a812c1fb5ead19e；native.stdout416B SHA78102aaf618eec323e60e5268e13d0139e3103ae7a4a5bff664f99196fddda15。
