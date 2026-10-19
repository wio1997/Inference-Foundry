当前入口已从方案收束为[已实现的窄范围尾行探针](../../runtime-patches/state-tail-probe-v1/README.md)。它默认关闭，只做cold C1 N=256/257的原生MTP调用/完成依赖和P/D末两行取证，不采纳y1、不新增kernel/全局同步，不证明完整前缀或提速。模型/NPU尚未运行，等待新的有限窗口。

32/33两台各18份源码SHA已在固定实际镜像中配对，检查约0.317/0.267秒，90秒有界；两台均无NPU/驱动/权重挂载、network none/read-only、CPU1/内存1G，resource_lease=null。访问pod→node→hostname先单独核对，完整命令和下载SHA见[本轮实际身份收据](../../evidence/20261018-p-first-state-audit/image-identity-live-v1/summary.json)。只证明这18个文件身份，不是实际proposer对象或模型状态运行通过。未检查的frontend stats等仍维持原固定提交参考等级。

**更正前轮表述：Mac-only只是上一轮范围，人类没有撤销32/33既有SSH/只读授权。** 本轮已用既有私有连接配置恢复访问，没有要求用户重新批准SSH。没有新建/续期设备预留，也未启动模型；无设备源码核验可继续使用既有授权。

`review-plan.json`保存两份原生基线配置SHA、精确prompt IDs、独立cold salt要求、P K1/D K5、greedy/min1/ignore_eos与P输出1/D输出2。当前取样范围已缩为N−2/N−1，优先发现足以否定候选的事实。这个review schema尚未绑定fresh lease，故仍`execution_ready=false`；**不是适配器尚未实现**。`runtime-patches/state-tail-probe-v1/prepare_launch.py`在新窗口获准、有两枚匹配fresh Pod收据时生成可供既有starter执行的计划；没有Pod收据就只生成不能启动的模板。

实现包含单份worker overlay构建、CPU-only hash路径stager、两个原生功能请求客户端、P/D row配对器，以及只按本次UID/CID保全和停止的finish脚本。复用现有collect_node/start_capacity/backup_artifacts，无新服务或传输协议。具体边界、命令、退出条件和限制统一见[运行说明](../../runtime-patches/state-tail-probe-v1/README.md)。

P探针只在原生Async输出已有event等待之后读取，记录等待前/后状态及writer/constructor stream，确认当前target→proposer→copied hidden尾部和移位token/位置；D在原生READ/futures返回后、finished入队前读取，记录实际所选P rank和kernel block。按实际注册的packed C8主view与split Indexer K/scale view取字节，缺slot/组件/完成依据或输入已重用就拒绝。客户端若发现P各rank的MTP调用/hidden/原生完成依据不全，在发D前停止。成功只叫sampled_rows_match；两个input/少量行不能升级为完整前缀正确、kernel trace或优化接受。

Mac检查包括源码接口、原生函数体AST保留/default-off gate、真实NumPy共享storage/stride视图、空/错位/错源rank/块/组件/写入依据/完成依据拒绝，以及fresh UID/过期lifetime拒绝。没有fake torch或完整框架导入。结果在[本轮本机检查](../../evidence/20261018-p-first-state-audit/state-tail-probe-v1/local-checks.json)；早先`probe-preparation-v1`是实现前快照，原件保留。

完整历史启动→API ready P11/D12为626.893/636.642秒，并非仅load-start后的201/213秒。新窗口建议18–22分钟、hard max25分钟，ready超过15分钟即止，保留原900秒启动下限和至少5分钟清理余量，不续期。这里没有申请实际窗口；fresh UID/CID/目录/配置不得使用过期D13或旧lease。停止与归档后由既有资源流程以精确新UID释放holder并复核不存在。

```sh
python3 glm5-3-vllm/diagnostics/p-first-state-v1/validate_packet.py
python3 glm5-3-vllm/diagnostics/p-first-state-v1/check_local.py
"$BUNDLED_PYTHON" glm5-3-vllm/runtime-patches/state-tail-probe-v1/check_local.py
```

首批6→6只约束首批行数，不能据此断言完整请求永远不可能少一次target forward；后续若进入query推进实验，必须按同预算/同采样完整请求的实际前向次数判断。本轮不实现该优化，也不作精度/容量/吞吐验收。
