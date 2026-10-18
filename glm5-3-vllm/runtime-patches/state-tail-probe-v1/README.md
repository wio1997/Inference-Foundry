这是已实现、默认关闭的 GLM5.3 尾行诊断，尚未运行模型或 NPU。它只比较 N=256/257 原生 PD 路径的末两行，不采纳 y1、不修改采样/传输协议、不作完整前缀或性能验收。P/D 都只加载一次，32/33 各16die；没有新增 kernel、clone、barrier、stream wait 或全局设备同步。现有 Tensor.cpu 仅在原生完成依据核对后取回少量字节，不能用这些读回补造完成依据。

两台固定镜像的18份源码现已实际逐文件SHA配对，CPU检查耗时约0.317/0.267秒，均无设备/驱动/权重挂载、无租约。[实测收据](../../evidence/20261018-p-first-state-audit/image-identity-live-v1/summary.json)只证明文件身份；实际proposer对象、tensor布局、调用/完成事件和尾行仍等待新NPU窗口验证。SSH/只读核验授权一直有效，前轮Mac-only只是那一轮范围，不需要重新申请SSH。

`probe.py`由`build.py`嵌入一份worker overlay，原生顶层定义AST和函数体保留。启用`GLM53_STATE_TAIL_PROBE=1`后：

- P记录实际Eagle proposer、模型类型、源码SHA和当前runnable绑定。原生MTP首轮记录输入/位置、分层slot metadata和writer stream；没有提前CPU读取、clone或新增事件。原生Async输出先照常等待其已有copy event，随后才检查event状态、MTP stream是否被原native依赖覆盖，核对target/source/copied hidden尾部和移位输入，再读缓存末两行。输入buffer若已被后续正工作复用，拒绝观测；零token原生KV轮询不算输入重写。max1防额外step的依据在已归档原生scheduler中，但最终仍以运行记录判断。
- D在原生transfer bucket成功返回后保留实际P PP/PCP/TP、layer和kernel block配对。所有原生futures返回、但finished尚未入队时，按真实registered tensor views取样，早于该请求D重算N−1。失败、非cold或不支持的拓扑/布局只报告拒绝，不改写原生READ结果。
- 布局限于本模型的native四维packed C8主缓存及split Indexer K/scale；验证spec、dtype、shape、stride、block/offset及共享storage标识，按tensor view取字节，不按裸指针猜布局。C4、复制Indexer、不同CP、缺component/slot等拒绝。每层只取N−2/N−1，单行≤4096字节、每rank缓存行≤1MiB；hidden只保留≤64KiB尾部摘要。非零不是有效性标准，纯字节相等也不能代替writer/完成依据。

新增的3份官方固定提交参考源码补齐aggregator构造链：engine在scheduler connector存在时初始化，factory无producer/consumer分支，因此参考路径支持P收齐全部rank。这3份文件没有纳入18份实际镜像SHA核验，不能升级为现场事实；客户端仍要求16份实际P记录，缺项即停。出处与SHA见[参考清单](../../evidence/20261018-p-first-state-audit/state-tail-probe-v1/aggregator-source-references.json)。

P和D每个进程单独写`/out/state-tail-probe/worker-<pid>.jsonl`。P的少量host锁只防观测时输入buffer重用，改变诊断时序，不能用本轮量性能。记录原生wait前/后的event状态和进程内时间；不同节点时钟不直接排序。CPU数据拷贝仍是既有Tensor复制路径，新的诊断拷贝尚无设备profile，不扩大旧SDMA实测覆盖。

Mac构建和检查（不会连服务器）：

```sh
python3 glm5-3-vllm/runtime-patches/state-tail-probe-v1/build.py \
  --base "$MATCHED_SOURCE/vllm-ascend/vllm_ascend/worker/model_runner_v1.py" \
  --output "$NEW_BUILD"
"$BUNDLED_PYTHON" glm5-3-vllm/runtime-patches/state-tail-probe-v1/check_local.py
python3 glm5-3-vllm/runtime-patches/state-tail-probe-v1/prepare_launch.py \
  --build-manifest "$NEW_BUILD/manifest.json" --run-id "$FRESH_RUN_ID" \
  --output "$NEW_REVIEW_PLANS"
```

构建生成worker、patch、manifest和CPU-only `stage_overlay.py`。没有fresh lease文件时只生成不能启动的review schema。已有Mac构建/配置结果见[本轮证据](../../evidence/20261018-p-first-state-audit/state-tail-probe-v1/)；本机实际生成目录记录在其中的`prepared-artifacts.json`。NumPy检查使用真实CPU storage/stride验证源代码规定的布局，不是实际NPU tensor构造或完整原生回放。

以下只在新的模型/NPU窗口获准后执行；本轮没有执行这些命令。先按review plans中的fresh name/owner申请两台各16die的≤25分钟holder，核对节点、现存进程归属和空闲条件，将只含这两枚新Pod的读取结果作为`$LEASE32/$LEASE33`。不复用旧lease/CID/D13，不续期。通过既有collector把已冻结worker写入任务自己的hash目录，不写镜像或原生安装：

```sh
python3 glm5-3-vllm/tools/collect_node.py --node 32 --script "$NEW_BUILD/stage_overlay.py" \
  --label "${FRESH_LABEL}_stage" --control "$SSH_CONTROL" --output "$EVIDENCE/stage32" --timeout 90
python3 glm5-3-vllm/tools/collect_node.py --node 33 --script "$NEW_BUILD/stage_overlay.py" \
  --label "${FRESH_LABEL}_stage" --control "$SSH_CONTROL" --output "$EVIDENCE/stage33" --timeout 90
python3 glm5-3-vllm/runtime-patches/state-tail-probe-v1/prepare_launch.py \
  --build-manifest "$NEW_BUILD/manifest.json" --run-id "$FRESH_RUN_ID" \
  --lease32 "$LEASE32" --lease33 "$LEASE33" --output "$NEW_LAUNCH_PLANS"
python3 glm5-3-vllm/tools/start_capacity.py --plan "$NEW_LAUNCH_PLANS/P.json" \
  --control "$SSH_CONTROL" --output "$EVIDENCE/P-launch"
python3 glm5-3-vllm/tools/start_capacity.py --plan "$NEW_LAUNCH_PLANS/D.json" \
  --control "$SSH_CONTROL" --output "$EVIDENCE/D-launch"
```

两个start独立，可由现有执行器并行派发。P11/D12历史完整启动至API ready为626.893/636.642秒；仅load-start后的201/213秒会漏掉约7分钟。预计总计18–22分钟、holder最大25分钟；15分钟未ready即保全停止。fresh生成器保留原900秒启动下限，模型lifetime≤20分钟且比holder提前至少5分钟结束，另留60秒启动配置余量。starter重新核对精确UID和剩余时间，过期计划拒绝，不自动续期。UID通用校验之外的25分钟限制只作用于本诊断标记，不改变其他实验的窗口策略。

两服务ready后，用node32既有host采集入口运行客户端（所有下划线变量都来自本次新plan/receipt）。P每rank若没有MTP调用/hidden/原生完成依据，客户端在发D前停止；其他HTTP/数量/首ID错误也立即停止，不重试：

```sh
python3 glm5-3-vllm/tools/collect_node.py --node 32 \
  --script glm5-3-vllm/runtime-patches/state-tail-probe-v1/run_requests.py \
  --label "${FRESH_LABEL}_requests" --control "$SSH_CONTROL" --output "$EVIDENCE/requests" --timeout 390 \
  --script-arg=--execute-functional --script-arg=--p-url --script-arg="$P_URL" \
  --script-arg=--d-url --script-arg="$D_URL" --script-arg=--output --script-arg="$CLIENT_RUN_DIR" \
  --script-arg=--p-probe-dir --script-arg="$P_RUN_DIR/state-tail-probe"
```

客户端固定两组prompt与SHA、独立salt、P输出1/D输出2、greedy/min1/ignore_eos；将原生P metadata原样交给D，保留真实P/D请求响应和同一客户端时间。N257只追加原256输入的末token，作为块边界诊断，不是精度题。若早期因P证据缺失停止，仅报告已观察的事实；不能把未运行的第二输入或D状态写成通过。

完成、负结果、错误或期限到达，先停止自己的精确实例并保全原件。`finish.py`核对fresh UID、CID、名称、image和/out mount，仅stop该CID，前后保存probe记录/plan/源码及console；不删除预留。使用既有collector与generic backup下载SHA核验：

```sh
python3 glm5-3-vllm/tools/collect_node.py --node 32 \
  --script glm5-3-vllm/runtime-patches/state-tail-probe-v1/finish.py \
  --label "${FRESH_LABEL}_finish" --control "$SSH_CONTROL" --output "$EVIDENCE/finish32" --timeout 150 \
  --script-arg=--run-dir --script-arg="$P_RUN_DIR" --script-arg=--reason --script-arg=complete
# node33同入口，--node 33、输出finish33、run-dir为D_RUN_DIR；失败时reason改failed/negative/deadline。
python3 glm5-3-vllm/tools/backup_artifacts.py --run "$FINISH32_COLLECTED_RUN" \
  --control "$SSH_CONTROL" --backup-root "$MAC_BACKUP"
```

正常完成时，`REQUESTS_JSON`可直接用该客户端collector的`metrics.json`：它含完整请求、P metadata、D chunks和结果，已有stdout/下载SHA核验，不是摘要。若客户端在最终打印前失败，再取回`CLIENT_RUN_DIR`内progress/requests原件。NPU进程归属/退出核验后，只以finish记录的新holder UID为DeleteOptions precondition释放自己的预留并复核不存在。若下载不完整，保留失败文件，模型仍按截止时间停，不为下载延长资源；host证据可在释放NPU后继续读取。未确认CID时先取回该新run的launch/失败原件，不复用历史硬编码stop脚本或盲重启。

解包后保持两节点目录分开，避免同PID文件重名覆盖：

```sh
python3 glm5-3-vllm/runtime-patches/state-tail-probe-v1/compare.py \
  --p-dir "$P_PROBE_DIR" --d-dir "$D_PROBE_DIR" --requests "$REQUESTS_JSON"
```

比较器按实际remote_request_id、被选P rank、layer/component/position及physical block配对，要求完整的末两行component集合和writer/原生完成依据。缺项是incomplete；字节/来源不匹配或确有P MTP未执行可否定当前直接推进路径；成功仅是`sampled_rows_match`，始终`full_prefix_proved=false`。原生READ API返回、Tensor copy及算子调用仍依赖其现有契约，这不是kernel trace或完整物理前缀证明。首批6→6也不决定整请求是否少target forward；那须后续同预算/同采样完整请求实测，当前不实施。
