# 169 重采样：ZCode 接口与旧诊断启动审查（离线）

本文件只根据本机旧脚本和历史输出整理；没有连接 169、查询当前占卡或启停容器。历史状态不能当成当前状态。**没有一个旧脚本可直接安全重跑**；以下是供主控审查和创建新脚本的计划，不是已执行命令。

## ZCode 的已有调用链

- `/Users/wio/work/173-sglang-matched-172-benchmark/run_verified.py --stage <新阶段名> --timeout <秒>` 从同目录 `<新阶段名>.task.txt` 读取任务，调用 `/Users/wio/work/tianjin910c-access-check/run_zcode.py --task-file ... --cwd ... --mode yolo --tool-profile build --timeout ...`。后者只保留 ZCode 配置中的 DeepSeek provider，设置临时私有配置，调用本机 ZCode CLI；不打印、不复制 provider 配置或凭据。用户授权 ZCode 承担机械执行，GPT 主控先审查确定性脚本及任务文本。
- 验证器 `verify_zcode.py` 从对应 session rollout 的 modelId 检查是否只有 `deepseek-flash`。需同时核查 ZCode 进程退出状态、明确的动作 exit-code、结构化状态和模型验证；`model_verified` 不证明服务器动作成功。应使用从未存在的唯一 stage / task 名称。
- **覆盖风险**：`run_verified.py` 对阶段 `.zcode.json`、`.evidence.txt`、`.verified.json` 使用覆盖写，并删除对应 `.exit-code`，不能拿旧 stage 或 `--task` 指向旧任务。验证器会保存 tool 输出与模型 response，任务脚本必须严格只输出白名单状态/哈希，避免输出环境变量、连接详情、密钥、请求正文或回复原文；尤其不要把 `docker inspect` 或服务日志原样输出。
- 旧任务使用有固定绝对路径、独立 exit-code trap 的 `.local.sh`，task 明确只运行审核脚本一次，禁止自主修改、重试或追加测试。新阶段同样需采用新文件，且先检查 stage/task/local.sh 及所有结果路径均不存在，再创建；不得复用已有历史证据名。

## 旧启动和恢复脚本的实际行为

- `prepare_diagnostic.py` 在指定旧目录及本项目搜索中未找到；准备逻辑嵌在 `start_diagnostic.py` 的 `prepare` 字符串内。其要求旧 `diagnostic/runtime` 不存在，复制原 runtime 的 `scripts/config`，以 `cp -a --reflink=auto` 复制原 cache，创建新 logs/home/tmp/traces；对原 serve 脚本 SHA 进行固定断言，再向独立副本仅追加 `--profiler-config`。旧诊断使用与基线相同镜像、privileged、只读 rootfs、bridge 网络、单独 IPC、绑定本机端口；对基线只读挂载进行继承，明确替换独立 runtime、tmp 的可写挂载，模型只读。
- 该脚本固定旧基线 ID、旧 staging、旧诊断名称和 `analysis/operator-ceiling-20261028/diagnostic/runtime`，且原目录/旧诊断早已存在或需要现场核验；状态文件、baseline 日志快照等为固定名字并可覆盖。**不得重跑该脚本**，不得删除旧目录以使断言通过。`restore_baseline.py` 固定旧诊断/基线，并覆盖旧 `baseline-restore.json`；`run_profile_client.py` 固定旧诊断容器 ID 和旧 client 路径；不能直接用于新采样。
- 启动前旧脚本以容器 pid 和 npu-smi 差集检查其他占卡，并检查 health 与 running/waiting 两种请求指标均为 0；仅停放固定 ID 的本人 baseline。停放后复查 NPU，再启动诊断。危险边界：停放后如发现其他 PID，旧代码 `raise SystemExit(0)` **不会被 `except Exception` 捕获**，baseline 会留在停放状态；不能依赖异常回滚。NPU PID 解析也依赖当时格式，不能把未识别行自动当空闲。
- 旧 profiler 配置为 torch、with_stack=true、with_memory=false、ignore_frontend=true、max_iterations=8，采集输出在该诊断副本。旧 profile 客户端从只读冻结 benchmark 目录 seed prefix，再开启 profiler 发请求，保存 metrics 与响应数据，采样结束 stop_profile；它本身不能保证本次将匿名请求、prefill chunk、动态长度、逐事件或 tensor 指针绑定。它还保存预测文本/raw SSE，若不需要则新客户端应只落盘匿名统计和时间、避免收集内容。

## 安全重用的最小新方案（设计，未执行）

1. **预备和唯一命名**：在 169 的 `/data/tiankuan/wio/main/` 下为本次独立创建新 run 根、runtime、logs、cache、traces、结果及唯一诊断容器名。每处先 lstat/路径验证与独占创建，已存在即停止；禁止覆盖或清理历史目录。把执行脚本和计划的 SHA、模型镜像、基线启动参数、必要挂载、实际输出范围以不含凭据的方式记录至新 run。复制只在自有 staging 容器中进行；模型/驱动等继承只读，缓存复制独立，检查真实 mount 及 nested overmount，不能依赖路径字符串误判隔离。文件内容/容器环境不直接打印。
2. **完整只读预检**：重新确认目标为 169、本人 baseline/staging 的完整容器 ID 和镜像、唯一端口/容器名与目的目录；校验模型和输入只读绑定与原本启动参数（profiler 参数除外）。运行/等待请求和卡的实际占用必须可可靠识别：有其他人的卡任务、请求未清空、进程归属不明或查询解析不完整就不开始；只对本人 baseline 停放，绝不停止其他容器。采集目标、容量及最小工作量需先固定，不应假定旧 profiler 默认能给出新所需数据。
3. **停放与启动**：先建好独立诊断但不启动；在停放前再次确认本人 baseline ID、请求为空和卡占用，完成并持久化状态日志。停放本人 baseline 后再次检查所有目标卡无非本任务进程，若发现他人/归属不明则不启动诊断，也不挤占对方恢复 baseline；按真实状态记录“baseline 已停放、等待恢复”。若准备/启动出错，确保显式处理 `BaseException`、超时和中断：停止仅新诊断，重新核查卡与 baseline ID；空闲可恢复本人 baseline，有人占用则暂停并告知，不能盲目重启。异步 watchdog/主控恢复路径需经独立审核，不能只依赖可能被 SIGKILL 的进程内 finally。
4. **就绪与采集**：诊断启动后检查自身 ID、镜像、健康及同卡无其他人，再按预先审核的 client 在隔离副本采集。记录选定 profiler 配置、观测范围、实际 prefix/query 统计、请求/调度匿名关联、采集起止；不让脚本输出请求内容。读取/导出文件只进新目录。若需要的新字段不是旧 profiler 提供的，先核对可用接口并给出缺失，不把旧 trace 误当完整数据；用户本轮允许重新诊断，但不能根据摘要伪造动态值。
5. **恢复**：结束或失败均仅停放本次诊断容器并等待进程释放；核验 baseline ID、卡和归属并重启本人 baseline，确认 health 和请求指标、记录启动状态。若出现他人占用，不抢占卡而保留待恢复状态，主控及时报告；保留所有旧容器和目录。全程不操作 172、不改目标模型/权重/框架源码。

## 复核哈希

- `start_diagnostic.py` `0bfd18d2e92653afaf2a829d7a1fb985027967a72911d3e1aab3a7965888c463`；`restore_baseline.py` `60fd838e6f4a0bac168d874851fbebab0887d043b5e7cd179f8a7a772c07d266`；`run_profile_client.py` `9b35f06edf36fa7091434e246a434abbe910f84b3dcf9e1990396c14cafdda1b`；`profile_client.py` `48b0111a80edd2d616d0d41e56384d57709b095196bce7ce5bf7a3bcf9fd7434`。
- `run_zcode.py` `98599eed6c7736ef68fb6f49ab939bd9ff56c75076fb31036a0b082d10aa1132`；`run_verified.py` `82691156c1e85a22c7a14cb87bda9b0c2bb421756eed9196b2a3ae22f7b925e8`；`verify_zcode.py` `8dc74c019d04ad4fad6c5f83c646aad3f5c1eea0a121fded14f099a21bbe55a0`。
