# Engram 元数据：隔离诊断接入方案，尚不执行

**建议：先补可审计安装和窗口结束检查，再考虑仅为确认入口 stride 做一次无 profiler 的隔离对照。** 现有 wrapper 和 CPU 模拟不能保证真实 16 个 worker 覆盖、graph replay 内部调用或逐调用无丢失；它也不能测 worker 排队。若目标是一次同时证明 storage 生命周期和真实 enqueue→dequeue，不值得按现有能力启动，因为没有可验证的关联入口。

## 1. 第二次诊断哪些部分可沿用

以下都只可作为**新目录中的审查后副本**，绝不修改或重跑已完成第二次诊断的固定状态文件：

- `remote_guard.py`：通过本人 stage 容器持有与既有诊断共享的 `owner-fence.lock`；严格解析 NPU PID/HBM，经验 HBM 包络需实时重算、连续观察。没有其他 PID 不等于 HBM 残留一定归属本人；异常格式、可疑占用或未知归属应按现有失败策略处理，不抢占别人的任务。
- `orchestrate_binding.py`、`watchdog_binding.py`：启动前创建独立 run UUID、先确认恢复 watchdog 及 lease；控制器退出/失败后看门狗在安全条件下重试。新运行须新容器名、新 controller reservation、独立 lease/运行记录；与之前试验仍使用**同一**容器内互斥锁，不能用两个锁。
- `start_binding_recapture.py`：可参照精确 baseline ID/image、本人 NPU PID 归属、health 200、running/waiting=0、停止前二次核验、先原子保存并复核两份身份/经验 HBM、精确诊断 ID、失败恢复；应新建副本并替换新 runtime/容器/日志路径。任何身份不符或有人占卡，不停用他人的服务。原样复用它会做高扰动采集，必须改副本。
- `recover_binding.py`、`supervise_binding_recapture.py` 的安全恢复部分：精确校验本次诊断 ID、标签及 UUID，仅停止自己的新容器；设备符合策略后恢复固定本人 baseline，并独立验证 health 200。服务失败、无完整采样、summary 出错都进入恢复；丢失双份身份或归属不清可暂缓并交人工处理。不能把启动成功当成恢复完成。

明确不能原样复用的部分：第二次 `start_binding_recapture.py` 给 `serve.sh` 追加 `--profiler-config`、开启 LD_PRELOAD 记录器及原 `instrumentation/sitecustomize.py`；后者 `_patch_profiler` 开启 stack/shape 和 record_function。`supervise_binding_recapture.py` 的 CAPTURE 调用 `profile_client.py`，后者调用 `/start_profile`、`/stop_profile`、开启 profiler gate。它们是上次研究用的高扰动链，不适合这次无 profiler 对照。不能只把 `with_stack` 改 false 就称没有 profiler。

## 2. 新容器的最小接入与 gate

以下为要实现和审查的精确设计，**当前尚不存在完整的可运行启动链**：

1. 新 controller 使用新名称（例如带全新 UUID 的 `engram-meta` 名称）和 `/data/tiankuan/wio/main` 下独立 runtime，不复用上次 `binding-diagnostic-selective2` 的 metadata、profile 或状态。仅复制冻结 baseline `scripts/config`，保留原始 `serve.sh` 参数及模型、量化、cache、并发、spec decode 设置；不追加 `--profiler-config`，不加载旧 `sitecustomize`、旧 profiler monkeypatch 或 C 版 LD_PRELOAD。启动源码及所有注入文件 hash 与 baseline 启动参数都需留存；若已有配置或 PYTHONPATH 内容与这套 bootstrap 冲突，不覆盖。
2. 新增**仅该容器使用**的独立 `sitecustomize.py` / import hook，随进程启动按模块名 `vllm_ascend.models.deepseek_v41.engram.embedding` 精确等待自然 import，不主动预载模型。模块执行完成后检查 `AscendParallelEngramEmbedding` 的真实 `__module__`、源路径/文件 SHA `a7741ed16522be4fe607e062e1c988c4e2435af7d258fe7db99f672d4508c4c9`、被包装 `embed_gathered` 的 class dict、签名及原 code identity；再调用经过审查的 `engram_metadata.install(cls)`。检查包装后 marker、`__wrapped__` 和原函数对象匹配。仅文件 hash 不能证明正在执行的 bytecode，原 code identity 必须与同镜像、同 Python 版本的已审可信参考比较；无法比较标 UNKNOWN。
3. hook 本身不能强行假设原始 import loader：原先 `_Finder` 直接调用 `PathFinder.find_spec`，可能绕过优先级更高的 finder 或替代 loader；fork/spawn、模块已先导入、动态替换类也可能绕过安装。新 hook 必须检查 loader/导入顺序并保留原有行为，且在 CPU mock 中覆盖 import 错误、重复导入、fork/spawn、目标已加载和非标准 loader。失败应记录、原模块正常运行，不能悄悄宣布成功。目前只完成 Tensor wrapper fake/CPU 测试，**未实现或验证该 bootstrap**。
4. 环境明确设置 `WIO_ENGRAM_META_ENABLE=1`、专用 `WIO_DIAG_DIR` 及 `<dir>/engram-metadata.jsonl`。禁用时不安装、不开文件；正常调用窗口外不记录输入，控制端在预热/前缀 seeding 完成后通过同目录临时文件原子 rename 建立 `PROFILE_ACTIVE`，完成所有目标响应后删除 gate。该文件只是主机观测开关，不使用 profiler，不保证恰好与 device 工作同步。窗口变化时刻与 run UUID 单独记录；gate 观察失败或跨窗口调用必须标 UNKNOWN。
5. 新客户端仅复用**冻结输入与输出计数协议**，不要调用 profile start/stop；不在日志保存请求内容或模型文本。对齐既有 benchmark 客户端的 seeding/冻结 prompts、32K/1K、并发 8、采样及 spec decode 选项和相同 token 统计；不能简单复用上次 `profile_client.py`（它必然调用 profiler 且保存文本）。新客户端/脚本先在 CPU mock 校验 gate 的创建/删除和失败清理，再由主控审查。

## 3. 每个真实 worker、graph 与丢失怎样判别

要**判别覆盖**而非假设覆盖，bootstrap 必须提供每进程一次有界安装确认（run UUID、随机进程代次、PID/TID、时间、module/code hash、`installed`/失败原因的受限枚举，不能输出环境变量或用户数据）。由自己的 container PID 与任务期明确的 rank/device 映射做一对一对照；映射仅能使用已验证来源，例如该 worker 既有日志/已初始化 group 的本地只读状态，以及 host/container PID namespace 的已验证映射。不能把 PID 大小、日志文件数量或 `LOCAL_RANK` 默认值当作实际 EP rank。若真实 rank 身份缺失/重复/解释不明，列为 UNKNOWN，不称 16/16；若出现预期 16 ranks 之外的调度进程，则只列为非 worker，不强行凑数。

每个真正进入 `embed_gathered` 的 Python 调用，需要有进程代次内单调调用号、门控开始/结束与一对 begin/end；窗口关闭时记录该 worker 的有界统计和闭窗 ack。比较观测到的调用尝试数、begin/end、缺号、异常、state-lock/log-lock 竞争、身份表淘汰、gate 未知、截断和日志 cap；结束前后校验日志大小/完成标记。每 worker 的 **ack 缺失、process crash/SIGKILL、gate 检查失败、短写、队列竞争或 cap** 都只能标不完整。需要独立保留小型 per-worker 状态记录而不能依赖可能已满的共享事件文件。

现有 `engram_metadata.py` 的 stats 只在 observer 正常退出时尝试输出；`calls_seen` 在锁获得后递增，锁竞争时没有 call ID；文件满后 stats 可能写不进去。**所以现版本不能保证每次掉落能追溯，不能直接用它证明完整。** 必须先设计并测试非阻塞、线程安全的饱和计数器和窗口结束 checkpoint，统计锁获取前进入次数、各类写入成功/失败、begin/end；保证进程终止前的 ack 与日志一致。可以评估低频 gate watcher 或其他进程内 checkpoint，但 watcher 也产生线程、文件系统开销及竞态；现有外部 `docker exec` 不能读取另一个进程里的 Python observer 状态。若无法实现/验证这些条件，就记录数据不完整，不宣称“确保”。

安装后每 worker 有确认不等于 graph replay 覆盖：Python wrapper 只证明 Python 实际进入，预热期间捕获后 replay 可能完全不再进 Python。必须分别标记预热、测量窗口的真实 Python 调用次数，并结合可验证的现有 graph 状态/配置判断；如没有独立 graph/replay 可见接口，graph 内部执行与零调用的原因均为 UNKNOWN。不能通过改变为 eager 来冒充生产路径；若另做 eager 诊断，结果只能作为不同配置的参考，不能与原路径直接相减。

## 4. 无 profiler 负载与吞吐口径

一次视图实测前后应保留无采集对照：A 为同镜像、同原始服务参数、无 profiler 和无 metadata wrapper 的独立参考；B 仅添加 wrapper/bootstrap，其他保持一致。两者都用同一套已冻结 C8、32K 输入、1K 输出、约 90% 前缀命中负载，同一 seeding/预热/停止规则，逐轮保存 8 个请求成功数、32768/1024 的 usage、prefix query/hit、generation、draft/accepted、抢占及错误计数。以原 baseline 的相同客户端、请求起止定义和统计方法计算 output tokens/s、TTFT、TPOT；如果更换客户端或窗口，则必须明确作为另一指标，不能声称直接对齐。原 baseline 的 519.84 tokens/s 是历史端到端记录，不是这次会话的即时值。

在资源允许时用交错或成对轮次记录 A、B 的分布，避免仅挑最快单次；同时记录卡占用、资源限制和可获得的频率/温度，比较实验扰动。没有足够重复或环境不匹配时只报观测与不确定性；更低的运行时间不能直接称模型优化收益。测量阶段用 gate 结束时只知道客户端请求完成，设备是否仍有异步工作不可假定，不能添加同步来制造可比性；记录何时恢复并严格遵守原服务保护。

候选是否值得：在上述安装/覆盖/丢失检查可证明且自有设备空闲、原服务可安全停放时，**值得做小范围 stride 可见性与采集扰动对照**，它能把此前 `[T,2,24] → [T,24]` 的推断变成有条件的真实入口证据；它仍不能证明可删除 copy 或可节省时间。若不能核验安装、graph 覆盖和数据完整性，就先补设计而不启动，避免重复收集无法解释的 trace。

## 已审对象哈希与验证范围

本地 `start_binding_recapture.py` SHA-256 `3d7a926b0ab982df00e7630bafe1eea46da1656d3228333b4520020dea80a38f`；`supervise_binding_recapture.py` `036b4bb48ed1db8c1f383e41f0b19f79383470754b1e5cd3b0a1bbeb9b6d73fb`；`recover_binding.py` `f8cd06cc8fa9479750fd017a062aa682720c36ff38695f24c9849602e6541b50`；`remote_guard.py` `f475694ae5f660394d60d5239fa6f2ec640d64ab09d67401827f4bfee83383d7`；待审 `engram_metadata.py` `05a2a59811bf049851ebad10e726542c9e4b4c6b940d996f4398b211947c5fd4`。CPU 测试见 `CPU-TORCH-RESULT.md`；此文不创建容器、不停止服务、不调用 NPU。
