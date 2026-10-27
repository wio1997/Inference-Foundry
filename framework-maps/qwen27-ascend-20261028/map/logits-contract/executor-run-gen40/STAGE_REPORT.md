# gen40：短诊断 logits 处理与 watermark 源码契约，阶段完成待 root 验收

本代回答 argmax 前后哪些源码分支能够改写 logits 或 token ID。沿用 gen38 同 payload、gen27 文本不一致、gen35 SG 32 个逻辑输出 ID 的已存证据；没有新推理、服务启动、SDK 借用或性能测试。地图及 main/index 最新候选仍为 gen39，本代仅提供可显式合并的补充，不用旧地图 QA 代替新内容验收。

## 已核验的源码路径

- vLLM CompletionRequest.watermarking=True 是请求许可字段，不能单凭字段判定水印运行。当前 exact-container/image、git HEAD 原字节的 NPUModelRunner 构造 `AscendSampler()`；非 speculative `_sample` 把 InputBatch.sampling_metadata 交给该 sampler；这个 runner 文件没有 watermark 字符串。其 InputBatch 使用 `build_logitsprocs`，实际内置顺序为 MinTokens、LogitBias、MinP；自定义项来自 entry-point 插件和显式类加载器，不能据默认配置声称没有历史插件。
- 已保存的基类 Sampler.forward 先把 logits 转 float32，再 apply_logits_processors，再 sample。argmax 前可有 allowed/bad-word mask、non_argmax_invariant processor、penalties 和 thinking-budget holder。已审查零温全 greedy 分支返回早于温度缩放、argmax_invariant/MinP、top-k/top-p。MinTokens 在 min_tokens=0 或已达到最少输出数时不登记；logit-bias 无值时不登记；no_penalties=True 原函数直接返回同 logits。原请求默认与归一化仍以 gen38 为准，CPU 早返回不证明历史 metadata 正确绑定。
- 水印实现位于另一条 `vllm.v1.worker.gpu` sampler 路径。GPUWatermarkSampler._sample_random 原源码同时要求 request watermarking 与 temperature!=0；全批无启用项时直接返回普通父类随机函数。即使有配置水印 sampler，零温仍被这一 gate 排除。GPU model_runner 根据 watermark_config 才选择该实现；不能将这一 GPU 路径误写成 Ascend 历史执行路径。也没有穷尽动态 monkeypatch/第三方插件。
- SG ModelRunner.sample 在调用 sampler 前调用自己的 _preprocess_logits：更新 grammar mask，按 penalty→grammar→logit bias 顺序处理；可选 observer 另有分支。Sampler._preprocess_logits 再执行可选 custom processor，随后调用 sanitize_nan_logits，然后全 greedy argmax。两层同名函数职责不同，不能漏掉外层 penalty/mask。
- SG custom processor 的 batch gate 是 `enable_custom_logit_processor and any(request processor)`。已保存的历史 server_args 明确 enable_custom_logit_processor=False；保存请求未给 custom processor，gen38 协议默认 None。这支持所审查源码路径上 gate 为假，仍不是历史实际对象/动态绑定遥测。
- `sanitize_nan_logits` 先调用 maybe_detect_nan；仅 SGLANG_SANITIZE_NAN_LOGITS 开启时才 warn 并 nan_to_num_：NaN/−Inf→−1e30、+Inf→+1e30。当前冻结 environ.py 中 SANITIZE_NAN_LOGITS 和 ENABLE_ASYNC_ASSERT 都默认 False。函数注释的“sanitize”不等于无条件执行；历史环境值、非有限 logits 是否发生均未知。CPU 浮点列表测试只是 gate/常数检查，不是 torch dtype、溢出或设备数值等价验证。
- SG `_sync_token_ids_across_tp` 位于采样后，仅强制同步标志或 grammar 存在时对 IDs 做 TP group MIN all_reduce；并非 sum/mean 或固定 rank 广播。gen35 已观测的四 rank 输出相同不能反推这个分支运行过。历史标志、每步同步前 IDs/分布式操作仍未知。

## 验证与身份

通过 task-private transport 只读 exact historical container ID/image ID，容器恢复 sleep 状态不变；执行 python -I 标准库文件读取/git show，不 import torch/模型。新下载 22 个文件观测（含当前 mounted config，非历史 loaded hash），14 个 Python 源文件与各自 git HEAD 字节相等。合并 gen38 输入共 44 个 SHA 校验观测、40 个不同路径，其中 32 个 git HEAD 相等观测；不要把包含 mounted config 的计数叫作 Python 源文件数。receipt 保存完整 SHA、commit、container/image 和远端 aarch64。Ascend runner 与源码树是当前观察字节，不是完整历史 import/依赖清单。

26 个 CPU 检查执行精确 AST：builder 常规/pooling/speculative、watermark 开关×零/正温、custom 开关、NaN gate 与替换常数、TP 同步两 gate 组合、batch custom 双 gate、min-token 激活/完成、no-penalty 保持对象、SG penalty/grammar/bias 顺序，以及历史日志开关。运行于本机 x86_64、标准库和哨兵，不建立 aarch64/NPU/数值/质量/性能结论。17 个函数锚点的完整 SHA/行号保存在 CSV/JSON，可直接复核。

首次下载包含一个不存在的 SG logprob 文件路径，读失败后没有产生部分下载；根据真实 import 改为 utils/async_probe.py 后成功。CPU Vec fixture 首次在 generator 内使用无参 super 导致 TypeError，改为 list.__getitem__ 后 26 检查通过。这些是审计工具错误，不是技术方案失败。下载原尝试源码保留在本代目录；fixture 失败及修复另记 failed-attempts.json。

现场只做一次状态比较：124/204，原 gen34 的 124 完成记录全 JSON 未变化；snapshot reported submitted124/inflight3/pending76，而精确 set difference 为4已提交未归档，保留口径差异不归一化。发现新增失败状态：rank-124 的 lineage-gen3 审计于2026-10-26T21:24:52.743654Z启动、22:15:32.605287Z结束，操作记录PID1796355、returncode137、日志0字节；其 fullstage 前序rc0记录9,348,349 task-like rows，不能据此认定lineage成功。worker4072855仍以原start ticks510808258存活，状态为failed_inspect_live_or_partial；rc137不是OOM或技术失败原因的证明。已做一次定向只读state/log检查、保存全SHA和操作命令，不重试或停止原worker。4 masters+4 CPU workers 的 PID/start/boot/command/log 已保存；旧 SDK 与 watchdog 不在、其确切容器 exited、slot resumed。未认领/释放 performance 锁、未 signal，原8卡所有权与后台作业跨本轮继续。

## 边界与下一步

watermarking=True 目前没有源码依据可以用作输出分歧根因；SG custom gate 与 NaN/TPsync 条件已明示。历史 sampler 绑定、完整插件与环境、logits/vLLM IDs、首个不同 token、跨框架质量/greedy 等价均未闭合。不能宣称两栈历史 logits 相同或优化已采用。

下一代先读取此rank-124失败收据，按原所有者协调边界审查失败操作源码/内存与退出证据，不能因mtime或worker睡眠重启、抢占或杀原pipeline。一次检查新增 native keys，只审新增产物；无增量时，将本代源码契约/17锚点/26检查结果与 guardrail 显式并入 gen39 派生的新地图和未分 R 的 main/index 候选，做对应新内容的可搬移 SHA/浏览器 QA。其后可独立审查两栈 Qwen3.5 lm_head→logits 的 dtype/scale/softcap/quantization 路径和实际配置身份，不能以源码差异冒充数值分歧原因。实际204、最终地图QA、刷新私有仓规则/账号/heads 后 R 发布仍待办，最终接受由 root 核验。

设备端点非独占 kernel/全 stream 关键路径；HCCL 保留等待与别名去重、不跨 rank 加墙钟；跨 CANN/卡组比较非纯框架因果；所有既有性能与质量限制不变。overall_complete=false。
