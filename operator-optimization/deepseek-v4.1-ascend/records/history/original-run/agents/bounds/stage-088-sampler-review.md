# Stage 088：运行镜像中的 scheduler 与原始 rejection sampler

## 快照核验

ZCode 回执 `operator-ceiling169-v2-088-sampler.verified.json` 验证执行模型为 `deepseek-flash`，退出码文件为 `0`。`sampler-source-snapshot-status.json` 显示已有 baseline 和 stage 容器均在运行，同一镜像 ID，两个目标文件在两者内内容哈希相同，且 Docker 检查没有显示覆盖目标文件的挂载。重新计算保存的快照 SHA 与 status 一致：

- `scheduler.py`：`822179da62be4bca0619c8590082ae09123f5c4415f9cc2037be673748239edb`。
- `sampler.py`：`0bb9d224887ccb1355e16199005cc23a4afc6fa75156eca419851b7247d99750`。

快照核验说明两个容器磁盘路径的源文件一致，**不自动证明运行中的 Python 对象没有被 monkey patch**。也没有将文件与公开 commit 的完整内容做逐字节比较。

## 已确认的静态行为

1. `sampler-source-snapshot/scheduler.py:63` 起定义 `max_num_seqs`，描述为一个 iteration 处理的最大序列数；`:81` 指该参数逐 DP rank 适用。这比此前只看公开源码更直接，适用于本次镜像中该文件的静态定义，不能把它理解成所有请求贯穿整个生命周期的独占槽位。
2. `sampler-source-snapshot/sampler.py:188-194` 的原始 `RejectionSampler` 将 `metadata.max_spec_len` 传给 `rejection_sample`；`:440-445` 的原始 `rejection_sample` 按 `[batch_size, max_spec_len+1]` 分配输出。该边界只在实际调用保持这个函数实现时成立。
3. 已冻结的 `sources/vllm-workspace/vllm-ascend/vllm_ascend/patch/__init__.py:1174-1186` 说明 Ascend 存在 worker `patch_rejection_sampler.py`，可覆盖 `rejection_sample` 等函数；但这份索引并不能单独证明本次 worker 的 patch 注册、实际代码与运行分支。

## 结论边界与下一步

目前仍没有证明所有输出都由原始 vLLM rejection sampler 产生，也没有证明它的动态 `max_spec_len` 与启动配置数值之间的一致性、首 token 路径、每个输出是否按同一 scheduler iteration 计数。因而不能把 `[batch,max_spec_len+1]` 直接转成已证明的每组输出 token 上限；`oem-and-scheduling-symbolic.md` 中 `g` 仍 UNKNOWN。

已准备 `compare_sampler_patch_readonly.py`，只读对比两容器的 Ascend rejection patch、worker 注册文件和 `adapt_patch` 入口；源文件、路径、镜像或内容不一致时停止。还需阅读实际 runner、metadata 构建及客户端输出归属，才能决定 `g` 是否可以收紧。本次核验和脚本准备未启动 NPU，也未改服务代码。
