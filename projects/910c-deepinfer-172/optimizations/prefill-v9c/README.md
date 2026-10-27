# 172 预填充优化 · prefill-v9c

固定 32768 输入 / 1024 输出 / 并发 8，DP2×TP8、EP16、16 个逻辑设备。
两轮均实际恢复 235904/262144 个前缀 token（89.990234375%）；16/16 完整输出与冻结 vLLM 一致。

| 指标 | 冻结 vLLM-Ascend | 冻结 DeepInfer v8 | 当前优化版 |
|---|---:|---:|---:|
| 合并输出 token/s | 518.42 | 668.35 | 737.99 |
| 平均首 token ms | 831.25 | 5334.60 | 4315.80 |
| 平均端到端 ms | 15370.60 | 11035.25 | 9861.35 |

相对 v8：首 token 下降 19.10%，合并吞吐提升 10.42%。
合并吞吐按 `16384 / (10.262384899999999 + 11.9384515)` 计算。
首 token 仍慢于 vLLM；仅两轮，保留轮间差异，不宣称一般负载下的同等收益。

## 改动

- 固定活动 slot 的页表直接复用设备视图；非连续 slot 采用有界缓存，保留顺序与重复项语义。
- 同一 forward 的位置张量复用。C1/C2 压缩坐标在模型计算前按 ratio 准备一次，KV/索引键/索引尺度共享同一写入计划。
- 命中真实前缀后不再为冷种子缓存边界额外切分；冷预热仍在两个原始边界采集真实缓存。
- 预填充分块 512→1024，当前固定输入由 8 次变为 4 次完整模型预填充。
- MC2 专用通信组 2560→5120 MiB；低并发预热按真实请求数预留容量。TP/control 组维持原配置。

Engram、目标模型计算、输入提示查找和 64-query 目标验证算法保持原有数学路径；未启用主干 CPU prefetch。

## 验证与资源边界

相关 CPU 回归 21 项通过，覆盖 2048 容量的 SWA 环、C2 奇偶起止边界和多请求物理坐标。
正式验证仍使用用户指定两轮 AISBench；[独立验收](results/independent-validation.json)逐请求重新比较完整输出并重新计算吞吐。
当前 1024 配置的 HBM 占用约 58.6/64 GiB 每逻辑设备，显存用量高于 v8；仅对本次固定容量/并发验证。

2048 分块没有被采用：2560 MiB 的 MC2 窗口先触发内核容量断言；9216 MiB 窗口后专家 GMM 又出现 OOM。
两次都在冷前缀预热阶段终止，没有正式分数；原始失败目录和 SHA 入口见 [证据索引](evidence/archives.json)。
采样中的 1.83–2.03 秒页表调用停留包含等待先前 NPU 工作，不能视作纯拷贝耗时或保证可全部消除的时间。

## 当前运行与复现

- 项目：`/data/tiankuan/wio/910c-deepinfer-172`，目标仅 172。
- 源码：`src/DeepInfer-v9`；控制目录：`serving/prefill-v9c`；原始冻结：`final/prefill-v9c`。
- 本地源码提交：`a713eca2b6ddfd5732922859a90237536619bc2e`，相对 v8 `55ae7d193425fe8f20531809ab7f9aebaee52c08` 的 [增量补丁](source/prefill-v9c.patch)。DeepInfer 官方上游未推送。
- 镜像仍固定于 `sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b`；此值为本地 image ID。
- 完整 worker/API 命令见 [运行配置](config/runtime.json)；关键 worker 参数为 `--chunk-size 1024 --mc2-buffer-mib 5120`。
- 使用原已授权的 privileged NPU 容器；模型、驱动、容器根目录挂载权限沿用 v8。
- [服务控制器](controllers/native_service_prefill_v9c_job.py)和[基准控制器](controllers/native_prefill_v9c_benchmark_job.py)保留实际执行版本。
  这些控制器有设备空闲和目录存在检查；复现时须使用新控制/日志/结果目录，避免覆盖记录。
- [原 v8 完整适配补丁](../../source/deepinfer-910c-w8a8.patch)、[原 vLLM 冻结基线](../../../910c-deepseek-v4.1-flash/README.md)保持原样。

服务在 2026-10-27T15:24:35.410985+08:00 归档检查时为 READY，并保留运行。未为归档再次推理或重启。
