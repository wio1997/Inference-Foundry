# 950DT 性能测试复现

## 1. 测试合同

| 输入 / 输出 | 客户端并发 | 普通组 | KV 预热组 |
| --- | --- | --- | --- |
| 32768 / 128 | 32、64、128 | 实际命中 0 | 实际复用 29440，89.84375% |
| 81920 / 600 | 16、32 | 实际命中 0 | 实际复用 73728，90% |

每个并发点使用两倍并发数的独立请求，两框架使用相同输入 token ID 和输出上限。DP8 / TP1 / EP8 / CP1，DSpark 5，上下文 98304，每卡 max-num-seqs=16。DeepInfer 名义预算 8192；固定 vLLM 版本名义 8256，扣除 4×16 个投机预留后有效预算 8192。两容器均使用 384 逻辑 CPU 配额及 1TiB 内存限制。

预热组先完成独立 HTTP 前缀请求、确认八卡排空，再发送新的完整请求。DeepInfer 通过原生序列 fork 复用，vLLM 使用原生 APC。不能以截短输入代替缓存命中，也不能把配置中的计划命中率当成实测值。

## 2. 代码与运行目录

实验目录：[reproduction/ascend950/performance-20261021](../../reproduction/ascend950/performance-20261021)。这是原实验的目录结构；启动脚本保留实际使用过的容器路径。

| 路径 | 用途 |
| --- | --- |
| `adapter/deepinfer_dp8_adapter_e27_cache.py` | 普通性能测试使用的 DeepInfer HTTP 入口 |
| `adapter/e27_prefix_cache.py` | 有限前缀父序列管理 |
| `tools/e26_ais_model.py`、`e27_ais_model.py` | 固定输入、每请求 DP 路由和缓存命名空间 |
| `tools/e29_ais_model.py` | 80K 首包的独立响应缓冲配置 |
| `scripts/remote/*e27*.sh` | 两框架实测启动参数 |
| `configs/e27/`、`configs/e29-clientfix/` | 已执行的 AISBench 配置；80K 使用 clientfix 版本 |
| `tools/e27_run_window.py`、`e29_run_window.py` | 单窗口执行、测前测后空闲检查及原始记录保存 |
| `tools/e27_review_window.py`、`e29_review_window.py` | 请求数、实际输出 ID、usage、缓存和排空复核 |
| `data/performance-summary.json` | 20 个正式窗口的汇总与比较限制 |
| `COPY_PROVENANCE.json` | 从原任务复制的文件路径与 SHA256 |

复用原实验启动器需要以下容器内布局：

```text
/work/DeepInfer/                 固定源码及部署补丁
/work/engine-venv/               DeepInfer Python 环境
/work/runtime-20261021/          JIT 缓存、通信拓扑
/work/performance-20261021/      本实验目录
/data/wio/DeepSeek-V4.1-Flash-1017/  原始模型，只读
```

DeepInfer 基础环境由 [部署脚本](../../deploy/ascend950/)准备。原 vLLM 比较容器还需将实验目录挂载到 `/work/performance-20261021`，将已构建 provider 放在 `e20-install/opp/vendors/customize`，并使用部署脚本中的固定 DSA 补丁。独立 vLLM 部署脚本采用 `/workspace/deploy` 布局；它与本节原实验布局不同，不能不调整挂载就混用两套启动器。

原启动器默认容器名是 `performance-20261021-deepinfer-runtime` 和 `performance-20261021-vllm-official-ds41`，HTTP 端口分别为 18121、18122。运行前检查脚本内容、容器挂载和网卡。每次只让一个框架占用八张 NPU。不要使用早期 16 CPU 容器配置替代本合同的 384 CPU。

## 3. 恢复相同输入

发布只保留排序后的词表 ID 集合、随机种子和原输入文件哈希。以下工具在 CPU 上生成输入，逐文件校验 SHA256，不加载模型、不发请求：

```bash
cd reproduction/ascend950/performance-20261021
python3 prepare_frozen_workloads.py --case E27-L32K-C32
python3 prepare_frozen_workloads.py --case E29-L80K-C16
python3 prepare_frozen_workloads.py --case E29-L80K-C32
```

另两个 32K 并发可用 `E27-L32K-C64`、`E27-L32K-C128`。程序拒绝覆盖已有输入。原生成器 `e27_prepare.py` 和 `e29_prepare80.py` 也保留，直接调用它们需要原 E26 seed 文件；上面的恢复工具不需要该文件。

## 4. AISBench 环境

实测 AISBench 源码提交：`e00d1f758310c45580576af291a3126919776c89`，来自 [Ascend/ais-bench](https://github.com/Ascend/ais-bench)。复用原镜像或从固定提交安装其 benchmark Python 包。请按该版本依赖建立 Python 3.12 环境；公开包未附完整 wheel 仓库。

客户端实际使用的前缀缓存插件文件保留在 `tools/ais_bench_prefix_cache/models/vllm_prefix_cache_api.py`，将 `tools` 加入 `PYTHONPATH`。原执行器通过 AISBench 容器调用 `/work/performance-20261021/e24-aisbench-venv/bin/python`。运行前核对该解释器和 `tools/e27_run_window.py` 中的容器名；新环境的路径变化需要同步调整。

## 5. 窗口顺序与验收

32K 使用 `NORMAL-R1`；缓存正式窗口使用 `HOT-R2`，之前的路径预热和独立 HTTP 前缀填充不能省略。`HOT-R1` 等首次运行数据保留为历史记录，不挑最快窗口。

以 32K C32 为例，在已完成服务初始化与输入恢复的原布局主机上：

```bash
python3 tools/e27_run_window.py --framework di --run-id E27-DI-L32K-C32-NORMAL-R1
python3 tools/e27_run_window.py --framework di --run-id E27-DI-L32K-C32-HOT-W3
python3 tools/e27_run_window.py --framework di --run-id E27-DI-L32K-C32-HOT-R2
```

首次服务需要先执行 `SHAPE-W1` 及完整缓存路径预热配置。每个窗口结束后确认实际返回 128 或 600 个输出 ID、usage 一致、无失败、八卡排空。通过 DeepInfer 每请求原生剩余 prefill 量，以及 vLLM 八卡缓存计数增量确认实际命中。80K 对应配置在 `configs/e29-clientfix/`，由 `e29_run_window.py` 执行；先阅读其 `--help` 并显式传入配置路径。

重跑时为输出目录和缓存命名空间使用新标识，不能覆盖历史正式记录。只有同一合同、测前测后状态和实际工作量均通过的窗口才可比较。

## 6. 结果边界

20 个正式窗口已经在原机器完成，2176 个请求均成功。发布后的新主机部署和整套复跑尚未执行。

原接口仍有额外响应工作差异：vLLM 首包附带输入 ID，个别请求还观察到长文本首包。现有 TPS、TTFT、TPOT 是各自原生 HTTP 路径结果，不能把整个差值都认定为 NPU 算子收益。统一输出边界的实卡对照、缓存组低干扰采集和最终因果归因尚未完成。详见 [全链路状态](FULL_CHAIN_STATUS.md)。
