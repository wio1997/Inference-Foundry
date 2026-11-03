# 169 DeepSeek V4.1 Flash W8A8：研究目录与新对话交接

> **新目标续作（2026-10-30）：** 当前主报告为 [publication-draft/RESEARCH-REPORT.md](publication-draft/RESEARCH-REPORT.md)，新增证据与 F33 详见 [continuation-20261030/CONTINUATION-ANALYSIS.md](continuation-20261030/CONTINUATION-ANALYSIS.md)。本轮有实质追加分析与采集，历史阶段发布不作为验收。

> **交接时用户指令：任务未完成，转新对话以目标模式继续。** 此处历史“已完成/交付”不代表本次验收；新目标按[逐项审计](continuation-20261030/REQUIREMENTS-AUDIT.md)核对，来源要求保留在 [CONTINUATION-STATUS.md](CONTINUATION-STATUS.md)。

这是本次研究的总入口。**当前对象是 169 / 910C / W8A8 / vLLM-Ascend**。目录名中的 `reproduce-172`、`source172` 表示历史来源，不表示下一步要连接 172；仓库中另一个 W4A8/910B3/c12 项目的 Current、PID、任务与规则不能套用到这里。

本次环境只读快照：**2026-10-30 18:12（Asia/Shanghai）**。当时原服务 running、health=200，两个 DP 引擎 running/waiting 均为 0；这不是未来占用情况的保证。交接整理没有重启服务或运行新推理负载。

## 1. 新对话先读这几份

| 顺序 | 文件 | 用途 |
|---|---|---|
| 1 | 本文件 | 定位环境、数据和已有工作，避免重跑。 |
| 2 | [RESEARCH-HANDOFF.md](RESEARCH-HANDOFF.md) | 已完成什么、主要发现、未知和后续边界。 |
| 3 | [publication-draft/README.md](publication-draft/README.md) | 完整研究报告；先读正文，历史折叠附录按需查。 |
| 4 | [ENVIRONMENT-SNAPSHOT-20261030.json](ENVIRONMENT-SNAPSHOT-20261030.json) | 容器身份、挂载、实际 serve.sh、源码 commit、版本、健康和 NPU 快照。只在私有交接目录保存。 |
| 5 | [STATE.json](STATE.json) | 本轮状态、四个旧 Task 的入口及已发布记录；历史状态不能覆盖最新用户目标。 |
| 结构化目录 | [RESEARCH-CATALOG.json](RESEARCH-CATALOG.json) | 重要文件、目录、位置映射及本机文件哈希。 |

研究报告已交付，未实现候选优化，未证明全机不可超越的数值 TPS 上限。不要因历史 Task 的 `active_loop` 或早期 E0 标签就重做已完成的采集，也不要把一次 profile 当作正式提速。

## 2. 本机、服务器和容器路径对照

以下定义 `P` 为服务器项目根，`R` 为相对研究目录；后文 `P/R` 表示拼接后的实际路径。

```text
本机项目根：
/Users/wio/work/169-vllm-ascend-reproduce-172

本机研究根（本文件所在目录）：
/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41

P = 169 宿主机项目根：
/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-169-20261028

R = 相对研究目录：
analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41

169 宿主机研究根 = P/R：
/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41

stage 容器研究根（宿主机 /data/tiankuan/wio/main 映射为 /work）：
/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41

常驻服务容器 APP：
/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash

APP 对应的宿主机目录：
P/runtime
```

**常驻服务容器没有直接挂载整个研究根。** 读取报告、原始 profiler 和导出表优先用 stage 容器的 `/work/...`；服务的脚本、缓存、日志则在 APP 中，对应宿主机 `P/runtime`。旧大 trace 和新采集主要在 169，不要假定 Mac 有完整副本。

## 3. 环境信息与复现入口

| 项目 | 本次已核对值 |
|---|---|
| 目标服务器 | `172.16.10.169`；本次不操作 172。 |
| 硬件 | 项目标识 Ascend 910C/A3；npu-smi 显示 Ascend910，NPU 编号 0–7 各有 2 个芯片，16 个逻辑芯片。每芯片 HBM 容量显示 `65536 MB`。型号/硬容量证据限制见报告，不据此推算严格峰值。 |
| CPU / 内存可见值 | aarch64，640 个逻辑 CPU；`/proc/meminfo` 的 MemTotal 为 `2111190676 kB`（约 1.97 TiB）。不等同于本进程可独占资源。 |
| 内核 / Python | `5.10.0-216.0.0.115.oe2203sp4.aarch64` / Python `3.12.13`。 |
| 常驻服务容器 | `wio-vllm-v41-repro169-20261028`，ID `b00f4d2018e3d495d876b6e396ba9579e78b47a97111b99c27d6678e31039092`。 |
| stage 分析容器 | `wio-vllm-v41-repro169-stage-20261028`；用于文件读写与 CPU 离线分析。 |
| 已结束诊断容器 | `wio-vllm-v41-official-20261030T084726Z-cb252d1d`；快照状态 exited，不能误当常驻服务。 |
| 镜像实际 ID | `sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b`；历史来源 tag 为 `quay.io/ascend/vllm-ascend:v0.30.0rc2-a3`，复现以 ID 为准。 |
| 软件 | vLLM `0.30.0+empty`；vLLM-Ascend `0.30.0rc2`；torch `2.10.0+cpu`；torch_npu `2.10.0.post7`；CANN 9.2.0；npu-smi 25.5.6。 |
| CPU 采样器 | 独立 py-spy 0.4.1 二进制；未安装进全局 Python 环境。位置见下。 |
| vLLM 源码 | 容器 `/vllm-workspace/vllm`，HEAD `ced6857afa0ea7b2e3f0846a62e1394e90f15607`；快照 git status 为空。 |
| Ascend 源码 | 容器 `/vllm-workspace/vllm-ascend`，HEAD `8fde6d4e335fe300e2b949aab0e31659cce0811d`；快照 git status 为空。 |
| 权重 | 宿主机及服务容器同路径 `/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8`，只读挂载。权重未包含在 GitHub 报告中。 |
| 服务启动脚本 | APP 下 `scripts/serve.sh`；宿主机 `P/runtime/scripts/serve.sh`；另读取 `config/serve.env`。 |
| serve.sh SHA256 | `c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba`；快照含实际完整文本。 |
| 服务端口 | **服务容器内 `8172`；宿主机 `127.0.0.1:30075` 映射到容器 `8172`**。服务名 `dsv41`。不要假定宿主机 8172 或 Mac localhost 可直接访问。 |
| 常驻容器模式 | privileged=true、只读 rootfs、bridge、private IPC、shm-size=256 GiB；APP、其 tmp 绑定目录可写。 |
| stage 模式 | privileged=true、只读 rootfs、network=none、private IPC；`/work` 可写。 |

常驻服务本身也是 bridge/private IPC；诊断容器的隔离命名空间不表示本次把原服务新改成了这两个模式。`serve.sh` 内有早期“eager/no speculative”注释，**以实际 exec 命令为准**，不能按过期注释恢复配置。

### 实际运行配置（不是建议重新调参）

- DP2 × TP8，启用 expert parallel；16 个逻辑芯片。具体通信组以同次运行证据核对。
- `--quantization ascend`，W8A8 动态量化，其他模块保留混合 BF16/FP32。
- `max-model-len=34816`，`max-num-batched-tokens=8192`，`max-num-seqs=4`，block-size=128，gpu-memory-utilization=0.9，seed=0。
- Engram `cpu_offload=true`、`dp_shared_memory=true`。
- `--async-scheduling`、CPU binding、fused MC2、DSA CP、FlashComm1、shared-expert DP 已开启。
- NPU graph ex 开启、static kernel 关闭；graph mode `FULL_DECODE_ONLY`，capture sizes `[1,2,4,8,16,24]`，上限 24。
- DSpark `num_speculative_tokens=5`、草稿侧 `enforce_eager=true`；不是 64 层结构改动，也不是取消主模型验证。
- HF/Transformers offline；缓存位于 APP/cache。HCCL_BUFFSIZE=1024、HCCL_OP_EXPANSION_MODE=AIV。

完整参数以环境快照中的 `runtime.serve_script.text` 为准。不要把上述已启用功能重新算作新增优化机会。

### 本机连接与 ZCode

- 工作中的 169 SSH 入口：`/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh`。目录虽叫 sglang，这里只复用连接脚本。
- 该入口使用本机 Keychain 与 sshpass，经现有跳板连接；**不需要把密码或旧 pem 文件粘到新对话**。换机器后仍需具备相应凭据，本文没有携带凭据。
- ZCode runner：`/Users/wio/work/tianjin910c-access-check/run_zcode.py`；本项目使用本机 ZCode 的 DeepSeek。简单启动、离线导出、机械提取交给它，主控核验实际产物和退出码；不要全交给它裁决根因或理论界。

```bash
# 已审核机械任务的既有调用形式；任务文件要限定具体动作。
python3 /Users/wio/work/tianjin910c-access-check/run_zcode.py \
  --task-file /absolute/path/to/task.txt \
  --cwd /absolute/path/to/task-directory \
  --mode yolo --tool-profile exec-only --timeout 300
```

只有本人容器及 `wio/main` 的任务目录可写；不安装宿主软件、不停止他人任务。需要设备实验时先核对实际占用/归属，**本人常驻服务的显存占用不是他人占用**。

## 4. 研究内容目录

以下链接相对于本机研究根；服务器研究根使用相同相对目录，原始数据位置另列。

| 内容 | 入口 | 说明 |
|---|---|---|
| 全面研究报告 | [publication-draft/README.md](publication-draft/README.md) | 场景、三层采集、初析、文献复析、候选和未知。 |
| 一手资料库 | [publication-draft/REFERENCES.md](publication-draft/REFERENCES.md) | 模型、硬件、框架、MoE/TopK/attention、异步、I/O 理论。 |
| 融合/拆分重组 | [publication-draft/FUSION-INVENTORY.md](publication-draft/FUSION-INVENTORY.md) | 32 主编号及派生方向；条件性候选，不是 32 个已验证加速。 |
| 新采集算子分析 | [diagnostic-20261029/official-analysis/OPERATORS.md](diagnostic-20261029/official-analysis/OPERATORS.md) | 两窗×16 rank、类型/完整形状组、时间线及范围。 |
| 新负载/粗 NPU 负载 | [diagnostic-20261029/official-analysis/WORKLOAD-FRAMEWORK.md](diagnostic-20261029/official-analysis/WORKLOAD-FRAMEWORK.md) | 场景验证、时延、约 2 s 实际间距的 npu-smi 数据；前端 trace 为空范围。 |
| 实际 EngineCore CPU 采样 | [diagnostic-20261029/official-analysis/FRAMEWORK-CPU.md](diagnostic-20261029/official-analysis/FRAMEWORK-CPU.md) | 后补真实双引擎栈与 CPU ticks；已有异步队列的回复路径。 |
| 40 层历史对照 | [captured-layer-attribution/interior-robustness.json](captured-layer-attribution/interior-robustness.json) | 16 rank×3 组；层 0 等异常对照，不是完整请求或硬件利用率。 |
| 主层整体类型对照 | [agents/diagnosis/target-fullspan-types.json](agents/diagnosis/target-fullspan-types.json) | 旧主流 40 层跨度口径，与新窗所有流口径分开。 |
| DSpark 历史严格子集 | [draft-mc2-analysis/strict_draft_calls.csv](draft-mc2-analysis/strict_draft_calls.csv) | 162 个归属窗口；不要当作全部草稿或每请求调用数。 |
| MoE / Indexer / 投影的源码研究 | [agents/fusion](agents/fusion)、[agents/diagnosis](agents/diagnosis)、[agents/bounds](agents/bounds) | 按报告引文定向打开；大量历史审计不默认通读或重跑。 |
| 单组投影隔离对照 | [diagnostic-20261029/single-group-projection-diagnostic.json](diagnostic-20261029/single-group-projection-diagnostic.json) | 合成 BF16 eager；有漂移、非逐位一致，不代表模型提速。 |
| 冻结源码副本 | [sources/vllm-workspace](sources/vllm-workspace) | 已提取文件，并非保证是完整可构建仓库；实际运行源码在容器。 |
| 模型主代码 | [model.py](sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py) | CP/投影/模块关系的源码入口。 |
| MoE 复合内核 | [dispatch_ffn_combine_kernel.hpp](sources/vllm-workspace/vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp) | 已有组级重叠、后续交接和 workspace 别名分析。 |
| 图与候选 | [graphs](graphs)、[candidates](candidates)、[fusion-scan.json](fusion-scan.json)、[层覆盖](agents/fusion/LAYER-COVERAGE.md) | 结构枚举不是语义等价/全局穷尽证明。 |
| 条件/符号理论模型 | [bounds](bounds)、[calculations](calculations) | 名义硬件能力与未知量分开；差距不等于可兑现提速。 |
| 四个旧 Task | [task-loop-runs](task-loop-runs) | fusion-coverage、layer0-diagnosis、theoretical-ceiling、ideal-scheduler-model；先读对应 resume-pack。 |
| 同步/发布凭据链 | [publication-record.json](publication-draft/publication-record.json)、[phase-sync-receipt.json](diagnostic-20261029/phase-sync-receipt.json) | Git blob、分支和文件 SHA 校验；不是登录凭据。 |

## 5. 原始实验与数据在哪里

### A. 新官方设备采集

Run ID：`official-20261030T084726Z-cb252d1d`。

```text
169 宿主机：
P/R/diagnostic-20261029/official-20261030T084726Z-cb252d1d/
  runtime/official-traces/             # 32 份原始 ascend_pt 目录
  analysis/operator-export/copies/     # 32 份官方离线分析副本
    <rank_window>_ascend_pt/ASCEND_PROFILER_OUTPUT/
      kernel_details.csv
      operator_details.csv
      op_statistic.csv
      step_trace_time.csv
      trace_view.json
  runtime/client-warmup/               # 私有请求/响应、metrics、窗口状态
  runtime/client-baseline/
  runtime/client-profile/
  npu-smi-samples.json
  comparison-validation.json
  runtime-versions.json
  controller-state.json
```

2026-10-30 交接检查确认原始目录数 32、导出 kernel CSV 数 32。Mac 上主要有分析摘要/脚本，不默认有这整批大文件。stage 内把 `P` 换成 `/work/vllm-ascend-reproduce-172-on-169-20261028` 即可访问。

本机解析结果在 `diagnostic-20261029/official-analysis/`：`operators.json`、`operator-shapes.json`（全 220 组）、`workload-framework.json`。公开版本选择/脱敏后放在 publication-draft。

### B. 新 EngineCore CPU 采样

Run ID：`framework-20261030T091949Z-3d9bcf87`。

```text
169 宿主机：P/runtime/logs/framework-20261030T091949Z-3d9bcf87/
服务容器：APP/logs/framework-20261030T091949Z-3d9bcf87/
stage：/work/vllm-ascend-reproduce-172-on-169-20261028/runtime/logs/framework-20261030T091949Z-3d9bcf87/
```

内含 `pyspy-DP0.speedscope.json`、`pyspy-DP1.speedscope.json`、`enginecore-proc.json`、`measure-results.json`、`measure-counters.json`、`sampler-windows.json`、`result.json`、`FINAL_HEALTH.json` 与 stdout/stderr。本机解析摘要为 `diagnostic-20261029/official-analysis/framework-cpu.json`；公开摘要在 `publication-draft/framework-cpu.json`。

py-spy 二进制：研究根 `tools/py-spy-0.4.1-aarch64`；服务容器 `APP/tools/py-spy-0.4.1-aarch64-972918b5`。ELF SHA256 为 `972918b5d6af92649ec086ffdd89cefa0d725fd14844434c766f58494a1c4f8e`。

### C. 历史基线、重 profiler 与冻结输入

- 无采集旧基线：本机 `evidence/baseline-result.json`；旧聚合约 519.84 token/s 使用 AISBench 窗口，不能和新完整客户端窗口直接判回退。
- 2026-10-28 原始 profiler：169 `P/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces/`；文件索引为本机 [profile-index.json](profile-index.json)。
- 2026-10-29 的补采/失败探针：本机 `diagnostic-20261029/` 下历史子目录；先按报告定位，**不要看到脚本就执行**。
- 冻结负载：169 `P/runtime/benchmark-20261028/workload/frozen/{warmup,measure1,measure2}`；服务容器 `APP/benchmark-20261028/workload/frozen/`。
- 本机负载/旧复现记录：`/Users/wio/work/169-vllm-ascend-reproduce-172/benchmark-20261028/`、`source172/`、`evidence/`。私有输入/响应不上传 GitHub。

## 6. 如何复算；哪些入口会真的启动实验

下面四条只核对本机公开摘要算术，不连接服务器、不启动推理：

```bash
cd /Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/publication-draft
python3 verify_summary.py
python3 verify_official_capture.py
python3 verify_official_operators.py
python3 verify_framework_cpu.py
```

原始数据的 CPU 离线分析脚本在 `diagnostic-20261029/official-analysis/`：

| 类型 | 既有入口 | 注意 |
|---|---|---|
| 官方离线导出 | `export_one.py`、`run_export_batch.py` | 已完成；脚本要求目标副本不存在，不是可随意重复的健康检查。 |
| 类型/时间线 | `analyze_remote.py`、`run_analyze_remote.py` | 读已有 CSV，注意模型/流/窗口口径。 |
| 完整形状分组 | `analyze_shapes_remote.py`、`run_analyze_shapes.py` | 读相同 32 CSV，输出整数 ns 直方图。 |
| 客户端/粗负载 | `workload_review_extract.py` | 读已保存请求时间与 npu-smi；不发送请求。 |
| EngineCore CPU | `analyze_framework_cpu.py`、`run_framework_cpu_analysis.py` | 读已保存 speedscope/proc；不启动设备。 |
| **真实启动入口** | `../launch_official_profile.py --execute`、`enginecore_framework_controller.py --execute` | 会切换诊断容器或发送真实负载；交接读取时不运行。没有未完成的同名 run 需要“补启动”。 |

理论/语义缺口保留为未知：真实 R/A、HBM 字节、完整暴露通信、同次 CPU↔NPU 因果对齐、候选精度/收益。复算脚本不能证明这些，也不应为填齐未知继续无边界循环。

## 7. 当前结果与下一阶段入口

- 本次无 profiler 参考：8192 输出 token / 16.004567 s = **511.854 聚合输出 token/s**；客户端首个非空文本均值 924.713 ms。
- 优先机会：初始大行数 MoE 内部工作/交接、Indexer token-K 特化、单组输出投影后端、Q_b head 分片前移/权重复用、已有异步 batch queue 的回复边界。
- CPU 业务期主线程各约 0.98 核，整次采样回复等待栈占比高；不是调度浪费或可删时间的直接证明。
- 候选优化未实现，真实模型精度/收益未验证，严格全机数值吞吐上界未证明。下一阶段可选具体候选继续分析或开发，不重跑整个扫描/采集流程。

研究报告与 12 个主要文件的发布 commit：`9e2870bb9982d83377b180f9523adadf8e5343bd`。目录及环境表述补充 commit：`309467049f06e3b554bc365286742a09bdedf723`。仓库 [Inference-Foundry](https://github.com/wio1997/Inference-Foundry)，分支 `analysis/ascend-910c-performance`，报告位于 `docs/performance-analysis/ascend-910c-vllm/README.md`。公开目录是 [INDEX.md](https://github.com/wio1997/Inference-Foundry/blob/analysis/ascend-910c-performance/docs/performance-analysis/ascend-910c-vllm/INDEX.md)。机器连接、内部挂载和完整环境快照留在本机/本人服务器目录。
