# DeepInfer 在 Ascend 910C 上部署 DeepSeek V4.1 Flash W8A8

版本日期：2026-10-28  
分支：`deepinfer-dsv41-910c`  
用途：记录 910C 专用 W8A8 执行路径的适配过程，并复现已验证的部署配置。

## 1. 部署范围

本部署使用服务器 172 的 8 张物理 Ascend 910C 卡，对应逻辑设备 0–15。两个 DP 副本分别使用 TP8，路由专家采用全局 EP16，每个 DP 副本提供 4 个请求槽位。

执行入口为本分支源码中的原生 W8A8 worker 和 HTTP 接口。模型加载、缓存、调度及目标模型验证由该入口执行；底层算子使用固定镜像中的 CANN 扩展和 custom OPP。CPU 接口复用镜像中的 DeepSeek V4.1 tokenizer。

本文验证范围为贪心生成、输入 32768 token、输出 1024 token、并发 8、真实约 90% KV 命中。该专用入口不等同于上游通用服务入口，其连续批处理、其他并发和随机采样性能未在本次记录中验收。

## 2. 环境要求

| 项目 | 已验证配置 |
|---|---|
| 目标服务器 | 172；主机名 `bm-jn-zs-zone1-910C-64G-10-172` |
| 硬件 | 8 张物理 910C 卡，16 个逻辑设备；640 个可用 CPU |
| 架构 | `aarch64` |
| 镜像标签 | `quay.io/ascend/vllm-ascend:v0.30.0rc2-a3` |
| 本地镜像 ID | `sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b` |
| Python | 3.12.13，镜像内执行 |
| PyTorch / torch-npu | `2.10.0+cpu` / `2.10.0.post7` |
| Triton / Transformers | 3.6.0 / 5.14.1 |
| CANN Python 路径 | `/usr/local/Ascend/cann-9.2.0/python/site-packages` |
| 权重 | `DeepSeek-V4.1-Flash-1017-w8a8`，ModelSlim signed INT8 W8A8 |
| API | `127.0.0.1:8172`；模型名 `dsv41` |

镜像标签与本地镜像 ID 同时保留。当前镜像没有可用的 registry RepoDigest，本地镜像 ID 不能用作远端拉取地址。更换镜像、驱动或算子包后，需要重新验证兼容性。

主机内存约 2.16 TB。本版本的 Engram 原始量化表约占 202.8 GB 共享物理内存；初始化、预填充及其他 CPU 数据仍需额外空间。该表大小不代表整机最低内存要求。

## 3. 目录与文件

全部项目数据位于 `/data/tiankuan/wio`。下文使用以下路径：

```text
/data/tiankuan/wio/
├── DeepSeek-V4.1-Flash-1017-w8a8/       # 权重，只读挂载
└── 910c-deepinfer-172/
    ├── delivery/deepinfer-dsv41-910c/   # 本文所在项目目录
    ├── src/DeepInfer-allopt-mc2-cap4096-v1-20261028/
    ├── serving/                       # 控制文件与 worker 状态
    ├── workload/frozen/               # 冻结输入、预热种子、参考输出
    ├── benchmarks/                    # 按版本或复测编号独立保存
    ├── cache/                         # CANN、Triton、tokenizer 等缓存
    ├── logs/
    ├── runs/
    ├── tools/
    └── tmp/
```

交付目录内包含完整源码归档、可直接阅读的适配源码、容器元数据、原始启动控制器、两轮基准驱动和 SHA256 清单。归档不包含模型权重、镜像或编译缓存。

## 4. 适配流程

| 阶段 | 实施内容 | 核验对象 |
|---|---|---|
| 权重加载 | 对接 ModelSlim 张量名称、INT8 权重布局、scale 与量化边界；保持原 W8A8 权重 | 实际量化计算与张量分片 |
| 主干执行 | 接入 40 层模型、mHC、RoPE、路由专家及共享专家 | 完整目标模型输出 |
| 注意力与缓存 | 接入索引器、C1/C2 压缩、分页 SWA 和压缩 MLA，维护跨层缓存源与消费者 | 长上下文、压缩边界与缓存回退 |
| 并行通信 | 建立 TP8 组与全局 EP16 MC2 组，分离普通通信与 MC2 工作区 | 实际分片、通信容量和完整路由 |
| 图执行 | 保持持久输入与元数据，捕获目标模型和 DSpark 执行图 | 图重放、状态更新和输出一致性 |
| DSpark | 加载 3 层学习式草稿模块，接入置信度调度及完整目标验证 | 草稿权重、置信度头、Markov 头和接受流程 |
| Engram | 保留 CPU 表，TP 按 hash head 分片，DP 通过共享 memfd 复用物理页；NPU 直接读取选行并反量化 | 实际共享页、查表路径和完整输出 |
| 工作区调整 | MC2 容量取 `max(4096, global_bs × topk)`，覆盖完整全局路由上界 | 算子实参、NPU 内存及两轮结果 |

实现位置、关键约束和未接入项见[适配说明](ADAPTATION.md)。历史“提示查找 63 + 目标验证 64”不属于本部署的 DSpark 路径。

## 5. 获取与初始化

### 5.1 获取交付目录

在具备 Git 的管理节点获取本分支：

```bash
git clone --depth 1 --single-branch --branch deepinfer-dsv41-910c \
  https://github.com/wio1997/Inference-Foundry.git
```

将 `projects/deepinfer-dsv41-910c` 整个目录传至服务器 172 的 `/data/tiankuan/wio/910c-deepinfer-172/delivery/`。当前 172 宿主未安装 Git；服务器侧使用已传入的交付目录，无需执行 Git 命令。SSH 目标由操作人员显式指定，认证信息不包含在仓库内。

### 5.2 检查已有服务

以下命令在 172 宿主执行：

```bash
hostname
whoami
npu-smi info
curl --noproxy '*' -fsS http://127.0.0.1:8172/health
```

健康接口返回 `{"status":"ok","backend":"DeepInfer-native"}` 且版本匹配时，使用既有服务。初始化和启动步骤仅用于恢复缺失环境；不对运行中的服务重复执行，不停止其他项目进程。

### 5.3 校验并准备目录

```bash
P=/data/tiankuan/wio/910c-deepinfer-172
D="$P/delivery/deepinfer-dsv41-910c"
cd "$D"
sha256sum -c SHA256SUMS
python3 scripts/prepare.py
```

`prepare.py` 校验归档后，将源码放入固定版本目录，并恢复冻结工作负载与 AISBench 适配器。对于已有文件，脚本核对内容并拒绝覆盖不同版本；归档中的历史运行状态保存在交付记录目录，不写入现场 `runs` 或 `serving`。

权重目录须已存在并包含原始配置、tokenizer、safetensors 索引、量化参数和全部分片。DSpark 所需 `mtp.0`–`mtp.2` 权重须随原始检查点保留。本流程不转换为 FP4、FP8 或全量 BF16 权重。

### 5.4 容器准备

```bash
python3 scripts/create-containers.py
python3 scripts/create-containers.py --execute
```

第一条命令列出容器创建参数；第二条创建缺失容器，保留同名且运行中的固定镜像容器。

| 容器 | 职责 | 权限 |
|---|---|---|
| `wio-deepinfer-s172-npu` | 16 个 NPU worker | 本次部署已授权的 privileged 配置 |
| `wio-deepinfer-s172` | CPU tokenizer、HTTP 接口、基准客户端 | 非 privileged |

容器根目录只读，权重与驱动只读挂载，项目输出写入 `wio`。容器 runtime 为 `ascend`，共享内存配置为 64 GiB，memlock 不限；完整参数见 [container-manifest.json](config/container-manifest.json)。Docker 自身的镜像与元数据由宿主管理。

## 6. 服务启动

### 6.1 核对固定参数

| 参数 | 值 |
|---|---|
| DP / TP / EP | 2 / 8 / 16 |
| 每个 DP 的请求槽位 | 4 |
| HTTP 批量聚合等待窗口 | 500 ms |
| 上下文容量 | 34816 token |
| 预填充分块 | 4096 token |
| MC2 独立通信组缓冲区 | 1024 MiB |
| MC2 容量 | `max(4096, global_bs × topk)` |
| DSpark | 开启；最多 5 个草稿 token；置信度调度开启 |
| 提示查找 | 关闭：`DEEPINFER_PROMPT_LOOKUP_TOKENS=0` |
| 目标词表头 | TP 并行 |
| Engram CPU 表 | shared memfd；NPU 直接读取 |
| Engram 选行预取 / pinned staging | 关闭 |
| 主干 CPU prefetch | 关闭 |
| CPU 绑定 | 每 rank 40 CPU 切片：36 主线程、1 ACL、1 release；预留 2 个 CPU，不修改 IRQ |

主干 CPU prefetch 与 Engram CPU 表卸载是不同机制。此版本仅使用后者。所有实际环境变量和启动实参见 [start-service.py](scripts/start-service.py)。

### 6.2 执行控制器

确认设备及端口空闲，且不存在本版本已用的运行状态路径后，在宿主执行：

```bash
cd /data/tiankuan/wio/910c-deepinfer-172/delivery/deepinfer-dsv41-910c
python3 scripts/start-service.py
```

该控制器为已运行版本的原始控制器，依次启动 16 个 worker、等待全部 rank 就绪、启动 HTTP 接口并应用 CPU 绑定。控制器持续驻留；由既有终端会话或进程管理器保持其运行。

控制器拒绝复用已有状态文件，且不会清理旧目录。恢复既有失败运行时，应先核对其 PID、退出状态及设备占用，再使用独立版本目录；不得直接删除状态文件并重复启动。

工作进程通过 `PYTHONPATH` 使用交付源码，同时保留 CANN 9.2 的 Python 与 TBE 路径。该入口无需执行上游面向其他后端的全量编译，也不依赖重新编译 NVIDIA 扩展。

### 6.3 检查就绪状态

```bash
P=/data/tiankuan/wio/910c-deepinfer-172
cat "$P/runs/serving-allopt-mc2-cap4096-v1-20261028-status.json"
find "$P/serving/allopt-mc2-cap4096-v1-20261028" \
  -maxdepth 1 -name 'ready-rank*.json' | wc -l
curl --noproxy '*' -fsS http://127.0.0.1:8172/health
```

状态应为 `READY`，就绪文件数量为 16。worker 日志位于 `logs/serving-allopt-mc2-cap4096-v1-20261028-worker.log`。冷启动包含模型加载、Engram 表注册及图捕获，不计入稳态吞吐。

## 7. 请求与性能复测

### 7.1 接口请求

```bash
curl --noproxy '*' -fsS http://127.0.0.1:8172/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv41","prompt":"请简要介绍张量并行。","max_tokens":32,"temperature":0,"stream":false}'
```

该请求用于接口检查。正式性能路径使用下述固定 C8 长上下文负载；短请求结果不作为性能基线。

### 7.2 固定测试方案

AISBench 固定提交为 `7bcb6440da11bef020943b2af9ed7c1ac8362306`。现有源码、前缀缓存插件和虚拟环境分别位于项目的 `tools/aisbench-7bcb6440da11`、其 `plugins/prefix_cache` 子目录及 `tools/aisbench-venv-7bcb6440da11`。该工具环境独立于 NPU worker；交付不包含虚拟环境。缺失时须按固定提交的 requirements 重建，并使用归档中的适配器及插件，不能直接替换为最新 AISBench。

每轮先加载该轮的前缀种子，再执行 C8、输出 32 token 的预热，随后进行一次正式测试。正式轮次参数如下：

| 项目 | 值 |
|---|---|
| 请求数 / 并发 | 8 / 8 |
| 每请求输入 / 输出 | 32768 / 1024 token |
| 解码 | `temperature=0`，`ignore_eos=true` |
| 正式轮次 | 2；使用两组独立冻结输入 |
| 每轮实际 KV 命中 | 235904 / 262144 = 89.990234375% |
| AISBench | `--mode perf --num-warmups 0 --num-prompts 8 --max-num-workers 1` |

两轮均有规定预热。真实 KV 命中由服务计数器验证；CPU tokenizer 缓存单独记录，不计为 KV 命中。

复测使用新的编号保存结果：

```bash
python3 scripts/prepare-retest.py --run-id retest-20261028-01
python3 /data/tiankuan/wio/910c-deepinfer-172/tools/retest-20261028-01-job.py
```

准备脚本复用已验证的驱动与两个配置，仅修改输出目录、绑定当前服务控制目录，并移除对历史容量对照结果的读取。它不启动模型、不运行测试，也不覆盖已有复测编号。运行脚本需要上述 AISBench 工具环境已存在。

正式验收要求两轮各 8 个请求成功、每请求完整输出 1024 token，并与冻结参考全文一致，同时核对实际执行路径和缓存计数。合并吞吐按两轮总输出 token 除以两轮总耗时计算，不取两轮吞吐的算术平均。结果与指标解释见[测试结果](RESULTS.md)。

## 8. 常见故障

| 现象 | 已知原因与处理 |
|---|---|
| ACL 初始化报 `507899` 或 `Conflict open udevid` | 先检查宿主设备占用与容器设备访问配置。本项目普通容器预检未通过，NPU worker 使用已授权的 privileged 容器 |
| 算子已注册但无法找到 tiling 或 custom OPP | `dsa910c.load_ops()` 须在首次 NPU 初始化之前执行；保留原始入口的加载顺序 |
| CANN Python 模块或 TBE 导入失败 | 检查控制器中的 CANN 9.2 PYTHONPATH，不能只保留项目源码路径 |
| HCCL 初始化期间 `halSqCqAllocate drvRetCode=17` | 曾发生于模型加载前的队列资源分配；保留失败日志，核对自有进程是否退出、设备上下文是否释放。此错误本身不能证明 MC2 容量不足 |
| 图捕获时发生 CPU 到 NPU 的动态分配或复制 | 保持图输入、常量及元数据的持久地址；修改形状后重新验证图重放 |
| Engram 内存未被两个 DP 共享 | 检查 shared memfd 的 `MAP_SHARED` 映射及进程 smaps。仅观察地址或文件名相同不足以证明物理页共享 |
| 控制器拒绝启动 | 检查设备、8172 端口及版本状态文件；保留历史目录，先确认归属与运行状态 |

## 9. 已知限制

当前部署仍有 CPU 侧 Engram 历史和 hash 计算、图入口前查表依赖及文件控制交互，尚未完成全流程 ZOS。共享专家与路由专家重叠、更多图形状、完整有界重放、lazy mHC 跨子层融合等优化仍需适配或验证。NVIDIA 专用内核和多机生产部署参数不能直接作为本机已支持项。

本次仅发布已测部署版本。单点探针通过但未接入完整服务的代码不计入已部署功能；测试收益及其适用范围按 [RESULTS.md](RESULTS.md) 解释。
