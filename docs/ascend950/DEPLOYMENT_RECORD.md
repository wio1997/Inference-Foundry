# 8×Ascend 950DT 部署 DeepSeek-V4.1-Flash-1017

> 记录日期：2026-10-21，北京时间。后端 DeepInfer，协议适配 deepseek-recipe。
> 这是从下载到真实中文输出的部署手册，同时提供固定版本、补丁、脚本及验收证据。

## 阅读入口与结果

| 需要做的事 | 阅读位置 |
|---|---|
| 从空目录重新准备 | [环境与目录](#环境与目录)、[下载](#下载模型源码与依赖)、[构建](#在不占用-npu-的容器内构建) |
| 启动八卡模型和 HTTP API | [拓扑](#生成任务专用拓扑)、[算子验收](#先做实卡算子验收)、[启动](#启动模型和协议-api) |
| 定位构建或运行错误 | [问题与修复](#本次解决的问题)、[详细排障](TROUBLESHOOTING.md) |
| 了解 CPU | [鲲鹏 950 与上一代](KUNPENG950.md) |
| 查版本与补丁 | [脚本与锁文件](../../deploy/ascend950/)、[验收快照](../../deploy/ascend950/evidence/validated-deployment.json) |

本次已经做到：

- 完整下载并验证模型：84 个文件、48 个 safetensors 分片，510,311,045,870 字节，84/84 SHA256 通过。
- 8 张 950DT 加载真实模型，完成前向、显存测量和中文生成。
- 数字题返回 `2`；中文推理题正确得到 `3×12−8=28`。
- 两轮共四个请求，每个请求八个 DP rank 的输出 token SHA256 一致。
- FlashMLA 4 组、DeepGEMM scale layout 3 组、候选 TopK 5 组正确性检查通过；Engram 八 rank 的 FP8/BF16 数据与缩放因子逐位一致。

这次运行使用 **Attention DP8 / MoE EP8 / CP1**，并非 TP8。当前 DeepInfer 固定版本提供 DP、CP、EP 和 Engram 并行，没有已确认的 Attention/dense TP8 实现。DP 与 EP 在不同模型部分复用同一组八卡，不是 64 卡。

当前 HTTP bridge 是文本、非流式、单请求功能入口：同一请求发送到八个 DP rank，由 rank0 返回答案。它证明功能链路可运行，不能代表八个独立请求的服务吞吐。生产 admission、独立 DP 请求调度、首 token 流式观测和正式性能测试另行验收。

### 本手册中的两类证据

1. **历史实卡结果**：随附 JSON 来自此次实际部署，包含正确性、版本和生成结果。
2. **整理后的复现工具**：将历史命令改为通用路径，新增下载锁文件、启动检查和拓扑生成入口。已做语法、补丁重放及拓扑结果核验；未在另一台空服务器上重新执行完整 510 GB 下载与 NPU 部署。

新部署须执行后文的设备检查和生成验收。重新构建所得 image ID、扩展二进制哈希可能不同，应保存自己的记录。

## 环境与目录

### 实际环境

| 项目 | 本次记录 |
|---|---|
| NPU | 8×Ascend 950DT；设备信息每卡总容量 98,304 MB |
| CPU | HiSilicon ARM64，2 路×96 物理核，2 线程/核，共384逻辑 CPU |
| CPU 时钟与 NUMA | 1.2–2.3 GHz；4 个 NUMA 节点 |
| 主机内存 | 约2.2 TiB；这是本机观测，不是经过压缩验证的最小需求 |
| Driver / firmware | 25.6.rc1 / 9.0.0.200.200，八卡兼容检查 OK |
| 容器系统 | Ubuntu 22.04 ARM64 基础栈 |
| CANN | 9.2.0 |
| Python | 3.12.13 |
| DeepInfer Python 栈 | torch 2.13.0+cpu、torch_npu 2.13.0rc1 |
| 编译器 | 主机 C++：GCC13.4；Ascend C：Bisheng15.0.5，使用GCC11头文件 |
| TileLang / TileKernels | 0.1.15 / 2.1.0 |
| deepseek-recipe | 0.2.0 |

CPU 的商品型号还未通过 DMI 确认；`lscpu Model name` 返回 `-`。本机规格与公开的鲲鹏950 96核档相符，但不能仅据此认定精确SKU。

### 镜像：基础镜像与最终运行目录都要保留

基础镜像：

```text
quay.io/ascend/vllm-ascend:v0.30.0rc2-a5
RepoDigest: quay.io/ascend/vllm-ascend@sha256:10f4e41479604545b74f3684e1a98e9e7116eff6adc72ecaec765d2dbecc7e81
历史本机 image ID: sha256:c3d00a196905159dbf0b97588d34f085086c2708712fa0d1c8fedd807f0e48a4
```

历史 DeepInfer 运行镜像：

```text
local/deepinfer:950dt-20261020
image ID: sha256:63345dc4a4ea2aa43178165c8acd26b36a73326141b5717105fa774dd9546376
```

`local/deepinfer:...` 是在本机生成的标签，不能直接从公共 registry 拉取。下文会从基础镜像重新构建，使用新标签 `local/deepinfer:950dt-rebuild-20261021`。

**历史成功状态还依赖 `/work` 挂载目录里的虚拟环境、源码、扩展与缓存。仅保存镜像不足以恢复。** 运行时修复没有全部重新烘焙进历史镜像。

基础镜像自带的 vLLM 默认 Python 使用 torch2.10/torch_npu2.10.post7。DeepInfer 使用另建的 `/work/engine-venv`，两套环境不能混用；`torch` 包版本中的 `+cpu` 不妨碍通过 torch_npu 使用 NPU。

### 操作前检查

在 ARM64 目标服务器执行：

```bash
uname -m
lscpu
free -h
df -h
docker version
docker ps --format '{{.Names}} {{.Image}} {{.Status}}'
ip -br addr
ls -l /dev/davinci* /dev/ummu /dev/uburma /dev/ubctl
cat /usr/local/Ascend/driver/version.info
```

确认 Docker 可用，目标八卡为本任务可用资源，`/lib/route.conf`、主机拓扑和设备节点存在。需要其他任务释放设备时按实际资源安排协调；脚本不会自动停止其他容器。

为权重、构建产物和缓存预留空间。权重本身约510.3GB，建议部署盘留至少1TB可用空间；镜像导出、离线备份或重复构建还需额外容量。运行容器的1TB内存上限是本次配置，不能解释为最小模型内存需求。

选择通用路径：

```bash
git clone --branch 950 --single-branch https://github.com/wio1997/Inference-Foundry.git
cd Inference-Foundry
export DEPLOY_ROOT=/srv/deepinfer-950dt
export MODEL_DIR=/srv/models/DeepSeek-V4.1-Flash-1017
mkdir -p "$DEPLOY_ROOT" "$MODEL_DIR"
cp -a deploy/ascend950/. "$DEPLOY_ROOT/"
cd "$DEPLOY_ROOT"
```

目录创建权限由服务器管理员按现有策略配置。不要使用已有其他任务目录覆盖安装。

## 下载模型、源码与依赖

### 1. 拉取固定基础镜像

```bash
export BASE_IMAGE=quay.io/ascend/vllm-ascend@sha256:10f4e41479604545b74f3684e1a98e9e7116eff6adc72ecaec765d2dbecc7e81
export DEEPINFER_IMAGE=local/deepinfer:950dt-rebuild-20261021
docker pull "$BASE_IMAGE"
docker image inspect "$BASE_IMAGE" --format '{{.Id}} {{.Architecture}}'
```

确认架构为 `arm64`。本次保持宿主驱动和固件原状；不能从 CANN 能 import 推导所有 graph/UVA/通信模式均已兼容。

### 2. 下载并校验模型

历史实际下载使用 ModelScope 的 `deepseek-ai/DeepSeek-V4.1-Flash-1017`，产物为 DeepInfer 可直接加载的 Hugging Face 目录格式。历史下载约29,561秒，耗时受网络和存储影响，不能作为固定时长承诺。

本仓库保存了84个文件的大小、SHA256和各自 ModelScope revision。整理后的下载器按文件 revision 获取，不依赖会变化的 `master`：

```bash
python3 download_model.py --dest "$MODEL_DIR" --workers 4
```

支持 `.partial` 文件断点续传；已正确的文件直接校验后复用。若已存在最终文件但校验失败，先检查该文件，下载器不会静默覆盖。完成后应显示：

```text
phase: complete
file_count: 84
sha256_verified_files: 84
weight_shards: 48
total_bytes: 510311045870
```

整理后的下载器把校验结果写入 `$DEPLOY_ROOT/model-verification.json`；旧记录中的模型内marker已改为工作目录内marker，以支持只读复用权重。单独复查：

```bash
python3 download_model.py --dest "$MODEL_DIR" --verify-only
```

也可从 [Hugging Face 官方模型仓库](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) 下载同一变体，但必须重新核对目标文件与锁清单，不把不同权重版本混用。模型的许可证和使用条件见官方仓库；GitHub 分支只保存校验清单，不上传权重。

### 3. 下载固定源码和嵌套依赖

```bash
python3 fetch_sources.py
```

此脚本按 commit 下载源码归档。它先下载主树，再填入 vendor 和嵌套依赖，避免主归档覆盖子模块。无需在目标网络反复递归克隆，但归档不含Git历史，构建出的某些版本字符串会包含 `unknown`，真实身份由锁文件确定。

| 组件 | Commit |
|---|---|
| DeepInfer | `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01` |
| deepseek-recipe | `935817c66f5d6e64838f419692545f06787e485f` |
| DeepSelect | `bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18` |
| FlashMLA | `2e5429fc5653bab6e081f09477126f731882a6a9` |
| DeepEP-Ascend | `3b25377d04b24fc6154698ded78a2bcb2c59afff` |
| DeepGEMM-Ascend | `3a35326271b069b8b5bf1a754435acd6cc91575b` |
| DeepJIT（两个 vendor 内） | `65f513952b8771408a19202f65e36398522db6db` |
| magic_enum | `b233b96e49d371bad00300f59b5ba581100b8745` |

对已打补丁的源码不要再次执行下载覆盖。需要更新版本时建立独立目录并重新检查补丁。

### 4. 下载 Python wheel

```bash
python3 download_wheels.py
```

下载器校验74个固定 engine 依赖 wheel，以及 deepseek-recipe0.2.0 的 ARM64 wheel。`locks/wheels.json`、`locks/recipe-wheels.json` 记录公开URL和SHA256。下载机只需Python3.11+和curl；安装和运行要求ARM64/Python3.12。

torch2.13 CPU wheel 的历史大小155,005,253字节，SHA256为 `6f307c2c32d764ffc6ff6893b801fad6d4752f3e67966cb8abf1843427c02604`。网络慢时先完成离线 wheelhouse，再构建，避免每次失败重下。

## 在不占用 NPU 的容器内构建

### 5. 构建 engine 与 API 环境

```bash
docker run -d --name deepinfer-950-build \
  --network host --cpus 16 --memory 64g \
  -v "$DEPLOY_ROOT:/work" \
  --entrypoint /bin/bash "$BASE_IMAGE" /work/bootstrap.sh

docker logs -f deepinfer-950-build
```

构建容器没有传入 NPU 设备。脚本依次：

1. 安装主机 GCC13、libdw/libnuma 开发依赖；Ubuntu包安装发生在容器层。
2. 在固定源码上按依赖顺序应用七个部署补丁，保存最终源文件哈希。
3. 创建 `/work/engine-venv`，从本地 wheelhouse 安装固定依赖。
4. 用编译器分流处理 C++20 与 Ascend C：C++源文件使用GCC13；设备源文件使用Bisheng和GCC11头文件。
5. 构建 DeepSelect、FlashMLA、DeepEP、DeepGEMM，再构建 DeepInfer 主机扩展。
6. 创建独立 `/work/recipe-venv`，安装协议适配所需的 deepseek-recipe/FastAPI/Uvicorn。

检查退出码与完成标记：

```bash
docker wait deepinfer-950-build
docker inspect deepinfer-950-build --format '{{.State.ExitCode}}'
cat "$DEPLOY_ROOT/build-state.txt"
```

必须退出码0并出现 `BUILD_COMPLETE_DEVICE_TEST_PENDING`。该字符串只说明构建完成，设备数值检查仍未进行。APT解析出的GCC13补丁版本可能随仓库变化，应保存本次 `g++-13 --version` 和包清单；历史使用的是13.4.0。

保存包含编译器和系统依赖的容器层，记录镜像锁：

```bash
docker commit deepinfer-950-build "$DEEPINFER_IMAGE"
docker image inspect "$DEEPINFER_IMAGE" --format '{{.Id}}'
python3 - <<'PY'
import json, os, pathlib, subprocess
image = os.environ['DEEPINFER_IMAGE']
info = json.loads(subprocess.check_output(['docker', 'image', 'inspect', image]))[0]
pathlib.Path('image-lock.json').write_text(json.dumps({'image': image, 'image_id': info['Id']}, indent=2)+'\n')
PY
```

`docker commit` 不包含 bind mount 的 `/work`。后续所有运行仍挂载同一个 `DEPLOY_ROOT`。

## 生成任务专用拓扑

### 6. 查询本机，而不是复制别人的 EID

本次环境默认的两层拓扑包含外部CLOS，但本任务只使用单机八卡；默认通信路径出现通道获取超时及大尺寸AllGather不能完成。

使用容器内DCMI只读查询生成八卡信息，并派生任务专有 layer0 rootinfo：

```bash
export DEEPINFER_IMAGE=local/deepinfer:950dt-rebuild-20261021
bash runtime.sh --topology
```

输出位于：

```text
runtime-20261021/topology/
  dcmi-rootinfo.json
  hardware-topology.json
  hccl_rootinfo.json
  hccl_rootinfo-local-only.json
  hixlep/ub_endpoint_npu_0.json ... ub_endpoint_npu_7.json
```

生成器使用带SHA锁的 `unofficial-ascend-tools==0.0.11` 查询，属于本次使用的辅助工具，不是官方支持认证。endpoint逻辑按实际查询到的local_id、port、EID及1DMESH/CLOS关系派生；无法推导peer时直接失败。

运行时只把这些文件只读挂载进任务容器，不改宿主 `/etc/hccl_rootinfo.json`、`/etc/hixlep`、`/lib/route.conf`。不同服务器重新生成，不能提交或复用其他机器的EID。历史拓扑描述文件来自 `/usr/local/Ascend/driver/topo/950/atlas_850_1.json`，文件名本身不能用于判断NPU型号。

同时使用已随补丁增加的显式开关：

```bash
DEEPINFER_UBMEM_NET_LAYER=0
DEEPINFER_ENGRAM_UBMEM_ONLY=1
```

它们选择本地UBMEM层，并让Engram跳过此路径未使用的Jetty。跨节点/其他拓扑不能照搬这两个开关，应重新核验网络与通信实现。

## 先做实卡算子验收

### 7. 准备接口并检查设备

```bash
ip -br addr
export HCCL_SOCKET_IFNAME='替换为本机实际网卡名'
bash runtime.sh --operator-checks
```

将占位符换成实际网卡。这里没有默认沿用历史服务器网卡名。启动脚本通过npu-smi确认0–7八卡空闲，不能确认时停止，不会清理其他进程。

依次检查：

| 检查 | 本次实卡覆盖 | 判据 |
|---|---|---|
| FlashMLA sparse prefill | 4组短/长KV，sink与topk_length | 原始参考函数数值检查通过 |
| DeepGEMM scale layout | 3种groups/granularity | 指数打包布局逐位一致 |
| candidate TopK | 5组，含空有效长度、短上下文、4096及超2048候选块 | 有效候选、分数、排序、填充及边界正确 |
| Engram FP8/BF16 | 8 rank各2种dtype，共16条rank PASS | 数据与缩放因子逐位一致 |

这些检查不是所有shape、所有kernel或完整模型质量认证。若扩展ABI、权重、CANN或编译器改变，重新执行。

## 启动模型和协议 API

### 8. 启动真实引擎

```bash
export MODEL_DIR=/srv/models/DeepSeek-V4.1-Flash-1017
export DEEPINFER_IMAGE=local/deepinfer:950dt-rebuild-20261021
export HCCL_SOCKET_IFNAME='替换为本机实际网卡名'
bash runtime.sh --start
docker logs -f deepinfer-950-engine
```

脚本暴露0–7设备、所需UB/URMA节点及驱动只读挂载，不使用privileged。主要容器参数：host network/IPC、16 CPU、1TB内存上限、memlock unlimited、OMP_NUM_THREADS=2；采用独立端口49361及JIT缓存目录。

核心配置由随附 `engine_bridge.py` 固定：

```python
ep_size=8
data_parallel_size=8
cp_size=1
max_model_len=4096
max_num_batched_tokens=256
max_num_seqs=1
gpu_memory_utilization=0.85
lookahead_tokens=0
```

环境包括 nextn=0、额外专家=0、main/draft eager、TASK_QUEUE_ENABLE=0、Engram NPU分片。日志应显示八rank真实加载和ready。历史首次模型就绪为北京时间2026-10-21 14:42:15；首次请求还有JIT开销，不要用首轮数字计算稳态性能。

### 9. 启动 CPU API 适配器

```bash
bash runtime.sh --start-api
docker logs -f deepinfer-950-api
curl --fail-with-body -sS http://127.0.0.1:18110/healthz
curl --fail-with-body -sS http://127.0.0.1:18110/readyz
```

API没有NPU设备，只调用loopback的真实后端18111。`/healthz`说明API进程可用；只有 `/readyz` 200且 `ready=true` 才说明八rank后端已就绪。未就绪返回503，不会用mock填充回答。

本任务 `api.py` 调用官方 `create_app(infer=...)`，使用模型对应的模板和tokenizer；官方server-py单独启动默认是mock示例，不能把它的Hello world当作模型运行成功。

### 10. 真实中文验收

```bash
bash try-inference.sh

curl --fail-with-body -sS http://127.0.0.1:18110/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":"小明有3盒铅笔，每盒12支，送出8支后还剩多少支？请用中文写出计算过程和答案。"}],"thinking":{"type":"disabled"},"max_tokens":256,"temperature":0,"stream":false}'
```

历史验收结果：

| 请求 | 输入/生成token | 实际结果 | 暖后完整响应耗时 |
|---|---|---|---:|
| 只回答数字：1+1等于几？ | 14/2，生成含EOS | `2` | 6.991秒 |
| 铅笔中文推理题 | 31/105，生成含EOS | 推导得到28支 | 30.566秒 |

首轮分别为376.249秒和89.049秒，包含冷编译等开销。以上均为单请求功能验证记录，不是TTFT或正式吞吐基线。

bridge 保存 `runtime-20261021/generations/job-*.json`，包括原始token、各rank耗时和SHA256。API仅依据真实EOS token及tokenizer配置去掉末尾EOS可见标记，不删原始证据、不少计已生成token。

如果从自己的电脑访问，可使用已配置SSH通道：

```bash
ssh -N -L 18110:127.0.0.1:18110 user@server
```

当前适配器只监听loopback，支持文本 `stream=false`；图像、流式、约束输出和logprobs不在此接口验收范围内。

## 本次解决的问题

| 问题/现象 | 原因 | 修复与证据 |
|---|---|---|
| DeepSelect构建引入CUDA测试依赖 | setup先导入tests.kernelkit再读显式平台 | 提前读取ASCEND构建目标，设备构建与导入检查通过 |
| 主机C++20与设备编译器不兼容 | GCC13头文件的受约束std::pair重载不被当前Bisheng设备前端接受 | 主机GCC13，设备Bisheng+GCC11头文件；保留语法probe |
| FlashMLA RINT转换编译失败 | 使用当前CANN未公开的 `_impl` 入口 | 改用公开RINT API，4组实卡数值检查 |
| Engram初始化OOM | materialize后又分配完整本地Engram表 | HF加载直接写入最终NPU storage，materialize约70.94→48.05GiB；8rank逐位校验 |
| HcclChannelAcquire超时 | 单机任务选中了外部CLOS路径及未用Jetty | 任务专用layer0快照、UBMEM显式选层；本机通信与模型加载通过 |
| 大尺寸AllGather不完成 | 容器仍暴露不适用的外部拓扑层 | 只对任务容器暴露local-only rootinfo |
| unknown format type | 通用torch::from_blob未建立NPU格式描述 | 使用at_npu::native::from_blob，重编译EP/GEMM扩展 |
| JIT找不到ascendc_assert | 缺少当前CANN公开头文件 | 包含utils/debug/asc_assert.h，保留检查 |
| 4K上下文candidate断言 | 配置候选2048大于实际块数 | 保留固定宽度，用valid end和-1填充；5组实卡检查 |
| 最终文本泄漏EOS标记 | 解码字符串直接返回 | 根据真实EOS条件仅移除末尾标记，复验中文/数字请求 |

详细复现、补丁顺序及修复后的验收方式见 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)。

## 恢复、备份与离线迁移

整理后的运行容器命名为 `deepinfer-950-engine`、`deepinfer-950-api`。同一目录重复 `--start`/`--start-api` 会校验任务标签、镜像和挂载，复用已有容器；不会自动删除不匹配容器。

维护本任务时先停止API，再停止引擎；恢复时先引擎再API，并重新检查ready：

```bash
docker stop deepinfer-950-api
docker stop deepinfer-950-engine
docker start deepinfer-950-engine
docker start deepinfer-950-api
bash runtime.sh --check
```

历史现场容器名称为 `deepinfer-engine-20261020`、`deepinfer-recipe-api-20261020`，路径和image ID也不同。保留原始现场时使用其原目录中的 `start-npu-after-release.sh --check/--start`；不要把两个部署目录的检查文件和镜像锁混用。

至少备份：

- 镜像：`docker save` 导出的完整运行镜像。
- 完整部署目录：engine-venv、recipe-venv、src、编译器分流配置、源码补丁及扩展、wheelhouse、wheels、image-lock、patch-state。
- 持久缓存：EP JIT、DeepGEMM JIT、TileLang cache；同硬件/软件恢复可复用，版本变化时需有意识地隔离缓存。
- 权重目录、84文件SHA锁和校验结果。
- 本次自己的运行日志、生成结果与验收JSON。

宿主设备拓扑在新机器重新查询。venv在容器内始终使用 `/work/...` 路径；迁移后仍把部署目录挂到 `/work`。不要在NPU模型还运行时执行需要占卡的operator-checks。

## 官方默认参数与后续性能研究

固定版官方命令 `python examples/offline_inference.py --ep 8` 默认CP1，DP未传时为EP/CP，因此也是DP8/EP8/CP1。其上下文32768、batch token32768、maxseq4、nextn5，并默认启用EPLB/冗余专家。这些是示例默认值，不是本机最佳性能认证。

本手册首先固定可运行配置；后续逐项验证较大batch/context、graph、MTP和负载均衡收益。当前用户已同意先测可行配置再选择表现好的方案深入分析。DeepInfer DP8/EP8/CP1 与 vLLM Ascend DP8/TP1/EP8 是待验收的对应布局；若测试其他TP布局，单独标明并行差异。

正式结果还需独立请求、正确EOS与计数、相同工作负载、正常维护开销、足够重复及无profiler基线。DeepInfer原生benchmark采用满接受换算并剔除部分LB flush步，不能直接作为API有效吞吐。性能数据和结论不在此部署成功记录中预填。

## 官方参考与本仓库产物

- [DeepInfer固定源码与README](https://github.com/deepseek-ai/DeepInfer/tree/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01)
- [官方离线示例及benchmark说明](https://github.com/deepseek-ai/DeepInfer/blob/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01/examples/README.md)
- [deepseek-recipe固定server-py说明](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/server-py/README.md)
- [DeepEP-Ascend固定版](https://github.com/deepseek-ai/DeepEP-Ascend/tree/3b25377d04b24fc6154698ded78a2bcb2c59afff)
- [昇腾官方V4.1资料索引](https://www.hiascend.com/activities/dynamic-news/796)
- [CANN官方TP8加载参考](https://gitcode.com/cann/cann-recipes-infer/tree/master/docs/models/deepseek_v4_1/deepseek_v4.1_low_latency_tp_guide.md)：这是另一套执行配置，host Engram/TP参数不能直接作为本次DeepInfer启动参数。
- [固定模型清单](../../deploy/ascend950/locks/model-manifest.json)、[wheel清单](../../deploy/ascend950/locks/wheels.json)、[源码清单](../../deploy/ascend950/locks/source-pins.json)
- [真实部署证据](../../deploy/ascend950/evidence/validated-deployment.json)、[中文验收](../../deploy/ascend950/evidence/chinese-verification.json)、[首次冷编译验收](../../deploy/ascend950/evidence/chinese-verification-initial.json)

GitHub产物不包含权重、token、密码、私有地址或设备EID；完整私有日志和原拓扑由实际部署者保留。
