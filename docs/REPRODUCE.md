> 当前默认版本为G42；G43为仅完成32K初测的候选。两版完整参数与边界见[003条目](003-g42-g43/README.md)。下文历史示例的G37/G30需显式指定`--generation`；固定依赖构建配方仍适用。

# 启动与复现

**本页下方保留G30历史复现说明；当前使用请先读[G37入口](002-g37-markov/README.md)。** 默认启动器已改为G37；G37执行时还必须提供legacy设备锁目录及实际租约策略文件。下方G30命令均显式指定历史版本，不代表推荐该版本。

原实测的代码、参数和数据已经核验；本次归档新增的便携构建/启动脚本只做了离线检查，**没有再次启动模型或重跑压测**。归档检查通过不代表在另一台机器重建通过。历史守护脚本包含当时的探针和状态前置条件，保留用于审计，不作为新机器的直接启动入口。

## 1. 依赖和数据

在有对应 910C 资源的 aarch64 目标机上，将本分支放在最大可用数据分区的任务目录。先确认共享资源归属、驱动和空间，再取固定镜像。镜像标签若发生变化，脚本会因 ImageID 不符而拒绝使用。

```bash
docker pull --platform linux/arm64 quay.io/ascend/sglang@sha256:8b52b8f5819cf48262f4a2212aa9218c5d4c063dbe71a78e641bb7d0b6d57ebf
docker tag sha256:08367522c4f9803fb876808c2cd4afe459e7e2b0adaab5a0daf36061edf8e517 quay.io/ascend/sglang:cann9.2.0-a3-20261111
docker pull --platform linux/arm64 quay.io/ascend/vllm-ascend:v0.30.0rc2-a3
python3 scripts/prepare_dependencies.py
python3 scripts/prepare_dependencies.py --execute
python3 scripts/unpack_data.py --with-results
```

构建脚本从旧依赖镜像提取原 vendor，过滤四类接口对应的 20 个 payload，重建当前 CANN 下的 host 注册，再编译三个 Torch2.13 桥接扩展。构建需要 cmake、C++ 编译器、binutils 和 patchelf；缺失时先准备对应 CPU 构建工具。构建容器不映射 NPU，网络关闭。不会将旧 vLLM 引擎作为服务运行，也不替换原 device/AICPU 实现。重建二进制的哈希记录在 `.work/rebuilt-extension-hashes.json`；必须在目标环境重新验证功能，不能把新构建的哈希不同直接认作精度相同。

## 2. 启动

模型路径由调用者给出，读取本地 SSD。先看命令：

```bash
python3 scripts/launch.py --generation g30 --model-path /path/to/local-ssd/model
```

实际运行需指定本集群已有、所有会话共用的设备锁和端口锁目录。启动脚本保留全部 16 个逻辑设备与端口锁，检查 NPU 空闲、每个 die 至少 53GiB 空闲 HBM、80 个目标 Q/KV 投影均为 FLOAT，以及依赖 payload 哈希。它不能代替集群已有 lease/预约规则；必须先完成相应资源协调。

```bash
python3 scripts/launch.py \
  --generation g30 \
  --model-path /path/to/local-ssd/model \
  --device-lock-dir /path/to/existing/shared-device-locks \
  --port-lock-dir /path/to/existing/shared-port-locks \
  --execute
```

服务前台守护，最多 21600 秒；日志位于 `.work/service.log`。退出或达到期限仅停止本脚本创建且标签吻合的容器。锁在作业退出后释放。重试前先检查已有容器、日志与资源；脚本不自动覆盖历史状态。

实际引擎启动参数保存在 [requested_args.json](../deployment/requested_args.json)，完整环境变量见 [environment.json](../deployment/environment.json)，原 Docker 命令见 [docker_command.json](../deployment/docker_command.json)。宿主路径已脱敏，容器内路径保持原布局。启动脚本自动挂载 [G30 覆盖文件](../compat/runtime/g30)，无需先手工应用补丁。

关键参数：TP16 / EP16 / attention DP16；ModelSlim W8A8；BF16 KV；context 270336；KV 600000（实际槽位 599936）；全局 prefill chunk 16384、每 attention DP 为 1024；max prefill 131072；max running 32；mem fraction 0.93；Engram cpu/per_node/all_reduce；DSPARK block5、verify6。

DeepEP auto，HCCL 512MiB，normal INT8、hybrid1；target/draft 完整 NPU graph 的 batch1/2；radix、overlap、多流/planstream、CPU 亲和、ZBAL graph/VMM 开启。`SGLANG_OPT_FUSE_WQA_WKV=1`，640 个目标融合模块初始化（40 层 × 16 ranks）；draft 保持框架原生不融合。前端 `--dp-size 16` 经框架归一化后的普通 dp_size 可为 1，attention DP 仍为 16。

## 3. AISBench

准备固定客户端镜像，不给客户端映射 NPU：

```bash
docker pull --platform linux/arm64 ghcr.io/aisbench/aisbench_benchmark:v3.1-20261013-master-openeuler22.03-py310-aarch64
python3 scripts/run_benchmark.py --model-path /path/to/local-ssd/model
python3 scripts/run_benchmark.py \
  --model-path /path/to/local-ssd/model \
  --server-container sglang-dsv41-910c \
  --benchmark-lock /path/to/existing/shared-benchmark.lock \
  --execute
```

客户端限 8 CPU、32GiB，使用 `/task/tokenizer` 的模型 tokenizer。四档顺序执行，每档 warm16 → formal32，阶段间 flush cache。每阶段 AISBench 命令与原测量一致：`--mode perf --num-warmups 0 --num-prompts 16或32 --max-num-workers 1`；并发 16 来自模型配置的 `batch_size=16`，worker 数不等于请求并发。见 [八份实际配置](../configs)。

`VLLMCustomAPI` 只是调用 SGLang OpenAI 兼容接口的客户端适配器。`retry=1` 在此版本是一次 HTTP 尝试；`retry=0` 不会发送请求。模型输出参数为 temperature0、ignore_eos、min/max1024。请求 total timeout 使用该版本默认 null；每阶段外部 7200 秒上限，总作业 6 小时上限。原始输入为合成词表文本，不用于质量打分。

runner 终止后仍须检查逐请求成功率、实际 token 数、并发、时序与占用，才可接受新结果。新增 runner 不会把 CLI 退出成功自动标为性能验收通过。
