# 005｜当前G133/G134 SGLang与vLLM对比：源码与复现入口

[分支首页](../../README.md) · [当前完整归档](../../reproduction/20261116/README.md)

本页对应2026-11-16采集、11月17日汇总的DeepSeek-V4.1-Flash-1017-w8a8对比。原仓为 **Inference-Foundry / sglang-dsv4.1flash1017-910c**；此页及相邻源码、脚本随同一提交提供。旧`launch.py`默认G42保持历史用途，**本轮使用`current_*`入口**。

SG短档G132原轮、长档G133固定确认轮相对最新G134 native吞吐差异为−1.42%、−9.39%、+7.92%、+8.51%；原轮/确认/早期间隔异常轮全部保留，SG四档超越目标未完成，平均TTFT四档较高。原始基线、采样修复前数据与当前成绩不可混用。

## 1. 代码、依赖、负载与结果

| 内容 | 固定在本提交内的入口 |
| --- | --- |
| 268个实测Python/C++元数据等文本（含所有23挂载覆盖） | [runtime目录](../../reproduction/20261116/runtime/) · [源→公开文件SHA](../../reproduction/20261116/runtime-manifest.json) |
| SGLang18个框架覆盖的累计补丁、SDK差异 | [框架patch](../../patches/sglang-a3-1017-current-g133.patch) · [SDK patch](../../patches/sgl-kernel-npu-current-g133.patch) · [基线/覆盖映射](../../reproduction/20261116/framework-patch-manifest.json) |
| 19个二进制原测身份与提取/CPU构建 | [二进制SHA](../../reproduction/20261116/binary-manifest.json) · [构建输入/adapter源码](../../reproduction/20261116/build/) · [准备程序](../../scripts/current_prepare.py) |
| 两边完整Docker、argv、env | [deployment](../../reproduction/20261116/deployment/) · [命令生成器](../../scripts/current_commands.py) |
| AISBench16份原配置（两框架×warm/formal×四档） | [configs](../../reproduction/20261116/configs/) · [运行器](../../scripts/current_aisbench.py) |
| 同一八档输入、种子/长度/tokenizer SHA | [数据manifest](../../reproduction/20261116/dataset-manifest.json) · [原数据恢复器](../../scripts/unpack_data.py) |
| 当前22阶段528请求、原轮/确认/当前native | [逐轮权威表](../../reproduction/20261116/measurement/rounds.json) · [主比较](../../reproduction/20261116/measurement/comparison.json) · [离线复算](../../reproduction/20261116/measurement/evidence/recompute.py) |

基础SG镜像`quay.io/ascend/sglang:cann9.2.0-a3-20261111`，本地image ID `08367522c4f9803fb876808c2cd4afe459e7e2b0adaab5a0daf36061edf8e517`，上游commit `d0e354bb7a0a75f2bd72f3724431198abcd9304b`。Python3.12.15、Torch2.13.0+cpu、torch_npu2.13.0、CANN9.2 B237，基础镜像完整包列表见[既有环境记录](../ENVIRONMENT.md)。本次268文本+19二进制身份覆盖后续适配，基础commit不代表未打补丁的完整运行代码。

native镜像`quay.io/ascend/vllm-ascend:v0.30.0rc2-a3`，ID `ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b`，vLLM commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`、Ascend commit `8fde6d4e335fe300e2b949aab0e31659cce0811d`；Torch2.10.0+cpu、torch_npu2.10.0.post7、CANN9.2 innerV100R001C25B114，0源码覆盖，[实际软件元数据](../../reproduction/20261116/native-actual-environment.json)和[19个驻留源码SHA](../../reproduction/20261116/native-source-hashes.json)。

客户端镜像`ghcr.io/aisbench/aisbench_benchmark:v3.1-20261013-master-openeuler22.03-py310-aarch64`，ID `05358d58d23f2679c71e48d93797d5d2a28d72eedc7fb4900479c89955465d76`。实际每档输入32768/65536/131072/262144，输出1024，C16，16warm+32formal；temperature0、ignore_eos/min_tokens1024、流式、retry1。八档数据SHA与最初归档逐项相同，可由现有词表/压缩包无损恢复，无需换新随机数据。

## 2. 准备目标机与依赖

在原测对应的aarch64单节点8物理910C/16逻辑64GiB die环境操作。先检查数据挂载点/空间，按当前任务规则将checkout和全部制品放到最大数据分区的授权任务目录。模型在本地SSD，Engram使用主机DRAM。**以下是可执行复现配方；本次发布没有在目标机重新构建、起服或压测。**

将本分支clone到该任务目录，进入checkout后执行：

```bash
git switch sglang-dsv4.1flash1017-910c
MODEL_PATH=/path/to/local-ssd/DeepSeek-V4.1-Flash-1017-w8a8
docker pull --platform linux/arm64 quay.io/ascend/sglang@sha256:8b52b8f5819cf48262f4a2212aa9218c5d4c063dbe71a78e641bb7d0b6d57ebf
docker tag sha256:08367522c4f9803fb876808c2cd4afe459e7e2b0adaab5a0daf36061edf8e517 quay.io/ascend/sglang:cann9.2.0-a3-20261111
docker pull --platform linux/arm64 quay.io/ascend/vllm-ascend:v0.30.0rc2-a3
docker pull --platform linux/arm64 ghcr.io/aisbench/aisbench_benchmark:v3.1-20261013-master-openeuler22.03-py310-aarch64
python3 scripts/current_prepare.py
python3 scripts/current_prepare.py --stage --runtime-root "$PWD/.work/sglang-current"
python3 scripts/current_prepare.py --build --runtime-root "$PWD/.work/sglang-current"
```

构建入口严格核验image ID/arm64；tag发生变化时拒绝替换。从固定native镜像用`docker create/cp`提取原始host objects/SDK（不启动该容器），在SG CPU容器链接隔离的MoE host/proto，保留原API/device kernel。此前三类ABI适配与strict QLI/Sparse vendor沿用已归档CPU配方，隐藏符号MoE桥接源码本次补齐。19个实测二进制不入Git，重建SHA记录在运行目录`rebuilt-binary-hashes.json`；与原测不一致时保留差异，不能将编译成功当新机功能/数值验收。

只做本机离线源码/路径检查可使用`--stage --skip-datasets`，输出明确标记未准备数据，benchmark执行会拒绝该目录。此选项不用于实际复现。

## 3. 生成并执行SGLang完整启动命令

先通过目标集群已有的资源锁/租约机制取得整机16逻辑die与所需端口，持有至自有服务退出。此命令生成器**不执行Docker、不认领资源**；不能代替集群资源协调，也不要同时启动两套框架争用卡组。服务端口18167、通信18280/14699及框架派生端口须按原部署预留。

```bash
python3 scripts/current_commands.py --engine sglang \
  --model-path "$MODEL_PATH" --runtime-root "$PWD/.work/sglang-current" \
  --check-files --allow-rebuilt-binaries --shell > .work/start-sglang-current.sh
bash .work/start-sglang-current.sh
```

第二条在服务终端前台运行，先检查生成文件。`--allow-rebuilt-binaries`仅允许已记录SHA的重建产物，未跳过source/control-file/bind核验，不代表新产物已在NPU通过。SG仍有实际权重映射、mHC/空Sparse等启动/预热门禁；不能移除这些检查后复用本分数。

实测引擎完整参数如下。实际执行还须配套完整Docker env、只读挂载和provider路径；不要只复制这段而省掉依赖：

```bash
python3 -m sglang.launch_server \
  --model-path /model \
  --served-model-name Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8 \
  --language-model-only \
  --trust-remote-code \
  --device npu \
  --attention-backend dsv4 \
  --tp-size 16 \
  --dp-size 4 \
  --ep-size 16 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --moe-a2a-backend ascend_fuseep \
  --quantization modelslim \
  --page-size 128 \
  --kv-cache-dtype bfloat16 \
  --context-length 270336 \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 131072 \
  --max-running-requests 16 \
  --mem-fraction-static 0.93 \
  --engram-config '{"placement":"cpu","replication":"per_node","lookup":"all_reduce"}' \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path /model \
  --speculative-draft-model-quantization modelslim \
  --speculative-draft-attention-backend ascend \
  --speculative-dspark-block-size 5 \
  --speculative-num-draft-tokens 6 \
  --host 127.0.0.1 \
  --port 18167 \
  --watchdog-timeout 1800 \
  --json-model-override-args '{"quantization_config":{"quant_method":"modelslim"}}' \
  --nccl-port 18280 \
  --max-total-tokens 1100000 \
  --random-seed 827671776 \
  --prefill-decode-interval 1 \
  --ep-num-redundant-experts 16 \
  --ep-dispatch-algorithm dynamic \
  --init-expert-location /task/continuation_G95_replica_model_fix_20261115/placement.json
```

## 4. AISBench

另一个终端、同一checkout中执行。使用目标集群已有且跨会话共用的benchmark锁文件：

```bash
MODEL_PATH=/path/to/local-ssd/DeepSeek-V4.1-Flash-1017-w8a8
BENCHMARK_LOCK=/path/to/existing/shared-benchmark.lock
python3 scripts/current_aisbench.py --engine sglang \
  --model-path "$MODEL_PATH" --runtime-root "$PWD/.work/sglang-current"
python3 scripts/current_aisbench.py --engine sglang \
  --model-path "$MODEL_PATH" --runtime-root "$PWD/.work/sglang-current" \
  --benchmark-lock "$BENCHMARK_LOCK" --execute
```

先确认服务已完成启动。runner默认仅列计划；执行前核验自有服务标签、固定镜像、挂载、模型控制文件和原数据/配置SHA，客户端runc/8CPU/32GiB/0NPU。每阶段前flush/reset缓存，各档16warm→32formal，命令与原测相同，阶段1200s、benchmark总3600s边界。SG正式前核验16rank实际权重映射和warmup的vendor/projection/smallHC/emptySparse收据，并freeze参数检查，防止把首次shadow资格检查混入正式计分。发现异物负载或失败立即中止客户端、保留原件，不自动重试/挑分。

`--profiles 32768 65536 131072`对应G132的三个有效档；`--profiles 262144`对应G133原轮，`--profiles 131072 262144`对应G133确认选择。**历史SG四档并非同一服务代次**。默认新跑四档只是新复现计划，不冒充已经执行的同代次四档成绩。

结果在`<runtime-root>/results/<engine>/<run-id>/`；CLI和token检查通过后仍标记`validated=false`，必须检查SQLite时钟/实际并发、归属、源码及完整指标后接受新成绩。runner不会以退出码直接认证收益/精度/重复性。换模型进程时使用新的runtime目录，禁止带着旧FROZEN/资格收据起新代次。

## 5. 切换优化native vLLM

SG自有服务和客户端退出、设备/端口释放并重新完成资源协调后，使用独立新runtime目录。native无需SG适配构建：

```bash
python3 scripts/current_prepare.py --stage --runtime-root "$PWD/.work/vllm-current"
python3 scripts/current_commands.py --engine vllm \
  --model-path "$MODEL_PATH" --runtime-root "$PWD/.work/vllm-current" \
  --check-files --shell > .work/start-vllm-current.sh
bash .work/start-vllm-current.sh
```

另一个终端使用同一benchmark锁：

```bash
MODEL_PATH=/path/to/local-ssd/DeepSeek-V4.1-Flash-1017-w8a8
BENCHMARK_LOCK=/path/to/existing/shared-benchmark.lock
python3 scripts/current_aisbench.py --engine vllm \
  --model-path "$MODEL_PATH" --runtime-root "$PWD/.work/vllm-current" \
  --benchmark-lock "$BENCHMARK_LOCK" --execute
```

native服务端口18168、DP RPC18288及派生通信端口均需预留；保留原生镜像0源码覆盖。其完整引擎参数如下：

```bash
vllm serve /model \
  --language-model-only \
  --host 127.0.0.1 \
  --port 18168 \
  --served-model-name Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8 \
  --max-model-len 270336 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 32 \
  --data-parallel-size 4 \
  --tensor-parallel-size 4 \
  --data-parallel-rpc-port 18288 \
  --enable-expert-parallel \
  --gpu-memory-utilization 0.93 \
  --tokenizer-mode deepseek_v41 \
  --trust-remote-code \
  --safetensors-load-strategy lazy \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}' \
  --quantization ascend \
  --block-size 128 \
  --kv-cache-dtype auto \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --async-scheduling \
  --enable-prefix-caching \
  --engram-config '{"cpu_offload": true, "dp_shared_memory": true}' \
  --speculative-config '{"method": "dspark", "num_speculative_tokens": 5, "enforce_eager": false}' \
  --additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": false}, "enable_cpu_binding": true, "enable_fused_mc2": 1, "enable_dsa_cp": true, "enable_flashcomm1": true, "enable_shared_expert_dp": true, "multistream_dsv4_dsa_overlap": true, "multistream_engram_overlap": true}'
```

## 6. 比较边界与发布验证

SG使用TP16/EP16、attention DP4/TP4、全局运行16（每attention DP4）、16冗余专家dynamic分发；native为TP4×DP4、EP16、每DP max-num-seqs32、冗余0。二者Engram都DRAM，HCCL都AIV/1024，DSPARK/图/前缀缓存/多流/CPU绑定实际开启，但CP实现、MoE通信和Torch/CANN构建不同。SG是query行分片/交换，generic prefill CP=false；native DSA CP=true，不能据相似参数声称单变量框架比较。

本次发布离线校验源码SHA/累计补丁应用/命令语义、数据manifest和528请求指标；便携脚本未在新目标机重跑，19重建二进制尚需新目标机验证。整模型质量、142权重分片完整hash、重复噪声标定和当前64K混合阶段算子根因仍未验收，性能目标保持未完成。离线复算入口：

```bash
python3 scripts/validate_archive.py
python3 reproduction/20261116/measurement/evidence/recompute.py
```
