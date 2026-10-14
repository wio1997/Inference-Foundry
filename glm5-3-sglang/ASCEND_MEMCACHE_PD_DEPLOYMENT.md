# Ascend MemCache：GLM-5.3 W8A8 / SGLang PD 部署

本页记录 GLM-5.3 W8A8 在 Ascend 910C 上接入 Ascend MemCache 的部署方法。目标负载为 **80,000 输入 token、600 输出 token**。现场地址、SSH 凭据和权重访问地址不入库；命令中的角色地址由部署者填写。

**验证状态（2026-10-14）**：指定 **4P2D（P EP64 / DP4 / attention TP16，D EP32 / DP16 / attention TP2）** 已完成80,000输入/600输出冷请求和清除L1/L2后的跨P节点L3回放，实际输出、HTTP/SSE与验证密钥均通过。D静态显存比例保持 **0.90**，Ascend DSA index预算缺陷已修复；0.80候选未执行。四个DDR提供者、64个P worker SDK预热及六台96个NPU die健康检查通过。服务在交付时保持运行。这里只验收一次固定合成请求对，不代表16并发吞吐或SLO。详见 [验证快照](deployment/ascend_memcache/validation-20261014.json)。

## 1. 拓扑与 SGLang 参数

4P2D 指 **四台物理 Prefill 主机加两台物理 Decode 主机**，每台 16 个 NPU die；P、D 是两个独立的分布式模型进程组。

| 角色 | 物理节点 | EP | DP | 每 DP 副本的 attention TP | SGLang `--tp-size` | 每台节点 rank |
| --- | --- | --- | --- | --- | --- | --- |
| P | P0、P1、P2、P3 | 64 | 4 | 16 | 64 | 0、1、2、3 |
| D | D0、D1 | 32 | 16 | 2 | 32 | 0、1 |

启用 `--enable-dp-attention` 后，attention TP = 全局 `tp-size / dp-size`。因此 P 使用 `--tp-size 64 --dp-size 4 --ep-size 64`，D 使用 `--tp-size 32 --dp-size 16 --ep-size 32`。不要将 P 的 `--tp-size` 直接设为16、D直接设为2，再配 EP64/EP32。

```mermaid
flowchart LR
    Client[80K 输入 / 600 输出] --> Router[SGLang PD Router]
    Router --> P[P0–P3: DP4 / attention TP16 / EP64]
    P -->|原生 Ascend / MemFabric SDMA| D[D0–D1: DP16 / attention TP2 / EP32]
    P <--> L2[每 worker 的 Host HiCache L2]
    L2 <--> L3[Ascend MemCache L3: 四个独立 DDR 提供者]
    Meta[MetaService + ConfigStore] --- L3
    D --> Client
```

**两个数据面分别配置**：P→D 的 KV 搬运使用原生 `ascend` 后端 / MemFabric SDMA；P 的 HiCache L3 使用 `ascend_memcache` / `device_sdma`。安装 MemCache 不会自动把 PD 传输改成另一种连接器。

## 2. 固定依赖与离线安装

本次工程环境为 Linux **aarch64**、Ascend 910C、CANN9.0.0、驱动25.5.2、Python3.11、SGLang `0.5.17.dev386+gc5bd3d7dc`。镜像原 tag 为 `sglang-tianjin:arm64-cann9.0.0-a3-v0.5.16`；tag中的版本不能代替镜像内实际版本。

固定使用 **memfabric_hybrid1.1.5 + memcache_hybrid1.1.5**。在有网络的工作机下载匹配 aarch64/cp311 的 wheel，传入离线现场，再在独立派生容器中安装：

```bash
python3 -m pip install --no-index --no-deps --no-cache-dir \
  /packages/memfabric_hybrid-1.1.5-cp311-cp311-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl \
  /packages/memcache_hybrid-1.1.5-cp311-cp311-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl
python3 -m pip show sglang memfabric-hybrid memcache-hybrid
```

| wheel | SHA256 |
| --- | --- |
| memfabric1.1.5 | `8e8aacd97e776462d11d8513f072209d41d3a638d898057a806b4dc662df957d` |
| memcache1.1.5 | `95914b6df5040aa7871e677d4368925592f04acd313cf9ee6ec32f26f5d8ed12` |

1.2.1 的匹配组合在本镜像的独立 SDK 生命周期探针中出现 native heap crash，因此本次未采用；这不是对所有环境或后续版本的结论。

原镜像、原启动文件和原容器保留。每台运行容器映射本机16个 `/dev/davinci*`、`davinci_manager`、`hisi_hdc`，只读映射 Ascend driver/firmware/dcmi，使用 host network / IPC、足够 shm、memlock，以及只读 `/model`。模型权重约764GB，**先复制到每台本地 NVMe，完整 SHA256 校验后再启动**。NFS/SSH传输只用于分发，不作为运行时权重目录。

## 3. SGLang 的 MemCache 适配

附带 [部署工具与补丁](deployment/ascend_memcache/) 基于 [SGLang PR26043](https://github.com/sgl-project/sglang/pull/26043)，固定源 head 为 `08246a9e20fef1d75b4ee6c3d9646c415f215a54`。本工具针对上述镜像内源代码做精确锚点替换，不升级整个 SGLang。

```bash
# 容器内执行；实际工具位置自行设置。
KIT=/workspace/glm5-3-sglang/deployment/ascend_memcache
bash "$KIT/fix_sonames.sh"
python3 "$KIT/patches/mc-apply-patch.py" \
  --sglang-root /sgl-workspace/sglang/python/sglang
# Ascend DSA index显存预算修复：固定源码SHA，保存原文件，拒绝未知版本。
python3 "$KIT/fix_ascend_dsa_budget.py" \
  --source /sgl-workspace/sglang/python/sglang/srt/model_executor/pool_configurator.py
# 再执行一次应报告所有修改已存在，不能继续重复添加。
python3 "$KIT/patches/mc-apply-patch.py" \
  --sglang-root /sgl-workspace/sglang/python/sglang
```

本适配的关键点：

- 注册 `ascend_memcache` 后端、零拷贝读取路径和 NPU layout 选择。
- GLM DSA + Ascend MemCache 选择 **HiRadixCache + 已有 NPU MLA Host Pool**；本环境的 UnifiedRadixCache/HostPoolGroup 路径未作为工作点。
- 使用 `kernel_ascend`、`page_first_kv_split`，同时存取 **K/V/index** 三个组件。缺少 index 必须判为 MISS，不能把不完整缓存当 HIT。
- 在本环境中，跨进程 `batch_put_from` 写入不可被另一参与者查询；使用 **copying `put` 写回退**，读取仍使用零拷贝。不能宣称零拷贝写已验证。
- SDK 显式、幂等 `close()`；独立探针及服务退出需要检查真实进程退出码。

`fix_sonames.sh` 在派生容器内补 wheel 库的 SONAME 链接，处理 bare `dlopen("libhcom.so")` 等解析；不修改主机驱动。锚点或原文件不同会报错，应先核对镜像版本，不能跳过安装失败继续启动。

## 4. MetaService 与四个 DDR 提供者

建议放在独立服务容器里，先起服务再加载 P 模型。示例端口分开使用：Meta25300、ConfigStore25301、Meta metrics25302；PD 的 MemFabric store24946另行启动。

Meta配置文件：

```json
{
  "meta_service_url": "tcp://P0_IP:25300",
  "config_store_url": "tcp://P0_IP:25301",
  "metrics_url": "http://P0_IP:25302",
  "ha_enable": false,
  "log_level": "info",
  "log_path": "/workspace/logs/meta",
  "log_rotation_file_size": 20,
  "log_rotation_file_count": 10,
  "evict_threshold_high": 90,
  "evict_threshold_low": 80
}
```

```bash
# P0 的独立服务容器中：
python3 "$KIT/start_meta_service.py" /workspace/metaservice.json
```

P0/P1/P2/P3 各启动一个独立 DDR 提供者，配置必须一致，ready文件各自独立：

```bash
python3 "$KIT/pool_provider.py" \
  --meta "tcp://${P_HEAD}:25300" \
  --config-store "tcp://${P_HEAD}:25301" \
  --world-size 128 --dram 16GB --max-dram 64GB \
  --device 0 --ready-file /workspace/provider.ready.json
```

`world_size=128` 是进程参与者的上限，覆盖64个 P 模型进程加4个提供者，不能写成物理节点数6。每个提供者 `dram_size=16GB`；**模型 SDK 客户端 `dram_size=0GB`**。`max_dram_size=64GB` 是每进程 VA stride 上限，不是每进程实际保留64GB物理内存。四台提供者合计64GB配置容量。

提供者先于 P 的 Host HiCache 分配。原先在大型 Host L2 分配完成后让每个模型进程再申请16GB DDR，曾出现 `HalMemCreate` 失败；本工作点把物理 DDR 从模型客户端移到独立提供者。正常停止通过 SIGTERM/SIGINT 调用 `close()`，等待 `POOL_CLOSE_RESULT` 和退出码，不把强制 kill 当 clean close。

## 5. 公共环境

填写 `P_HEAD`、`D_HEAD`、当前 `NODE_IP`、网络接口和节点 rank；它们是现场值。以下在模型容器内执行：

```bash
export P_HEAD D_HEAD NODE_IP COMM_IFACE
: "${P_HEAD:?set P0 address}" "${D_HEAD:?set D0 address}" "${NODE_IP:?set this host address}" "${COMM_IFACE:?set communication interface}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ASCEND_LAUNCH_BLOCKING
export NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HCCL_SOCKET_IFNAME="$COMM_IFACE" GLOO_SOCKET_IFNAME="$COMM_IFACE"
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1000 TASK_QUEUE_ENABLE=2
export STREAMS_PER_DEVICE=32 SGLANG_SET_CPU_AFFINITY=1
export SGLANG_NPU_USE_MULTI_STREAM=1 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1 DEEPEP_NORMAL_LONG_SEQ_ROUND=72
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
export ASCEND_MF_STORE_URL="tcp://${D_HEAD}:24946"
export ASCEND_MF_TRANSFER_PROTOCOL=sdma
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

在 D0 的独立服务进程中启动SDK内置store：

```bash
export ASCEND_MF_STORE_URL="tcp://${D_HEAD}:24946"
python3 -m memfabric_hybrid.launch_ascend_mf_store
```

SGLang原生 Ascend engine 使用上述 `ASCEND_MF_STORE_URL`。首次启动前检查监听端口归属，避免复用别的任务的服务。

## 6. P 启动：EP64 / DP4 / attention TP16

```bash
export SGLANG_ENABLE_UNIFIED_RADIX_TREE=0
export SGLANG_ASCEND_MEMCACHE_ENABLE_WARMUP=1
export SGLANG_ASCEND_MEMCACHE_COPY_WRITE=1
export SGLANG_HICACHE_MEMCACHE_CONFIG_PATH=/workspace/memcache-local.json
printf '{}\n' > "$SGLANG_HICACHE_MEMCACHE_CONFIG_PATH"

# 必须在四台一致；含权重身份、KV schema、page geometry和并行布局。
export MC_TAG="glm53w8a8-${WEIGHT_MANIFEST_SHA16}-kvs3-pg128-tp64dp4ep64-ctx98304"
MC_EXTRA=$(python3 - <<'PY'
import json,os
print(json.dumps({
    'meta_service_url': f"tcp://{os.environ['P_HEAD']}:25300",
    'config_store_url': f"tcp://{os.environ['P_HEAD']}:25301",
    'protocol':'device_sdma','dram_size':'0GB','max_dram_size':'64GB',
    'world_size':128,'hbm_size':'0','max_hbm_size':'0',
    'log_level':'info','extra_backend_tag':os.environ['MC_TAG']
}))
PY
)
python3 -m sglang.launch_server \
  --model-path /model --trust-remote-code --device npu --attention-backend ascend \
  --dtype bfloat16 --quantization modelslim \
  --tp-size 64 --dp-size 4 --ep-size 64 \
  --enable-dp-attention --enable-dp-lm-head \
  --nnodes 4 --node-rank "$P_RANK" --dist-init-addr "${P_HEAD}:29834" --dist-timeout 3600 \
  --host "$NODE_IP" --port 31400 --served-model-name glm-5.3-pd \
  --disaggregation-mode prefill --disaggregation-bootstrap-port 31934 \
  --disaggregation-transfer-backend ascend \
  --enable-hierarchical-cache --hicache-io-backend kernel_ascend \
  --hicache-storage-backend ascend_memcache --hicache-mem-layout page_first_kv_split \
  --hicache-size 32 --hicache-write-policy write_through \
  --hicache-storage-prefetch-policy wait_complete \
  --hicache-storage-backend-extra-config "$MC_EXTRA" \
  --context-length 98304 --mem-fraction-static 0.75 \
  --chunked-prefill-size 16384 --max-prefill-tokens 98304 --max-running-requests 4 \
  --enable-metrics --moe-a2a-backend deepep --deepep-mode normal \
  --disable-cuda-graph --disable-shared-experts-fusion \
  --reasoning-parser glm45 --watchdog-timeout 3600
```

`P_RANK` 为0/1/2/3，P0是组 leader。`--hicache-size32` 按每worker计算，整机预算还要加 index Host Pool、模型进程、通信缓存和独立 DDR；不得当成整机只用32GB。

## 7. D 启动：EP32 / DP16 / attention TP2

D不启用 HiCache L3。P负责前缀缓存，D保留原生PD接收路径；本次先使用 eager、无MTP，不同时改变推测解码与传输策略。

```bash
export HCCL_BUFFSIZE=64
export DEEPEP_HCCL_BUFFSIZE=650
export TASK_QUEUE_ENABLE=1
unset DEEPEP_NORMAL_LONG_SEQ_ROUND DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
python3 -m sglang.launch_server \
  --model-path /model --trust-remote-code --device npu --attention-backend ascend \
  --dtype bfloat16 --quantization modelslim \
  --tp-size 32 --dp-size 16 --ep-size 32 \
  --enable-dp-attention --enable-dp-lm-head \
  --nnodes 2 --node-rank "$D_RANK" --dist-init-addr "${D_HEAD}:29836" --dist-timeout 3600 \
  --host "$NODE_IP" --port 31401 --served-model-name glm-5.3-pd \
  --disaggregation-mode decode --disaggregation-transfer-backend ascend \
  --context-length 98304 --mem-fraction-static "${D_MEM_FRACTION:-0.90}" \
  --max-running-requests 16 \
  --moe-a2a-backend deepep --deepep-mode low_latency \
  --disable-cuda-graph --disable-shared-experts-fusion \
  --reasoning-parser glm45 --watchdog-timeout 3600
```

`D_RANK` 为0/1；D0是leader。实际工作点为静态显存比例0.90、Decode `low_latency`、MoE650MB/TP与Attention64MB、任务队列1、每rank dispatch上限32。预算修复后每个D分组容量为186368token，KV约19.08GiB，分配后空闲约6.1GiB；已经过80K/600真实请求验证。DP16不代表已完成16并发长请求的容量验收。

## 8. Router 与启动顺序

```bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation --policy round_robin \
  --prefill "http://${P_HEAD}:31400" 31934 \
  --decode "http://${D_HEAD}:31401" \
  --host "$D_HEAD" --port 31480 \
  --request-timeout-secs 7200 \
  --health-check-timeout-secs 10 --health-check-interval-secs 15 \
  --health-failure-threshold 3 --health-success-threshold 1
```

依次完成：权重与镜像核验 → 独立服务容器 → PD MemFabric store → MemCache MetaService → 四个DDR提供者 → 四个P rank、两个D rank → 所有rank readiness和L3 warmup → Router → 小请求 →80K/600 → L3回放。每个rank用本次启动独立日志和退出码，不能接受旧日志里的ready标记。

## 9. 80K/600 与真实 L3 回放验证

原生 `/generate` 测试固定 `input_ids` 长度80,000、`max_new_tokens=600`、`ignore_eos=true`，使用相同请求文件；在合成输入内埋一条唯一nonce，并检查输出。输出600包括原生 completion 中的 reasoning token；不等于最终可见正文一定600token。

先冷请求，等待 `sglang:backuped_tokens_total` 对应已完成写入；然后 **POST P leader `/flush_cache` 清L1/L2，保留Meta和DDR提供者**，发送相同文件。要求同时成立：HTTP200、SSE完整结束、实际输入80000/输出600、nonce存在、请求SHA相同、L3完成读取计数增加、P日志显示前缀命中。仅看到前缀命中不足以区分L1/L2/L3。

P leader的HTTP metrics只覆盖本机DP；其他物理P的DP counters需读取其本次运行的 Prometheus mmap目录并导出。可在每个P peer的服务容器中运行附带的只读导出器：

```bash
python3 "$KIT/peer_metrics.py" \
  --log /workspace/logs/p-current.engine.log --host "$NODE_IP" --port 31490
```

64个TP进程的计数不可直接相加当作逻辑token数；需按参与当前请求的TP归一化。跨机墙钟可能有偏差，TTFT与总时间由同一客户端 `time.monotonic` 测量。

本次指定4P2D的单请求实测如下，计时来自同一客户端的`time.monotonic`：

| 条件 | 实际输入/输出token | TTFT | 完整请求时间 |
|---|---:|---:|---:|
| 冷请求 | 80000 / 600 | 46.5966s | 214.6486s |
| 清除L1/L2后的L3回放 | 80000 / 600 | 9.5940s | 177.1510s |

冷请求由P3完整计算80000token；回放由另一物理P节点P1命中79872token，只新增计算128token。冷请求完成L3写入增量80000；回放L3读取计数增量1277952，按attention TP16归一化为79872，与日志一致。两次request SHA256均为`cd65d7a9067f48b1fbcce65718a139be692a8f12fdd8bb7539cff7428e8e41c8`，HTTP200、SSE结束、实际600输出及生成文本中的验证密钥全部通过。TTFT比值4.8568×，这里只描述本次回放；每条件n=1，不是多轮吞吐结论。

历史原2P2D单请求记录仍保存在验证快照：冷TTFT34.923s、总318.578s，L3回放TTFT3.862s、总285.910s。拓扑不同，不作为本次4P2D收益的对照组。

新4P2D首启观察：P的64个模型进程分别完成MemCache SDK warmup；D每进程权重约35.63GB，KV池204288token/约20.91GB，比例0.90。首次 `aclnnNotifyDispatch` 的HCCL动态分配失败，报请求2098200576字节；随后SGLang清理进程产生137/3退出码，容器 `OOMKilled=false`。这不能读作完整服务启动成功。

随后low_latency、所有域650MB、任务队列2时，GroupedMatmul工作区申请28MiB失败，空闲仅31.79MiB；任务队列0也未完成初始化，随后记录动态内存错误。将MoE650MB与TP/Attention64MB分开并采用任务队列1后，4P2D先通过小请求；进一步修复NPU index预算后完整重启，完成上面的80K/600与L3验收。通信配置与队列调整没有单变量拆开测量，不能分别宣称其性能贡献。

### 修复是否包含模型硬编码

本次容量修复没有新增GLM模型名判断，也没有把78层、128维、512维或186368token写入运行时计算。`index_head_dim`仍由模型配置取得，`kv_size`由KV dtype取得，有效层数沿用现有分层计算；Ascend且未启用HiSparse时，index预算改为`index_head_dim * kv_size * effective_num_layers`。容量仍由实际显存预算计算，D静态显存比例保持0.90。上述具体维度和容量仅是本次现场的验证值。

安装脚本固定原源码SHA及替换锚点，以拒绝未知版本并支持备份、重复执行；这是版本保护。650/64MB通信缓存、任务队列1和P/D拓扑属于显式部署参数，并非自动调优结果。兼容性只验证了本文固定源码和Ascend非HiSparse路径；升级SGLang或切换pool实现时，需要重新核对实际分配方式。

### 从 vLLM 迁移时的显存预算差异

vLLM Ascend的`determine_available_memory`先执行`profile_run`，按非KV内存计算余量；本镜像SGLang的`KVCacheConfigurator._profile_available_bytes`按当时空闲显存减去静态比例预留量。因此相同逻辑拓扑和相近比例不能保证相同KV预算，首次通信与算子动态分配还要实际核对。融合MoE与分离的DeepEP/GroupedMatmul路径也不能仅按EP/DP/TP数字视为同一工作点。

本镜像已核对：NPU的`moe_ep`为独立通信组，`DEEPEP_HCCL_BUFFSIZE`用于default/MoE组显式配置，其他组使用`HCCL_BUFFSIZE`；D的小batch通信域可以独立预算。上面的650/64MB是本次通过的工作点，不是所有模型和负载的通用最优值。

已定位本镜像的DSA index预算缺陷：默认容量计算沿用GPU的128B量化index加4B scale，而`NPUMLATokenToKVPool`实际使用BF16×128维，即256B。78层每token少计9672B；204416token时少计1.8413GiB。实际分配加一paddingpage为20.9211GiB，与日志20.92GiB一致。[修复脚本](deployment/ascend_memcache/fix_ascend_dsa_budget.py)为Ascend/非HiSparse路径使用真实index dtype字节数，GPU/HiSparse分支保留原计算。原源SHA为`2a38f9aeaad230e4b511861738b972ed9bbde507e72dfa579229a6a9cb197fb3`，修复后为`9eea6885eeb86936f094239eade5d340cdfc8d0274b02eeb07218010e26b0c6b`；六台实际源SHA一致，修复后D容量186368token、KV约19.08GiB，分配后空闲约6.13–6.37GiB，与精确字节预算吻合；80K/600与跨P节点L3回放均已通过。

这里的Ascend MLA池是NPU显存中的L1 KV/index缓存；Host HiCache是主机RAM中的L2；Ascend MemCache独立DDR提供者构成共享L3。它们是不同层次的缓存，模型权重另从本地NVMe加载。

## 10. R0129/R0130 的迁移边界

历史vLLM实验对PD地址描述符施加range cap；已测的64KiB对256KiB有较低搬运延迟，在原80K/60K输出、cc16负载中也观察到TTFT与吞吐改善。当前SGLang使用MemFabricSDMA，原连接器、超时变量和性能数字不能直接迁移。

本次4P2D保持cap=0，保留原描述符；请求窗口捕获46条SDMA调用，全部ret=0，单调用最大883.336ms。记录真实搬运bytes、段数、构造耗时、SDK调用耗时与返回值。若传输占比值得优化，再做单独的64KiB对照，固定请求、缓存状态、拓扑、SDK和解码策略。64KiB是候选值，不是当前SGLang已验证的最优值。原多线程分片K=4曾在真实vLLM实验中严重退化并失败，本次不引入。

## 11. 停止与回滚

先停止接受新请求并确认 `/get_load` 中 `num_reqs`、`num_waiting_reqs` 都为0；停止模型 → DDR提供者正常close → Meta/PD store/router → 确认全部NPU无遗留计算进程与端口无冲突。服务容器与模型容器分开，关闭P时不会无意关闭整个L3池。

保留原镜像和原2P2D的完整启动配置。失败后恢复**整个匹配的原cohort**，重启Meta、DDR、P、D和Router并做真实生成验证；PID1为sleep的容器仅 `docker start` 不会自动恢复推理服务。修改 TP/DP/EP、权重或KV schema时必须更换缓存namespace，避免复用不兼容的L3对象。

本部署工单的现场操作已由用户单独授权；它不自动恢复其他研究任务的设备操作，也不把旧性能结果写成新研究结论。

## 来源

- [SGLang Ascend GLM-5.2 部署教程](https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/glm_5_2.mdx)：架构与Ascend PD部署参考；不能把GLM-5.2的结果当作本模型验收。
- [Ascend MemCache](https://gitcode.com/Ascend/memcache)：SDK与分布式KV存储引擎。
- [SGLang MemCache接入PR26043](https://github.com/sgl-project/sglang/pull/26043)：本次固定源版本的基础，尚不代表所有SGLang版本均可直接安装使用。

- [vLLM Ascend KV预算实现](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/worker/worker.py)。
- [SGLang Ascend环境变量](https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/ascend-npus/reference/environment_variables.mdx)。
