# GLM-5.3-Flash W8A8 在 910B1 上的 SGLang 适配与复现

dsh 交接时使用 **G32 源码覆盖层、G316 P/D 服务、G565 路由**。历史 32K、64K 用 G181，128K、256K 用 G316，两代复用同一个 G32。G181 补齐可写缓存目录，G316 将 P bootstrap 和 D waiting 超时从 300 秒调到 3600 秒，G565 将路由探测改成 /ready。 [迁移与目录修复](013-matched-host-SG-migration.md) · [G316 配置与验证](022-first-128K-warmup-failure-and-timeout-repair.md) · [G565 路由修复](028-G316-first256-cold-router-failure.md)

## 六项改动与源码

构建器按文件名顺序应用 01–05，再装入新增文件。框架补丁涉及 **17 个文件**。清单固定每个补丁目标的修改前后 SHA256，及 28 个运行时新增文件的来源和 SHA256。第六项直接装入仓库已归档的最终源码。 [构建与指纹清单](../environment/glm53_g32_source.json)

| 改动 | 实现及作用 | 源码 |
| --- | --- | --- |
| 权重加载和 mHC | 映射导出权重的 attn_hc、ffn_hc 和 forget_gate 命名，接入 mHC，并通过严格加载账本核对各 rank 的真实分派。NextN 使用自身 checkpoint 的 embedding 和 head，由原生加载器完成 TP 分片。 | [01 加载与 mHC](../patches/glm53-g32/01-loader-mhc.patch) · [加载账本](../code/glm53_g32/glm53_sglang_adapter/ledger.py) · [NextN 权重绑定](../code/glm53_g32/glm53_sglang_adapter/checkpoint_bindings.py) |
| KPool 与稀疏 MLA | 给压缩索引器接上 NPU 路径，为 target 和 draft 构造对应的 KPool。适配零 RoPE 维度、压缩页和槽位映射、尾部缓存及 MLA 的读写接口。复用 vLLM Ascend 供体算子，接回 SGLang 的调度和缓存分配。 | [02 KPool/MLA](../patches/glm53-g32/02-kpool-mla.patch) · [页池](../code/glm53_g32/glm53_sglang_adapter/pool.py) · [压缩索引桥接](../code/glm53_g32/glm53_sglang_adapter/bridge.py) · [供体来源与指纹](../code/glm53_g32/vendor_provenance.json) |
| KDA | 将 GLM 线性注意力分派到 Ascend KDA。prefill 和 target verify 共用门控张量布局检查，只展平显式的 head 和 head_dim 尾部，保留 token 轴。 | [03 KDA](../patches/glm53-g32/03-kda.patch) |
| NextN/MTP 和图 | 让 draft extend、target verify 和接受/回滚使用匹配的缓存槽位，为捕获和重放补齐位置、padding 与批次桶元数据。 | [04 MTP/图](../patches/glm53-g32/04-mtp-graph.patch) · [图批次桶](../code/glm53_g32/glm53_sglang_adapter/mtp_graph_bucket.py) · [缓存端点](../code/glm53_g32/glm53_sglang_adapter/decode_endpoint.py) · [draft 池入口](../patches/glm53-g32/02-kpool-mla.patch) |
| P/D 传输与显存预算 | 注册并发送 draft KPool 尾状态，避免重复注册，修正零字节描述符和非空注册的错误处理。提前初始化既有通信资源，让 KV 预算计入其占用，再对 P 侧空闲分配器缓存做条件回收。 | [05 P/D 与显存](../patches/glm53-g32/05-pd-memory.patch) · [draft 尾状态](../code/glm53_g32/glm53_sglang_adapter/pd_drafttails.py) · [通信预算](../code/glm53_g32/glm53_sglang_adapter/comm_profile_g24.py) · [P 显存余量](../code/glm53_g32/glm53_sglang_adapter/runtime_headroom_g29.py) |
| FP32 Sinkhorn 融合 | 在 HC4 下将 20 轮 FP32 Sinkhorn 循环合到一个 Triton kernel，保留投影、残差混合及迭代公式。 | [独立优化补丁](../patches/exact-sinkhorn.patch) · [mHC 最终源码](../code/sinkhorn/mhc_candidate.py) · [融合算子](../code/sinkhorn/fused_sinkhorn.py) · [组件证据](../evidence/sinkhorn_component_acceptance.json) |

仅覆盖 Python 运行时，Ascend/CANN 二进制依赖取自固定镜像。 [SGLang 许可证](../code/glm53_g32/LICENSE-SGLANG) · [供体许可证](../code/glm53_g32/LICENSE-VLLM-ASCEND)

## 固定环境、权重与配置

| 项目 | 历史身份与要求 |
| --- | --- |
| 硬件 | 两台独立 aarch64 主机，各 8 张 Ascend 910B1 64 GiB。Device RDMA 网络互通，使用宿主 Ascend 驱动及固件。历史两端 npu-smi 显示 26.1.1，驱动与固件版本文件未冻结，复现时应补记。 |
| 镜像 | SGLang Ascend daily A2，按[镜像 digest 与 arm64 image ID](../environment/glm53_g32_source.json)核对。SGLang 0.5.24.dev536+g8a0edcf79，CANN 9.1 V100R001C11SPC001B243，torch_npu 2.10.0.post6。底座源码为 commit 8a0edcf799d06bdee8832356d382495e7fb50f22。 |
| 权重与 tokenizer | 从授权渠道取得原 Eco-Tech GLM-5.3-Flash W8A8 快照，含 84 个 safetensors 分片、索引、配置和 tokenizer。P/D 使用同一快照。[环境合同](../environment/contract.json)固定历史模型内容指纹，装配程序核对[配置、索引及关键 NextN tensor](../code/glm53_g32/nextn_checkpoint_identity.json)。其他分片和 tokenizer 随同一授权快照提供。 |
| P/D 通用 | TP8/EP8/PP1，BF16 计算和 KV，modelslim W8A8，DeepEP auto、INT8 dispatcher，page 128、context 263296、max-running 16、mem 0.92，NEXTN 3 steps/top-k 1/4 draft tokens，开启 overlap 和前缀缓存。 |
| P | chunk 8192，eager，开启显存余量保护。服务 19083、bootstrap 8998、MemFabric store 8678，由 P 服务创建 store。 |
| D | chunk 1024，target full graph 最大 batch 16，并使用 NextN draft 图。early 分配，开启 D radix cache，服务 19084。 |
| 路由 | 18080，原生 P/D router，最大并发 16、排队 100、请求超时 3600 秒、禁用重试，健康端点 /ready。 |

[P 配置](../code/sg_pd_g316/candidate_p.json) · [D 配置](../code/sg_pd_g316/candidate_d.json) · [P 实际启动清单](../evidence/matched_SG_G316/readiness/2201/launch_manifest.json) · [D 实际启动清单](../evidence/matched_SG_G316/readiness/2202/launch_manifest.json)

## 从固定镜像装配

在分配给本任务的 P/D 主机上，分别从本分支 checkout 根目录操作。要求 Bash、Python 3.11+、GNU patch、Docker、ip、flock、timeout 和可用的 910B1。将 TASK_DIR 设为本次独立任务目录，MODEL_DIR 设为授权的模型目录，COORD_DIR 设为节点现用的共享资源锁目录。三者均为绝对路径。按[网络配置示例](../environment/glm53_g316_network.example.json)填入实际 hostname、RDMA IPv4 和网卡名，两台机器的 TASK_DIR 下各保存同一份 network.json。

~~~bash
mkdir -p "$TASK_DIR"
IMAGE=$(python3 -c 'import json; print(json.load(open("environment/glm53_g32_source.json"))["base_image"])')
EXPECTED="sha256:7da963869ee54984dfeea2445befdf181710ef846448466db2361247759904ec"
docker pull "$IMAGE"
test "$(docker image inspect --format '{{.Id}} {{.Architecture}}' "$IMAGE")" = "$EXPECTED arm64" || exit 1
(
  set -eu
  BASE_COPY=$(mktemp -d "$TASK_DIR/image-source.XXXXXX")
  COPY_CID=$(docker create "$IMAGE" /bin/true)
  trap 'docker rm "$COPY_CID" >/dev/null' EXIT
  docker cp "$COPY_CID:/sgl-workspace/sglang/python/sglang" "$BASE_COPY/"
  python3 scripts/prepare_glm53_g32.py assemble \
    --base-sglang "$BASE_COPY/sglang" \
    --output "$TASK_DIR/runtime_sglang_g32" \
    --model-dir "$MODEL_DIR"
)
~~~

构建器核对底座、应用补丁、复核成品指纹、验证 NextN 权重 tensor 并写入新的校验回执。镜像提取只创建临时容器。仅审查源码可改用固定 commit 的 python/sglang，给 assemble 加 --source-checkout 并省去 --model-dir，此模式用于离线验证。

## 生成配置并启动

沿用前一节的变量，P 节点设 ROLE=p，D 节点设 ROLE=d。配置生成器将原模板的主机占位符换成本地配置，并保留完整 argv 和环境变量。先预览，生成时省略 --plan。 [配置生成器](../scripts/prepare_glm53_g32.py)

~~~bash
python3 scripts/prepare_glm53_g32.py configure \
  --role "$ROLE" --task-dir "$TASK_DIR" --model-dir "$MODEL_DIR" \
  --coord-dir "$COORD_DIR" --network "$TASK_DIR/network.json" --plan

python3 scripts/prepare_glm53_g32.py configure \
  --role "$ROLE" --task-dir "$TASK_DIR" --model-dir "$MODEL_DIR" \
  --coord-dir "$COORD_DIR" --network "$TASK_DIR/network.json"
~~~

有资源权限的 root 在两端各自的终端运行控制器。控制器检查主机、网卡、端口和 8 卡占用，使用共享锁；权重与 G32 只读挂载，运行日志与缓存写入本次目录。遇到已有容器、配置目录或资源冲突，应先查明现有归属。

~~~bash
python3 "$TASK_DIR/sg_pd_g316/node_controller.py"
~~~

P/D 的 /ready 均返回 200 后，在 P 的另一个终端运行路由，再从第三个终端检查 /workers 列表。

~~~bash
bash "$TASK_DIR/sg_pd_g316/run_router.sh"
~~~

~~~bash
curl -fsS http://127.0.0.1:18080/workers
~~~

## 验收和恢复

获得业务验证授权后，从 checkout 根目录发送一条历史短题请求，并把响应保存在本次任务目录。期望 HTTP 200、最终答案为 4，响应含合法 usage。历史记录中响应耗时约 112 秒，首次启动应留足请求时间。 [请求体](../evidence/matched_SG_G316/functional/math_request.json) · [原始响应](../evidence/matched_SG_G316/functional/math_response.json)

~~~bash
curl -fsS --max-time 600 -H 'Content-Type: application/json' \
  --data-binary @evidence/matched_SG_G316/functional/math_request.json \
  http://127.0.0.1:18080/v1/chat/completions > "$TASK_DIR/math_response.json"
~~~

- 核对两端 /ready、/server_info 与路由的两个健康 worker；检查 native_readiness.json 和 launch_manifest.json，确认导入的模型、指纹及 argv。target 和 draft 各 8 rank 的账本均应无缺失、无未分派权重，非空 P/D 内存注册应返回 0。
- 检查 D 日志的 target verify 及 draft decode/extend 图捕获，核对真实生成使用图执行。获准发送请求后，经路由验证短生成、边界请求，以及 32768 输入、32 输出的 P/D 请求。 [G316 原始功能结果](../evidence/matched_SG_G316/functional/result.json)
- 失败时保留 server.log、state.json、首次响应、权重账本和缓存证据。回退时先排空本次请求，退出本次路由，确认归属再由原控制器响应 stop.request。确认容器停止、八卡释放后，使用新任务目录重新装配，保留失败现场。
- 历史 G316 通过选定功能项。128K 的无编译证据有限，256K 首次 cold 为 18/32 成功、14 条路由 503，hot 未提交。G565 的长负载效果待验证。 [128K 证据更正](025-G316-D-compile-root-correction.md) · [256K 首次失败](028-G316-first256-cold-router-failure.md)
- 本次新增归档的验收范围为干净源码重建、指纹核对和 P/D 参数生成。真实镜像上的模型加载、P/D、图和性能结论来自历史记录，新装配方案仍待新 NPU 容器复验。 [离线复核记录](../evidence/G32_offline_reconstruction.json)
