# GLM-5.3-Flash W8A8 的昇腾 910C 适配

本分支提供 **DeepInfer 的实验性适配源码、运行环境记录和验证材料**。这是独立创建的分支，不继承本仓库 main 的内容或历史。

源码基于 [DeepInfer 上游](https://github.com/deepseek-ai/DeepInfer) 的 `553741557a57df44a46fb3ab72b6aafeaf945337`，叠加本次适配。上游 Apache-2.0 许可证保留在 [DeepInfer/LICENSE](DeepInfer/LICENSE)。本分支未包含模型权重、第三方子模块实体及六个上游图片资源。

**当前状态：已冻结源码有下述硬件验证；便携请求、rank 入口和参数化运行器已做限定硬件验证；复现仍依赖操作者提供兼容的资源登记和预算策略。整体精度、性能及生产部署均未验收。**

## 目录导航

| 目录或文件 | 内容 |
|---|---|
| [DeepInfer/](DeepInfer/) | 包含适配修改的源码快照 |
| [portable/](portable/) | 参数化请求构造、分布式运行器和结果对比脚本；验证范围见下文 |
| [validation/](validation/) | 算术、多轮记忆及状态恢复验证材料，含源码文件哈希清单 |
| [OPERATORS.md](OPERATORS.md) | 算子与模型接口适配清单：改动、源码位置、实际后端和验证限制 |
| [SUBMODULES.md](SUBMODULES.md) | 上游第三方子模块的固定版本 |

## 已验证的开发环境与镜像

| 项目 | 实测记录 |
|---|---|
| 主机架构 | aarch64 |
| NPU | 昇腾 910C；运行时设备标识 `Ascend910_9382` |
| 驱动版本 | `25.5.0` |
| 基础镜像标签 | `quay.io/ascend/vllm-ascend:v0.30.0rc2-a3` |
| PyTorch | `2.10.0+cpu` |
| torch-npu | `2.10.0.post7` |
| flash-linear-attention-npu-a3 | `26.9.1+deva4a7958` |
| CANN、自定义 OPP | 使用上述镜像中的组件；精确 CANN 版本仍待补充核验 |

已记录的**本地镜像 ID**为：

```text
sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b
```

该值不是已核验的镜像仓库 manifest digest，不能直接当作 `docker pull ...@sha256:...` 的地址。镜像标签可能变化；以上版本仅描述实测环境，尚未验证镜像重建流程，也不代表换镜像或依赖组合后仍然通过。

## 模型权重与源码身份

权重以只读方式使用，不随分支分发。复现前应核对自己的模型文件与以下记录是否一致：

| 对象 | SHA-256 |
|---|---|
| 量化权重索引 | `dfaf0918f4c088c98b011def38475d9b1cb395f35ed5e279abc0c57610e23be4` |
| 模型配置 | `33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943` |
| 对话模板 | `34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891` |
| 实测模型源码清单 | `7b6f73ccab6987daec9d29fdc46804a7a67a25c684222d5310914fb36c8b1857` |

单文件哈希见 [validation/source-files.sha256.json](validation/source-files.sha256.json)。`portable/` 在实测源码快照之后修改；请求构造器和 rank 入口已按下述范围进行硬件验证。

## 已验证的范围及结果

验证使用四个协同 rank，执行完整 45 层模型，实验上下文上限为 128 token。

- **算术案例**：DeepInfer 与参考路径的生成 token 一致。
- **多轮记忆案例**：最终答案均为 `42`，但参考路径与 DeepInfer 的推理序列存在差异，原因尚未查明，不能据此宣布整模型精度通过。
- **生成中途状态恢复**：四个 rank 的 archive-v2 恢复能够复现 44 个生成 token；关闭缓存和启用 128 MiB 缓存两种配置均有验证材料。
- **缓存**：观测到的缓存命中次数为零，没有证明复用缓存带来提速。
- **停止条件**：生成停止标记 `154827` 与 tokenizer 的 EOS `154820` 是两个不同的 token；自然命中配置的停止标记不应写成命中 EOS。

上述结果仅覆盖具体的短请求和配置，不代表独立整模型精度、性能或生产可用性已验收。生产级 TP/EP、批处理、服务生命周期及长上下文质量仍未验证。

## 便携运行入口与复现边界

[portable/runner.py](portable/runner.py) 是操作者持有的参数化分布式运行器（两 rank 和四 rank 固定短例已实测）：它使用既有共享锁和登记文件，核对分配、卡占用、镜像及源码/权重哈希，调用操作者的有界预算 helper，启动并登记进程，确认宿主 PID 到 card/chip 的映射后释放 `go` 屏障。它不停止其他工作负载。运行器配合 [launcher.py](portable/launcher.py) 与 [worker.py](portable/worker.py) 在四 rank 上执行一次短多轮恢复：每个 rank 的 prompt argmax 与 44 个生成 token 均匹配 round 360，archive-v2 在 42 token 后恢复，停止 token 为 154827，进程组均已销毁。脱敏结果见 [restore-round361.json](validation/restore-round361.json)。

可显式传入 `--kda-backend ascendc` 选择显式归一化的实验递推路径；默认仍为 `ascendc_installed_decode`，可通过同一选项明确回退。实际便携命令在四 rank 短多轮案例中验证了 `ascendc` 选项传播、首 token 后 archive-v2 恢复和自然停止，生成 45 个 token 与 round 373 一致；见 [结果](validation/portable-recurrent-round376.json)。这不是全模型质量或生产验收。

先将 [operator-config.example.json](portable/operator-config.example.json) 复制到任务目录并填写自己的**真实**路径、锁、已运行隔离容器 ID、镜像本地 ID、授权 die/card/chip 对应关系、赋权和登记 JSON、源码清单及 SHA-256。示例中的零值、路径和设备号只是占位符，不表示授权或已验证配置。`pinned_files` 必须包含可信源码清单、权重索引、模型配置与模板哈希。运行器会在执行前要求源码清单路径恰有一个匹配的 SHA-256 pin；如旧配置未填写该项，须按可信清单更新配置。四 rank 固定案例回归见 [记录](validation/manifest-pin-round400.json)。容器内外的源码、运行目录和 launcher 路径必须分别映射到同一内容；模型权重应只读挂载。

运行器要求操作者提供可信的预算 helper 和策略文件；helper 必须实现 `reserve(root, lock, goal, source_sha256, attempt_id, workers, timeout_seconds)`，并在传入的同一共享锁下原子核对授权和持久化预算。公共仓不包含现场策略或现场控制器。分配与登记 schema、设备映射、容器环境、NPU 可用性和 helper 的正确性由操作者按现场资源协议核验；没有兼容策略时不能运行。公开代码不提供服务、部署或自动抢占。

在已授权且配置完整的现场，参数化调用方式为：

```sh
python3 portable/runner.py --config /path/to/operator-config.json \
  --messages /path/to/task-root/messages.json --run-id example \
  --goal reproducible-release --context-tokens 128 --max-new-tokens 64 \
  --cache-mib 0 --expected-final 42 --restore-after-first-token \
  --experimental-fp32-dsa
```

消息文件必须位于任务目录内，且与 `--expected-final` 一致。结果写入唯一运行目录；若 worker 状态不确定，登记保留，必须先核实后接管。示例命令之外的部署、镜像重建、生产 TP/EP、服务生命周期、长上下文及独立整模型精度尚未验收。参数化参考入口只在下述四 rank 分配和固定案例上验证。

## 第三方依赖

固定版本见 [SUBMODULES.md](SUBMODULES.md)，上游地址见 [DeepInfer/.gitmodules](DeepInfer/.gitmodules)。本分支的根提交未保留子模块 gitlink，需按清单从各上游仓库显式获取对应提交，不能假定运行 `git submodule update` 就能恢复全部依赖。上述实测镜像之外的构建和依赖兼容性仍未验证。

## 四 rank 状态回滚探针

[portable/rollback_probe.py](portable/rollback_probe.py) 的四 rank 硬件结果见 [验证记录](validation/portable-rollback-round388.json)。它是实验探针，要求由操作者提供已持锁且已登记的四 rank 分配、隔离容器、超时、独立运行目录和 host PID 到 die/card/chip 的核验；探针不会停止或接管其他作业。各 rank 运行前将 `GLM_MODEL_DIR`、`GLM_SOURCE_DIR`、`GLM_ASCEND_LIBRARY`、`GLM_EXPERIMENT_RUN_DIR` 只含该 die 的 `ASCEND_RT_VISIBLE_DEVICES` 和 `WORLD_SIZE` 设置为现场真实路径、设备及 2 或 4，并以 `python3 rollback_probe.py <rank> <die> <共同端口> bounded` 启动。各 rank 创建 `rankN/ready` 后，操作者须验证宿主 PID 与授权设备映射，再分别创建对应的 `rankN/go`；启动、登记、错误处理和回收由操作者负责。权重索引、配置、运行时依赖和实际 probe 哈希必须与验证记录核对；该示例不替代现场的资源策略。

公开运行器也支持在已有配置和消息文件的调用中增加 `--rollback-probe`，使用同一互斥、登记、授权、PID 映射和 900 秒工作进程上限启动探针；仍需 `--experimental-fp32-dsa`；回滚模式不接受消息或文本生成选项。示例：

```bash
python3 portable/runner.py --config "$OPERATOR_CONFIG" --run-id rollback-check --goal reproducible-release --experimental-fp32-dsa --rollback-probe --kda-backend ascendc
```

探针模式要求显式指定 `--kda-backend ascendc`，并固定使用四个短 token；普通文本模式仍为默认模式。四 rank 实测和普通模式回归见 [运行器记录](validation/public-rollback-runner-round389.json)。

## 四 rank 参考对照

公开运行器支持 `--reference-probe`，调用容器中已安装的 vLLM-Ascend 参考实现。先在操作者配置的 `container_env` 中填写 `GLM_REFERENCE_ROOT`，指向容器内安装的 `vllm_ascend` 包目录；参考模型和预取源码必须匹配 [探针](portable/reference_probe.py) 的哈希，参考环境、权重挂载、离线缓存目录和受控预取设置必须可用。沿用同一资源锁、登记、分配和 PID 映射屏障；`--experimental-fp32-dsa` 是当前运行器要求的显式实验开关，参考模型本身不使用 DeepInfer 的 DSA 工厂。参考模式不支持 DeepInfer 缓存或归档恢复。

```bash
python3 portable/runner.py --config "$OPERATOR_CONFIG" --messages "$MESSAGES" --run-id reference-check --goal reference-compatibility --experimental-fp32-dsa --reference-probe --context-tokens 128 --max-new-tokens 6 --expected-final 13
python3 portable/runner.py --config "$OPERATOR_CONFIG" --messages "$MESSAGES" --run-id adaptation-check --goal bounded-input-cli --experimental-fp32-dsa --kda-backend ascendc --context-tokens 128 --max-new-tokens 6 --expected-final 13
python3 portable/compare_reference.py "$REFERENCE_RUN_DIR" "$ADAPTATION_RUN_DIR"
```

两次运行必须顺序完成，消息及生成配置一致。对照工具接受运行目录（含 `summary.json` 和 `rank0`–`rank3/result.json`）或已有聚合 JSON；目录哈希按文件名与内容长度及字节计算，聚合文件哈希按文件原始字节计算。实测固定 `8+5` 案例的四 rank 生成 token 一致，详见 [结果](validation/portable-reference-round392.json)；三消息回忆案例的参考序列与保留参考一致，但与适配序列不同，见 [多轮结果](validation/portable-reference-recall-round393.json)。这是单案例检查，不等同整体质量或生产验收。

回滚探针现依次验证 rank 2 的 head 异常和单 rank 输入不一致，再核对状态与重试；四 rank 实测见 [结果](validation/portable-rank-mismatch-round395.json)。同版本普通文本模式及参考模式的四 rank 回归分别见 [普通模式](validation/public-normal-round396.json) 和 [参考模式](validation/public-reference-round397.json)。

### 实验性 world size 参数

普通文本模式可显式传入 `--world-size`（默认 4，候选 2、4、8、16）；分配数量、授权、登记和有界 helper 必须匹配该值。实际硬件检查覆盖两 rank 和四 rank 的固定短例；8/16 rank 的资源、通信、显存和完整模型行为均未验证。参考探针仍限定四 rank；回滚探针允许两或四 rank。更新源码后须重新生成并固定源码清单，不能沿用旧配置的源码哈希。[四 rank 实测](validation/rank-portable-round408.json) 仅覆盖固定短案例，不构成其他规模或独立质量验收。 launcher 的 rank 边界现随 `WORLD_SIZE` 检查；模拟执行覆盖 2、4、8、16 的边界，四 rank 固定短例回归通过，其他规模尚未由该项验证。[记录](validation/rank-launcher-round410.json)。运行器允许使用任务分配内的不同 die 子集，同时保留共享锁与占用核验；两 rank 固定短例在授权 die 实测通过，提示 argmax 和生成 ID 与四 rank 一致，未比较完整 logits。[结果](validation/rank-subset-round410.json)。8/16 rank 硬件与整模型行为仍未验证。 两和四 rank 的非零状态回滚也分别检查了同步故障、输入不一致及重试；实验超时与发布版超时的差异见 [记录](validation/rollback-ranks-round410.json)。

### 实验性 rank 批量事务

`PreparedTextRunner` 在已启用 rank commit agreement 且为逐 token 栈执行时，可在创建请求前显式调用 `enable_rank_batch_commit(max_tokens=2)`；上限可设 1–8。所有 rank 必须按相同顺序启用和调用。该路径先核对元数据与 token，再逐行执行，完成同步准备后一次提交状态；不接受页面绑定。四 rank 实测了单 rank 过期位置、格式错误和 token 不一致，以及第二 token head 同步异常；45 层状态保持不变，重试 logits 与状态匹配干净逐 token 执行。测试为固定短例，不覆盖栈内 collective 异常或进程、设备故障。[探针](portable/rank_batch_probe.py) 需要操作者提供已持锁登记的四 rank 分配和 PID/die 屏障；[范围](validation/rank-batch-commit-round411.json)。 实验调用也可显式设置 `layer_major_dense_kda=True`；调用者须先将选定 dense KDA 层设为 `ascendc_chunk_prefill` 并启用 `native_chunk_prefill`，所有层保持 `native_tokenwise`。各 rank 在执行前核对层掩码；sparse 与 DSA 仍逐 token。四 rank 双行回滚通过，但同权重 recall17 的首 token 与参考仍不同（99419 对 1474，154880 logits RMSE 约 0.712）；这不是质量通过。[实测范围](validation/dense-layer-prefill-round411.json)。 对最后一行预填，可在选定 dense KDA 层上显式设置 `attention.chunk_prefill_single=True`，在调用后、解码前立即恢复 `False`；rank 预检核对该模式及层掩码。四 rank 单例交接通过，recall17 首 token 仍未与参考对齐（99419 对 1474），完整 logits RMSE 约 0.710；这不是质量通过。[验证](validation/single-prefill-round412.json)。 同一输入第0层在位置0和40的首个已捕获差异均为 attention 输出；此前 embedding 和 attention 输入逐位相等，尚不能确定原因。[边界记录](validation/single-prefill-boundary-round412.json)。 新增默认关闭的 `attention.reference_sharded_projection=True` 实验投影，rank 预检校验选定层掩码。四 die 单层投影与参考位置0匹配，但 recall17 首 token 仍不一致、参考 logits RMSE 上升；这是实验，不是精度修复。[验证](validation/reference-projection-round413.json)。 若在非 layer-major 的分布式请求中误启用该选项，rank 会在入栈前协同拒绝；旧单 token rank commit 路径为此增加一次 all-reduce，开销未测。[验证](validation/reference-projection-guard-round413.json)。 旧单 token rank commit 的本地请求校验现先协同汇总；四 rank 中一方位置失效时均在入栈前拒绝且45层状态不变。[验证](validation/legacy-rank-preflight-round414.json)。

### 实验性单层 chunk 预填入口

[chunk_prefill.py](DeepInfer/deepinfer/model_executor/models/glm5_next/chunk_prefill.py) 提供单序列、1–64 token、已投影和已卷积输入的显式 chunk 调用，返回输出与新的 FP32 状态，不修改输入状态。四 die 实测 41 token 入口及 20+21 分段的状态交接与对应安装版算子调用一致；第0位置与独立参考捕获一致。它尚未接入完整文本 runner，分段与整块的数值等价、整体模型精度和性能均未验收。[验证记录](validation/chunk-prefill-entry-round409.json)。

实验层 `Glm5NextKDA.from_checkpoint(..., backend="ascendc_chunk_prefill")` 对 2–64 token 使用四个本地 head 分片调用 chunk，单 token 切换到 `ascendc` 递推；不改变已有后端。四 die 单层使用真实 41 token 输入后接一个重复输入，验证 chunk 核心第0位置与参考捕获一致、输入状态不变及 41→42 token 状态交接。四个本地分片不是分布式 TP；该选项不接入文本 runner，也不表示整模型 chunk、质量或性能已验收。[结果](validation/kda-chunk-handoff-round409.json)。

显式实验工厂参数 `load_native_decoder(..., kda_backend="ascendc_chunk_prefill", chunk_prefill=True)` 仅允许 dense KDA 层、每次最多 8 行；DSA、sparse 专家层和完整文本 runner 尚未接入。四 die 真实权重两行测试检查了分派、状态不变和上限，并与同层显式批处理对照；默认逐 token 整模型 8+5 回归的提示 argmax 与生成 ID 保持一致。前者使用参考归一化注意力行作为确定性残差输入，不是整模型精度对照；完整 logits 未比较。[验证范围](validation/decoder-chunk-round410.json)。

当前 rank-commit 源码在四 die 上通过 archive-v2 的45层保存与恢复检查：固定 8+5 提示后恢复续写的完整 logits、IDs 和全部最终层状态一致。[实测记录](validation/archive-restore-round415.json)。[参数化 probe](portable/rank_snapshot_probe.py) 已接入发布 runner 的 `--snapshot-probe`；需提供现有 operator 配置，runner 使用配置中的互斥、登记及映射校验。固定 probe 不接受自由消息参数，worker 超时900秒。运行命令见实测记录。

当前源码的四 rank 45 层固定 7+6 请求中，逐次加载与128 MiB可复用专家会话的28个逐步 logits 均一致；会话关闭后无驻留字节或活动 lease。测试观测零 cache hit，不代表重用或性能收益。runner 的 `--cache-equivalence-probe` 已实测；固定输入及命令见 [记录](validation/cache-equivalence-round416.json)。

当前源码的四 rank 三消息 recall17 在第一个生成 token 后进行 archive-v2 恢复，生成的38个 IDs 与保留的未恢复适配运行一致，正常以配置 stop 结束；独立参考的首 token 仍不同。该检查未比较完整 logits 或全部状态。[实测记录](validation/recall-restore-round417.json)。 恢复时现在逐层比较快照与反序列化状态，并先跨 rank 确认；同一 recall17 在四 rank 的45层状态均逐位相等。

发布回滚探针现包含旧单 token 路径的单 rank 过期位置预检；四 rank 的45层状态保持与重试结果见 [实测记录](validation/legacy-rollback-runner-round418.json)。

固定 archive probe 显式要求四 rank；错误 rank 数在读取配置或认领资源前拒绝，修正后四 rank 硬件恢复回归见 [记录](validation/snapshot-rank-guard-round418.json)。

旧单 token rank 提交路径现会在入栈前比较请求身份指纹、位置和 token 数。四 rank 实测发现旧版允许不同请求 ID 提交；修正后协同拒绝并保持45层状态，详见 [验证记录](validation/rank-request-identity-round419.json)。该检查使用56位指纹，存在碰撞可能；新增 collective 的开销尚未测量。

同一 preflight 也已实测各 rank 自身位置有效但跨 rank 不一致：2/3 token 状态均保持不变，随后正常重试通过。[结果](validation/rank-request-identity-round419.json)。

单 token rank 路径在共享 preflight 中校验 CPU 整数 token 和词表范围；embedding 读取异常在入栈前跨 rank 汇总。四 rank 注入与状态重试见 [结果](validation/rank-embedding-preflight-round419.json)；新增 collective 的性能未测。

单 token rank 提交 preflight 现同时比较 token ID，已实测 rank 2 token 不一致时在 embedding 前协同拒绝；状态和重试结果见 [验证记录](validation/rank-token-preflight-round420.json)。

指定非零 reusable cache 时，rank 专家适配器现使用与请求相同的预算；64 MiB 四 rank 曾因固定128 MiB绑定失败，修正后短请求通过。零 session 仍使用原128 MiB临时 cache 上限；这些预算不包含临时副本、native 分配或 allocator 开销。[验证记录](validation/cache-binding-round420.json)。

真实 recall17 的 layer0 首8 token 实验回放中，rank0 的 position0 投影与保留参考相同，core、gated 和 attention 输出有数值差异；固定40 token长度并只改变后32个输入时，首位置 core 在各自重复中稳定但两种尾部之间不同。该单 die 检查不证明内部根因或整模型精度。[记录](validation/kda-eight-boundary-round421.json)。

在该单 die 固定输入的40 token尾部单因子控制中，单独替换后续 raw gate 即复现了首位置 core 差异，单独替换后续 Q/K/V/beta 未产生逐位差异；这不构成算子内部根因或整模型精度验收。[记录](validation/kda-eight-boundary-round421.json)。

默认关闭的 `reference_sharded_projection` 实验选项还将输出投影拆为四个本地 BF16 分片并以平衡树相加。单 die、真实 recall17 第0层 position0 的40 token回放中 core 和输出与保留参考逐位相等；8 token 仍有差异。默认选项关闭时，四 die 的短算术请求 token 结果与此前同运行时记录一致；未验证一般输入的 HCCL 等价、完整 logits、整模型精度或性能。[范围](validation/kda-eight-boundary-round421.json)。

同一当前源码的四 die recall17 首生成 token 对照中，仅切换实验性 KDA 输出分片：开启时 argmax 与保留参考均为 1474，关闭时为 99419；对参考的 154880 logits RMSE 分别为约 0.7751 和 0.8634，仍有大量不同值。该单一提示没有预设精度阈值，尚未验证后续生成或整体质量。[范围与原始指标](validation/recall17-output-tree-round422.json)。

使用上述实验路径对该 recall17 提示执行未强制生成时，四 rank 的13个生成 token 与保留参考完全一致，以配置停止 token 154827 结束，未到 tokenizer EOS 或32 token 上限；首 token 完整 logits 仍不一致。这仅是一个固定例子，不代表整体验收。[结果](validation/recall17-output-tree-round422.json)。

同一实验路径的四 die recall42 未强制生成返回最终答案42，但生成序列为48 token，保留参考为45 token，首个差异在索引2。初始32 token上限到达后扩至64，最终以配置停止 token 154827 结束，未到 tokenizer EOS 或最终上限；此例没有完整参考 logits 对照，不能判定整体精度通过。[实测范围](validation/recall42-output-tree-round423.json)。

针对 recall42 的前两个相同生成 token，四 rank 对位置42的44层输出与保留参考比较：第一个已捕获差异位于第0层，最大绝对差约0.0007324；第43层最大绝对差为2.3125。参考与候选 prefill 形状不同，未据此判定内部因果或质量阈值。[范围](validation/recall42-output-tree-round423.json)。

实验性 layer-major KDA 分块和本地参考投影路径在四 rank 的短输入中，非零状态归档恢复后全部45层状态一致；随后两 token 分块续算的 logits 和最终状态与未中断路径逐位一致，所有请求 owner 已释放。该检查不覆盖设备丢失、跨版本恢复或整模型独立精度。[范围](validation/chunk-archive-round423.json)。

便携文本命令可显式增加 `--experimental-layer-major-kda --kda-backend ascendc`，选择四 rank 下前3个 dense KDA 层的分块预填与本地参考形状投影；每组最多8 token，其余层逐 token。该选项默认关闭，不能与固定探针或参考模式组合；所有 rank 必须执行相同选项。四 rank 的实际CLI短算术案例以 `[8,8,8,2]` 分组并返回13；默认路径的提示 argmax 与生成 token 也匹配先前记录，默认回归的测试运行器仅将超时缩短至1200秒。recall42仍存在生成序列差异，未证明整体精度或性能。[实测范围](validation/layer-major-cli-round424.json)。

实际便携CLI开启实验分块后，四 rank 的 recall17 提示分组 `[8,8,8,8,8,1]`，未强制生成的13个 token 与保留参考一致，停止于配置 token 154827，未到 tokenizer EOS 或上限。该CLI测试仅将运行器超时缩短至1200秒；同例首 token 全 logits 仍存在差异，recall42 token序列不一致，不能据此宣称整体精度通过。[结果](validation/layer-major-cli-round424.json)。

实际便携CLI的实验分块与 `--restore-after-first-token` 组合在四 rank 固定8+5案例通过：位置27的 archive-v2 恢复后45层状态均一致，rank协调成功并返回预期3个 token。该CLI案例未比较续算全部logits；测试运行器超时缩短为420秒。[范围](validation/layer-major-cli-round424.json)。

实验分块的实际便携CLI recall42 输出与此前探针48 token一致，仍与独立参考45 token在索引2开始不同；最终答案42，停止于配置token 154827，未到EOS或上限。[结果](validation/layer-major-cli-round424.json)。另一个四rank独立探针对固定7+6在缓存关闭与128 MiB会话缓存之间逐位比较了每rank六个完整logits张量和生成token，全部一致；关闭后缓存零字节且无活动租约，记录到命中但未测性能。独立探针与CLI缓存测试为不同检查。[范围](validation/layer-major-cache-round425.json)，[探针](portable/layer_major_cache_probe.py)。

实际便携CLI实验分块与128 MiB缓存组合在四rank固定8+5案例中与缓存关闭的提示argmax和生成token一致；关闭后各rank缓存零字节、无活动租约。此CLI案例未比较完整logits，独立7+6探针另做了完整logits对照；均未测性能。[结果](validation/layer-major-cache-round425.json)。

实验分块、128 MiB 专家缓存和首生成 token 后 archive-v2 恢复组合，在四 rank 固定 8+5 请求中恢复了全部45层记录状态并得到预期生成 token；各 rank 关闭后缓存为零且无活动租约。此案例未比较恢复后续算完整 logits，亦未测性能。[结果](validation/layer-major-cache-restore-round426.json)。

四 rank 实验分块路径对非零45层状态进行归档恢复后，依次续算8 token与1 token；恢复路径和未中断路径的完整续算 logits、全部45层状态逐位一致，所有请求 owner 已释放。该短输入测试不覆盖设备丢失、跨版本恢复或独立整模型精度。[范围](validation/chunk-archive-tail-round427.json)。

实验性 layer-major 接口现在可显式选择 linear-attention + sparse MoE 层进行 KDA 分块；该层的 MoE 仍逐 token 调用。四 rank 实测选择第4层及三个 dense 层，两 token 的受控失败回滚保留全部45层状态，重试 logits 和状态与干净运行逐位一致；改变后的默认 CLI 固定短例也匹配先前结果。现有 CLI 实验开关仍只选择 dense 层；此结果不代表独立质量或性能通过。[范围](validation/sparse-kda-chunk-round428.json)。

便携 CLI 在 `--experimental-layer-major-kda --kda-backend ascendc` 基础上可另加默认关闭的 `--experimental-sparse-kda-layer4`，显式选择已测的第4层 sparse MoE KDA；该选项要求四 rank，MoE 仍逐 token。固定8+5实际命令四 rank 返回预期 token，未比较完整 logits 或独立整模型质量。[范围](validation/sparse-kda-cli-round429.json)。

启用第4层 sparse KDA 的实际 CLI recall17 固定例返回最终答案17，但生成4个 token，与保留参考13个 token 从首 token 起不同；所有 rank 一致且以配置停止 token 结束。这是未解决的质量差异，实验选项保持默认关闭。[结果](validation/sparse-kda-recall17-round429.json)。

同一第4层 sparse KDA 实验选项的 recall42 实际 CLI 固定例返回最终答案42，但生成4个 token，与保留参考45个 token 从首 token 起不同；四 rank 一致，以配置停止 token 结束且未到上限。未比较完整 logits。[结果](validation/sparse-kda-recall42-round429.json)。

加入第4层选项后的默认 CLI 四 rank 固定8+5回归与先前默认记录的提示 IDs、argmax、生成 IDs 和停止字段一致；未比较完整 logits。[范围](validation/sparse-cli-default-round429.json)。

四 rank 非零状态下，仅 rank2 关闭第4层 sparse KDA 分块时，所有 rank 均在 chunk 执行前协同拒绝，45层状态保持不变；恢复一致配置后重试 logits 和状态与干净运行逐位一致。该受控检查不覆盖设备丢失或整体精度。[范围](validation/sparse-kda-mask-round430.json)。

四 rank 实验分块选择第4层 sparse KDA 时，非零45层状态归档恢复后的8 token 加 singleton 续算，与未中断路径的完整 logits 和最终45层状态逐位一致；该 sparse MoE 逐 token 调用。仅固定短输入，不覆盖独立精度、性能或设备故障。[范围](validation/sparse-kda-archive-round431.json)。

四 rank 的 sparse KDA 归档恢复测试中，8 token 后仅 rank2 在 singleton head 注入故障；各 rank 协同失败且45层状态不变，重试后的完整续算 logits 和状态与干净路径逐位一致。仅受控 head 失败，不覆盖设备或通信故障。[范围](validation/sparse-kda-singleton-rollback-round431.json)。

实验性 sparse KDA CLI 在短输入、128 MiB cache 和首 token 后状态恢复组合下，四 rank 的45层状态及输出检查通过；完整续算 logits 未比较。[范围](validation/sparse-kda-cli-restore-round431.json)。

rank W8A8 输出投影支持显式启用最多40个输入 token，并将本地投影分为最多8行；默认仍为8。四 rank 算子范围验证见 [记录](validation/rank-linear-tiles-round434.json)；40 token 整模型路径尚未集成。

实验性45层40+1 token预填充在四 rank 完成，recall17 首生成 token 与保留参考一致；完整 logits 和生成序列尚未比较。[范围](validation/prefill40-round436.json)。默认仍是8 token，公共 CLI 尚未接入此模式。

40+1 预填充的 recall17 完整生成自然停止，与保留参考的前6个生成 token 相同、索引6首次不同；完整 logits 未比较。[记录](validation/recall17-prefill40-round437.json)。

40+1 预填充的 recall42 生成结果和参考 token 对比见证据；完整 logits 未比较。[记录](validation/recall42-prefill40-round438.json)。

公共 CLI 支持显式 `--experimental-layer-major-kda --experimental-kda-prefill40 --kda-backend ascendc`，可与 `--experimental-sparse-kda-layer4` 合用；单 token 预填余项走 chunk。实际四 rank recall17 结果与范围见 [记录](validation/cli-prefill40-round439.json)。默认仍保持先前路径。

当前源码默认 tokenwise CLI 的四 rank 固定输入回归与先前基线 token 和逐位置 argmax 一致；完整 logits 未比较。[记录](validation/default-cli-round440.json)。

公共 CLI 40+1 预填充的 recall42 生成与参考 token 对比见记录；完整 logits 未比较。[记录](validation/cli-recall42-prefill40-round441.json)。

实验性40 token续算在非零前缀后的受控 head 故障回滚与重试范围见 [记录](validation/rollback40-round442.json)。

公共CLI可显式使用 `--rollback40-probe --kda-backend ascendc --experimental-fp32-dsa` 运行40 token事务受控回滚验证；须按既有方式配置四rank设备互斥。[范围](validation/cli-rollback40-round443.json)。

实验性40 token预填后的45层archive-v2恢复和续算完整logits验证见 [记录](validation/archive40-round444.json)。

实验性40 token预填的专家缓存等价验证见 [记录](validation/cache40-round445.json)。

实验性 40+1 预填在固定 recall17 输入上的首 token 与保留参考相同，但完整 logits 存在差异；具体数值和范围见 [记录](validation/reference40-logits-round447.json)。

实验性40 token预填的跨rank分块上限不一致和重试验证见 [记录](validation/bound40-round448.json)。

实验性40 token预填的第4层sparse KDA独立probe验证见 [记录](validation/sparse40-round449.json)。

组合 `--experimental-kda-prefill40 --experimental-sparse-kda-layer4` 的四rank实际CLI固定例与独立参考存在生成差异，实验选项默认关闭；范围见 [记录](validation/cli-sparse40-round450.json)。

40+1 sparse KDA CLI在首生成token后进行archive-v2恢复，四rank的45层状态和无恢复路径生成IDs一致；范围见 [记录](validation/cli-sparse40-restore-round451.json)。

40+1 sparse KDA实验CLI的recall42固定例返回42，但与独立参考的生成序列从首token起不同；范围见 [记录](validation/cli-sparse40-recall42-round452.json)。

实验性稀疏KDA与稀疏KDA与40 token预填的专家缓存等价验证见 [记录](validation/sparsecache40-round453.json)。

recall17 前8 token 的第0层第0位置对比发现 core、gated 和最终 attention 均与保留参考不同，首个已测差异在 core；详情见 [记录](validation/first8-boundary-round454.json)。

recall17 前8 token 的第0层第0位置投影、门控及QKV卷积与参考逐位一致，核心输出存在差异；详情见 [记录](validation/first8-operands-round455.json)。

recall17 第0层固定前8个核心输入和零状态时，8 与 41 token 分块的第0位置核心输出不同，41 token 结果与保留参考逐位相同。[记录](validation/chunk-shape-round456.json)。

recall17 第0层将前8 token 的核心输入补零至41后，未逐位复现实际41 token的前8个输出，最终状态也不同；该候选不可据此部署。[记录](validation/chunk-padding-round457.json)。

直接启动 rank 时若未设置 WORLD_SIZE，launcher 会将校验过的默认值4传给 worker；四 rank 实机验证见 [记录](validation/launcher-world-size-round458.json)。

稀疏KDA与40+1预填的archive-v2恢复后，续算完整logits及45层状态在受测配置下相等，见 [记录](validation/sparsearchive40-round460.json)。

分配器在启动 worker 前以 `npu-smi info -m` 核对卡/芯片到逻辑 die 的映射；不匹配或无法解析时拒绝启动。[验证](validation/allocator-mapping-round461.json)。

独立 `launcher.py prepare` 支持显式 `--experimental-layer-major-kda`、`--experimental-kda-prefill40` 和 `--experimental-sparse-kda-layer4`；后两项要求前者，且该模式要求 `ascendc`。runner 验证构造结果与所请求选项一致。[四rank验证](validation/prepare-options-round462.json)。

请求构造器更新后的默认回退路径，在省略后端和实验开关时选择 `ascendc_installed_decode`，四rank固定用例验证见 [记录](validation/default-builder-round463.json)。

请求构造器更新后的四rank参考模式固定用例与保留参考输出的对比见 [记录](validation/reference-builder-round464.json)。

四rank归档请求身份不匹配的协调拒绝及探针重试验证见 [记录](validation/archive-identity-round465.json)。

稀疏层4 KDA 40+1预填的缓存、设备和进程内存采样见 [记录](validation/sparse40-memory-round466.json)。

归档准备阶段支持存活rank错误协调；注入失败与正常恢复的四rank验证见 [记录](validation/archive-prepare-round467.json)。

归档恢复后的请求准入支持存活rank错误协调；注入失败与正常恢复的四rank验证见 [记录](validation/archive-open-round468.json)。
