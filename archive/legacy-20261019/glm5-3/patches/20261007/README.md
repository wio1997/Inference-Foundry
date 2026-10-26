# GLM-5.3 调试结果 patch 包 — 2026-10-07

MTP 图模式单独发布为 [H11-mtp-graph.patch](H11-mtp-graph.patch)。本目录收录 **13 个独立代码/配置 patch，加 2 个原始字节溯源版本，共 15 个 patch 文件**。每项都有基线/候选 SHA256、依赖和已有实验结论；它们不是可一次全量启用的性能栈。

用户已要求**暂停两台机器的代码调优和测试**。本次只做本地打包及 GitHub 同步：没有连接服务器、安装代码、重启服务、提交模型请求或新增 NPU Run。恢复机器上的调优/测试须由用户另行明确要求；历史 checkpoint 的 next_action 不构成恢复授权。

下载：[完整 zip](../GLM53-patches-20261007.zip)、[zip SHA256](../GLM53-patches-20261007.zip.sha256)。包内有 [manifest.json](manifest.json)、[SHA256SUMS](SHA256SUMS)、[本地校验结果](verification.json)、[校验工具](verify_bundle.py) 和原始源码字节 fixture。后者供应用/回退与源码比对使用，不是部署脚本。

## 已有结果与启用状态

百分比是对应研究 Run 中完整标准 PD 请求的 D 端指标；P 端漂移另行记录。各次基线/Graph 模式不同，收益不能相加，也不能迁移成当前完整 stack 对 stock 的累计收益。完整 API、正式 SLA、稳定性与最终产品 Current 尚未验收。

| Patch | 根因与改动 | 已有验证 | 性能裁决 / 当前处理 |
| --- | --- | --- | --- |
| [H6](H6-mc2-capability-cache.patch) | MC2 dispatch/combine 重复查 V4 capability；固定 worker 生命周期缓存两处 bool | Run256/258，native 分支/并发、exact 自然 EOS、完整 PD | **已验证研究性能候选**；D TPOT 降低 13.62% / 14.20%，D wall 降低 14.36% / 14.09%；保留 H6 |
| [H5](H5-same-stream-moe-events.patch) | shared expert 同 stream 路径记录无跨 stream 消费者的事件；按 overlap 条件省去 | Run253/261，依赖/ownership、exact EOS、完整 PD | **已验证研究性能候选**；H6 基线上 D TPOT 降低 6.77% / 7.04%，D wall 降低 6.98% / 6.94%；保留 H5 |
| [H4](H4-moe-gather-into-tensor.patch) | 条件允许时 list gather 改为 `all_gather_into_tensor` | Run251 native byte/layout/lifetime 与 PD；Run260 对照 | 完整收益未重复；停用 |
| [H8](H8-indexer-projection-reuse.patch) | 同一 forward 内 indexer 复用已计算投影 | Run262 CPU、all16 native/exact、匹配工作量的完整 PD | 两对收益不重复；停用 |
| [H9](H9-sfa-stream-ordered-replay.patch) | 受限 SFA replay 同 stream 的 host synchronize 改为 stream 顺序 | Run266 exact IDs/EOS/KV、all16 guard | D saving 0.883 / 8.071 ms，小于 17.918 ms 漂移；停用 |
| [H10](H10-early-padded-mtp.patch) | 同步记账分支提前提交 padded MTP | Run267 CPU guard/state；实际现场 async=true，候选分支未执行 | **NOT_APPLICABLE**；无候选设备正确性/性能结果，归档供复查 |
| [H11 / MTP](H11-mtp-graph.patch) | 合并 draft model / logits / sample Graph；持久 DCP metadata；避免嵌套外层 capture | Run273 all16 实际 capture/replay、8 token 短请求、自然 23 token EOS、KV/PD | **限域设备正确性通过**；Run275 11/11/11/11 工作匹配，D saving −2.313 / −9.623 ms，没有重复收益；默认关闭 |
| [H12](H12-rope-contiguous-cache.patch) | 不可变 split RoPE table 在注册时一次 contiguous | Run279 all16 布局/生命周期、短/EOS/PD | Run280 工作量混杂，无重复完整收益；停用 |
| [H13](H13-local-mc2-prepare.patch) | 只构造本 TP rank 的 slice/padding；空 rank fresh zeros | CPU 3888；Run283 all16 实际 capture/replay、byte/stride/ND/mask/input 隔离及短/EOS/PD | Run284 首对匹配 D wall 增加 39.982 ms；第二对工作量混杂；INCONCLUSIVE，停用 |
| [F1](F1-kv-failure-logical-block-ids.patch) | KV 传输失败回报 logical block IDs，避免展开后的 physical IDs | Run264/265 失败归约、失败请求不执行模型、clean PD/all16 transfer | 限域功能修复；不计性能收益 |
| [F2](F2-openai-empty-terminal.patch) | Chat/Completions SSE 在无 text/token 时仍处理 error/finish 终态 | Run265 canonical 500 与 clean exact PD | 限域功能修复；完整 API 覆盖尚未完成 |
| [B1](B1-aisbench-cache-isolation-adapter.patch) | 每轮 fresh salt；分 P/D endpoint 记录 HBM/external query/hit delta；补齐 5 个 adapter helper 和 phase_runner 依赖 | 先前源码/部署 hash 与 CPU 检查，未做正式模型压测 | benchmark 正确性修复；本次同时把实际 helper 源码发布到 main |
| [B2](B2-aisbench-80k-600-93-workload.patch) | 正式条件 80,000 输入 / 600 输出 / 93% prefix / 93% declared KV hit；SLO 默认 600 | 先前配置与 GitHub 同步、源码检查 | **main 已应用**；此 patch 仅供旧基线复现，不重复应用 |

[H11 原实测增量](provenance/H11-exact-tested.patch) 和 [H13 原候选字节](provenance/H13-exact-source.patch) 是对应主 patch 的溯源替代版，不能与主 patch 串联应用。H13 主 patch 保留原基线 CRLF，使 diff 只呈现实际代码变化；原候选 LF 字节另行保留。

## 应用基线与依赖

这些 patch 对应现场使用的源码快照，并非针对任意最新版 upstream 的通用补丁。**以 manifest 中每个文件的 before SHA256 为应用前提**；同一仓库的候选也可能来自不同阶段的基线。SHA 不匹配时先复查源码和依赖，不强行覆盖。fixtures.zip 提供每个 patch 独立的 before/after 文件供比对，不能替代安装匹配的完整依赖环境。

| Patch | `git apply` 所在仓库根目录 | 必要条件 |
| --- | --- | --- |
| H6 | op-plugin 子仓库 | Ascend/pytorch `5dd8ef3f9b375b5ae4a83538d5785754148c3302`，op-plugin `8b9c8534fa41eff367a41c155843daa530ab3a08`；需匹配 torch_npu/CANN 编译安装 native 库 |
| H5、H4、H8、H9、H10、H11、H12、H13、F1 | vllm-ascend 仓库根 | 对应现场文件 SHA；匹配 vLLM / torch_npu / CANN 和 SFA/DCP API |
| F2 | vllm 仓库根 | 两个 OpenAI serving 文件一并应用；PD 失败场景与 F1 配套 |
| B1、B2 | Inference-Foundry 仓库根 | B1 基于 main `74910eab…` 缺失 helper 的状态；B2 是已发布 `c0159929…` 的历史增量 |

H6 缓存包括不支持 V4 的 false 结果，要求 worker 生命周期内已加载的 CANN/custom libraries 不变。后续恢复验证时，须使用匹配的 native 构建并重启 worker；仅改 Python 或源码文件不会使已经加载的 native 库生效。本次未构建或安装该无诊断 selector 的发布版；收益证据来自 Run256 共用 rebuilt binary 的 cachemode1，它与发布版使用相同静态缓存机制。

H5 五个文件必须一起应用；`multistream_overlap_shared_expert=false` 时使用同 stream 顺序，overlap=true 时保留必要的 event、wait 和断言。其性能证据固定以 H6 为基线。H6/H5 保留为 active research performance stack，最终产品 promotion 另验。

H4 和 H13 改同一个 prepare 文件；H9/H10/H11 改同一个 runner。它们分别研究和验证，没有验证“全部 patch 同时安装”的组合。H12/RoPE 也不能因独立正确性通过就按完整 stack 验收。除了 H11 的 opt-in 开关，部分发布版候选没有独立关闭 selector；“停用”指当前不采用该候选，不是安装全部后统一设环境变量即可关闭。

## MTP 图模式：代码范围、开关和已有证据

主 patch 修改两个文件，必须一起应用：

- `vllm_ascend/spec_decode/llm_base_proposer.py`：为 GLM greedy K1 增加合并 Graph；在 metadata builder 前准备持久 block table；核验 tensor 指针、shape、stride、KV/cache/Graph 上下文；输出 clone 具有独立 ownership。
- `vllm_ascend/worker/model_runner_v1.py`：H11 合并 Graph 已承担外层 capture，避免再对 drafter model 套嵌套 `BreakableACLGraphWrapper`。

恢复设备验证时的明确 opt-in 条件如下；这些是配置说明，本包不启动服务：

```text
VLLM_ASCEND_GLM_K1_MTP_GRAPH=1
target model: FULL graph
speculative_config:
  {"method":"deepseek_mtp","num_speculative_tokens":1,"enforce_eager":false}
```

环境变量缺失时默认关闭。受支持范围是 `glm_moe_dsa`、greedy K1、TP16/DCP16/EP、DP=PP=PCP=1、精确 `NoopOffloader`；LoRA、结构化输出、随机 generator、SP、压缩/MRoPE/XDRoPE/MM、动态 EPLB、LMHead TP 等不在已验证范围。实际限制以源码 guard 为准；不符合运行时 contract 的调用回退 eager。

Run273 已完成所有 16 rank 的真实 capture/replay，并逐 token 验证短请求和自然 EOS 的标准 PD。Run275 的完整 A/B/A/B 工作量匹配，但 D saving 为负，两对均未证明性能收益。因此 **“MTP 图模式已经实现并限域正确”成立；“MTP Graph 已加速或已产品验收”不成立**。首个 target prefill/qlen1 eager 路径和普通后续 target Graph 是另一问题，H11 不宣称消除该首步成本。

发布版与原实测快照的关系：

| 版本 | proposer after SHA256 | runner after SHA256 | 区别 |
| --- | --- | --- | --- |
| 主 H11 patch | `9bd8d2fc6ad292921df6b49125de993ec68567471075480f49872c3de53a55dd` | `20be0f27426d5a38e4bbbe71146b36bb40de8cd720f04a5911389eb3d905c930` | proposer 完全相同；runner 在干净 eaa40a 基线上重放原实测单外层 capture hunk |
| H11-EXACT | 同上 | `4c90054ad40f4c430c2562bc1603aacb497684429988bf1a60edce585d6d2923` | 对原 d017c2 实验基线应用；基线包含停用的 H10 诊断 selector |

主 H11 runner 的 `_dummy_run` 整函数 AST 与原设备 candidate 相同；包装校验另外验证 proposer 字节完全相同。主版移除了对无关 H10 诊断基线的依赖，**本次没有对这份干净组合做新的 NPU 测试**。精确原实测增量另附用于复现，不能把主版整文件 hash 写成实测版 hash。

H13 同样保留原字节溯源：主版仅把原候选 LF 还原为基线 CRLF，整模块 AST 相同；原候选 SHA 为 `7bb35edfe568bb62b3432c3406e0a66d23202cd1608bebcb1ce1016789d73dbb`，主版 SHA 为 `10e6f7a69238b3e9d5291398bae1e03a6da745617b8acb0422d2b9d8e07161bc`。设备验证使用双 bank 诊断 shim 调用相同 production helper AST；发布版不带 controller、bank selector、额外 Graph pool 或 observer，不能据此宣称完整 stack 已验收。

## 本地检查、应用和回退

下载并解压后，先在本地运行包装校验。工具只在临时目录应用/回退 patch、核验原始字节/hash、解析 Python AST；不导入模型代码、不编译 native、不访问网络/服务器、不使用 NPU：

```bash
python3 glm5-3/patches/20261007/verify_bundle.py
```

`verification.json` 是本次本地校验输出，不是模型正确性或性能报告。SHA256SUMS 覆盖目录内除自身外的文件；下载 zip 的校验值单独提供，避免循环自校验。

以下以 H11 为应用说明；`PATCH_DIR` 替换为本机解压后的绝对路径。**当前两台机器暂停调优/测试，不执行安装、配置切换、重启或复测。**

```bash
PATCH_DIR=/absolute/path/to/glm5-3/patches/20261007
cd /absolute/path/to/matching/vllm-ascend
# 先核验 manifest 中两个文件的 before SHA256，保存本地改动。
git apply --check "$PATCH_DIR/H11-mtp-graph.patch"
git apply "$PATCH_DIR/H11-mtp-graph.patch"
# 回退同一个 patch：
git apply --reverse --check "$PATCH_DIR/H11-mtp-graph.patch"
git apply --reverse "$PATCH_DIR/H11-mtp-graph.patch"
```

代码进入其他分支或依赖环境后，仍需 correctness → matched A/B/A/B → 完整 PD E2E；最终完整 API、正式 SLA、稳定性和整体 stack 收益另外验收。本包没有把未重复收益的候选升级到研究性能栈，也没有撤回已经验证的 H6/H5。

## AISBench 与缓存命中条件

正式条件固定为 **80,000 输入 / 600 输出 / 93% 共享前缀 / 93% 声明 KV 命中条件**，K=1000。93% prefix 和 93% declared hit 分开记录；必须保留实际 tokenizer/chat template 后的长度、warmup 和缓存计数。

B1 每轮 fresh UUID salt，warmup/full 同一 salt，避免上一轮留下的 KV 抬高本轮命中；按 P/D endpoint 区分 HBM 与 external prefix query/hit delta。D external KV transfer 接近 100% 不能代替 P prefix HBM 命中率；目标 93% 不能直接填进 measured hit。salt 隔离不等同证明所有 cache/allocator 状态全冷。

main 的上一提交 `74910eab…` 仅有 cache 隔离 README。本次除发布 B1 patch 外，**实际同步** `default_api.py`、`dataset_generator.py`、`prefix_bench.py`、`hit_rate_collector.py`、`result_writer.py` 五个 adapter 源码及其直接依赖 `runtime/phase_runner.py`，保留已发布的 config 和 600-token SLO。对应目录说明见 [AISBench README](../../adapters/aisbench/README.md)。先前外部 legacy driver 和本仓库 adapter 是不同文件，不能宣称整个 legacy driver 与这里逐字相同；manifest 给出本仓库实际字节及对应的已有部署证据。

B2 已在 main 应用，历史冻结 Run 的原长度和结果不改写。本次没有新的 AISBench 压测，不宣称正式 93% 实测命中、SLA 或稳定容量达标。

## 证据与复现边界

[manifest](manifest.json) 列出每项 source snapshot、每个文件 before/after SHA、patch SHA、Run 引用与裁决；[evidence](evidence/) 提供关键 Run 的紧凑摘录，每份附原 reduced 文件 SHA 和固定到 research commit `7ddaabc8` 的 GitHub 来源链接。原始 trace/失败 Run/裁决沿研究分支保留，此包没有加入新大 profile 或修改历史结果。

源码 fixture 保留原有许可头。包内各 patch 的依赖和授权范围承接对应 upstream 项目。本阶段没有新增代码级性能 KEEP。
