# GLM-5.3-Flash-W8A8：单机 Ascend 910C 移植与优化

本目录提交 SGLang Flash 配套移植、R3/R4/R5 三项 opt-in 优化，以及既有 R2–R6 的精简结果。它是独立的 Flash/SGLang 项目，不改变 Foundry 既有普通 GLM/vLLM、DeepSeek 源码、Current/KEEP 或部署状态。本次仅整理和发布，没有重新编译、运行模型或采集 profile。

入口：[公司提交评审包](lab/glm53_flash_ascend/review-packages/glm53-company-submission-20261026/README.md) · [全部服务指标与限制](lab/glm53_flash_ascend/review-packages/glm53-company-submission-20261026/results.md) · [源码与 AOT 复现边界](reproduction.md)

## 源码与四段补丁

补丁只适用于固定官方底座 `sgl-project/sglang@ce0a8fea5f4245839d174b2b8c55692be1810a7f`，按顺序应用。配套实现来源为 `wio1997/sglang-ascend-tuned@d69841962fec04820d75b4ca6994df2550728eb4` 的 `ascend_tuning/profiles/glm53-flash-eager/overlay/`。

| 补丁 | 内容 | 对应原实测源码身份 |
|---|---|---|
| [0001](patches/0001-r2-paired-flash-single-node.patch) | R2：配套 Flash backend/recurrent、模型、DSA/KPool、W8A8；joint 生命周期兼容及实际 rank 身份记录 | ddc0a988976fd0b8d85894f7f15421bd5887c552 |
| [0002](patches/0002-r3-mhc-sinkhorn-fusion.patch) | R3：仅融合 HC4 FP32 Sinkhorn 归一化尾段；整模型 B 选择 ordered TileLang AOT | ed21672733718055b29bb6b7629f8d1aaac9384e |
| [0003](patches/0003-r4-kpool-forward-metadata-cache.patch) | R4：同一 forward 复用四份 KPool CPU 元数据快照 | c17af2c3138db705fca7e9ed0631fa1efa8dd5ed |
| [0004](patches/0004-r5-sparse-kv-length-cache.patch) | R5：同一 forward 复用 Sparse NoPE KV-length 设备转换 | 49002ca8aa4c64276905a4d7973a4213f54d6902 |

[源码溯源清单](manifests/source-provenance.json)记录每段补丁、阶段文件及最终 54 个运行文件的 SHA256。离线检查已从精确底座依次重建 48/52/53/54 文件阶段。应用补丁所得 checkout 有自己的新 Git 身份，不能把其 HEAD 冒充原实测提交。

同事 47 文件 profile、原有 indexer stride 去特化和 attention-TP query 分片属于继承实现，不计为我们新增的优化。R2 是运行兼容与基线建立，不宣称算法提速。正式源码通过补丁承载，不把整个私库历史或整套引擎重复复制到此目录。

## 实测结论

| 轮次 | C1 输出吞吐观察 | C4 输出吞吐观察 | 结论及延迟范围 |
|---|---:|---:|---|
| R3 mHC | +64.365359% | +67.832562% | 最大改善；TTFT、TPOT、总时延均下降 |
| R4 KPool | +1.765464% | +0.350532% | C1 小幅观察改善；C4 两块方向相反，收益不可靠 |
| R5 Sparse lengths | +1.1020% | +1.6488% | 两独立服务吞吐等权均值；四对延迟方向均改善，幅度有波动 |
| R6 2048/4096 | 跨重复变化不同 | 中央/尾部延迟存在取舍 | 保留 2048；4096 不统一晋级默认 |

R3 是主要收益；R5 的对照已经包含 R3＋R4，因此是该组合基础上的增量改善。不能将逐轮百分比相乘作为受控累计收益。完整 26 个 canonical 性能服务的 TTFT/TPOT/E2E p50/p90/p99、吞吐、token 数、成功/失败、采样显存、块和 preliminary/失败记录均在评审包中保留，不挑最好结果。

R3–R6 主负载：实际 chat-template/tokenizer 输入 16384、性能输出 1024、C1/C4，正式 `ignore_eos=true`；自然停止正确性另测。固定 service/wire seed20261022、temperature0、top_p1、checkpoint 默认 thinking、前缀缓存关闭。R2 是 512/128，GC/观察条件也不同，不能直接作为后续提速分母。R3/R4 两块不等于独立服务重启；R5/R6 每组每并发各两独立服务。以上不授予统计显著、完整精度、长稳或生产资格。

## 开关、回退和 AOT 缺口

三项优化在源码中默认关闭，实验时显式选择：

- R3：`GLM53_MHC_SINKHORN_FUSED=1`，`GLM53_MHC_SINKHORN_IMPLEMENTATION=tilelang-aot`。
- R4：`GLM53_KPOOL_FORWARD_METADATA_CACHE=1`。
- R5：`GLM53_SPARSE_KV_LENGTH_DEVICE_CACHE=1`。

维持已测 eager/Graph-off/MTP-off、TP16/DP1/EP1 组合，以及原诊断、GC、观察器和状态检查。单次 packed extend≤4096 由真实批次调度预算保证，不能自动截断请求或删除断言。此贡献不部署完整 PD/MemCache，也不覆盖 site-packages、升级依赖或修改权重/宿主机。回退某项开关不等于任意新组合已获原实验资格。

**原已测 AOT SO 不在 Git。** loader 仍严格只接受 SHA256 `8b4ab5f9c4eb37f64b4d9746def519a2247a959119db6e19caae0297af988945`。本包含精确生成的 [AscendC CU](native/r3/wrapped_kernel.cu)、[MIT 通知](native/r3/LICENSE-MIT.txt)及必要构建/导出/探针源码；已有 exporter 消费固定缓存，不是新核编译驱动。不能因此声称一键复建同字节原二进制。不同重建 SO 必须建立新身份并重新完成数值和整模型验证，不绕过现有哈希保护。

## 离线核验与材料访问

`node tools/verify_submission.cjs` 核对公开文件清单；增加 `--source-repo <已有SGLang对象库>` 可在临时 Git index/object 目录重建四阶段，不改变源 checkout、导入 Python、加载 NPU 库或运行模型。精确官方底座对象需已可读取；检查器不会下载、安装依赖或请求凭据。

[构建/探针来源](manifests/reproduction-sources.json)逐项记录原固定 Git 文件及哈希；实际探针须按原相对布局放入另行准备的 candidate 并提交后使用。[合成负载参考](manifests/workload-reference.json)保留原负载哈希和精确长度协议，不复制原 prompt。

原实验私库链接需要单独访问权限。此处是公开代码与精简数值证据，不是原始证据镜像：完整 SSE/server logs、trace、运行环境、native 二进制、权重和控制平台未随包发布。完整原件回放、设备关键路径归因、完整 ABI/codegen 闭包与精度复核仍有明确访问/验证限制，详见评审包与复现说明。本次公开提交不等于在新发布 SHA 上重新实测。

SGLang 源码随附[原许可证](LICENSE-SGLANG)，TileLang 衍生代码保留固定上游 MIT 通知和出处。所有新材料位于本独立子目录，原仓库其他项目和历史结果保持原位。
