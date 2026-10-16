# 权重、镜像与交接材料就绪检查（2026-10-16）

**当前只有部分材料满足要求，不能标为两条新项目线全部就绪。** 原GLM53权重在15台可访问宿主机均找到；新Flash四个目标节点的已检查目录没有发现Flash权重；新版镜像尚未部署到新组。当前SGLang六节点环境保留，其他机器有资源占用和角色冲突。

## GitHub保存了哪些内容

| 内容 | Git内情况 | 入口 |
|---|---|---|
| 项目目标、分工、当前阻塞和接手动作 | 已保存；本轮补现场快照 | [服务器表](SERVER_ALLOCATION.md)、[HANDOFF](HANDOFF.md)、[资源JSON](RESOURCES.json) |
| 两条实验线、SGLang适配与优化步骤 | 已保存，状态区分计划/已验证/未部署 | [PLAN](PLAN.md)、[Flash](tracks/flash-adaptation/README.md)、[原模型](tracks/glm53-4p2d/README.md) |
| 独立前缀、四轮、80K/600计量口径 | 已保存 | [BENCHMARK_CONTRACT](BENCHMARK_CONTRACT.md) |
| Flash多模态测试方案 | 已保存，尚未执行 | [MULTIMODAL_TEST_PLAN](MULTIMODAL_TEST_PLAN.md) |
| 旧六节点实际源码/配置/镜像身份/SDK映射SHA | 已保存127个归档文件；历史快照不可当当前进程证据 | [current](archive/20261016/current/OPERATIONS.md) |
| 已做修复、有限A/B与未上线候选 | 已保存源码、证据、状态边界 | [优化账本](archive/20261016/OPTIMIZATION_LEDGER.md) |
| 历史权重源清单、分片大小和payload SHA | 已保存；不是本次重哈希全部权重 | [weights](archive/20261016/weights/) |
| 历史测试摘要、原始数据位置与SHA | 已保存；大raw包在外部 | [results](archive/20261016/results/)、[ASSETS](archive/20261016/ASSETS.json) |
| 镜像/权重/编译SDK二进制、全部raw | 不入Git，位置与版本/校验值入库 | 下表及ASSETS |
| 凭证、私钥、ZCode provider配置 | 不入Git | 现场私下提供 |

因此GitHub是源码、配置、方案和证据的完整交接索引，**不是含TB级权重和所有二进制的即用部署包**。未完成项逐项保留，不把计划写成已验证。

## 资产与要求

| 资产 | 固定身份与所在位置 | 当前结论 |
|---|---|---|
| 原GLM-5.3 W8A8 | `172.16.12.1:/data/wio/GLM53/GLM-5.3-w8a8/`；193文件、764,240,829,290 bytes；`GlmMoeDsaForCausalLM` | 15台可访问宿主机均有本地NVMe副本，文件名/大小匹配源清单；14台完成182个索引分片头/范围/索引张量存在检查；45仅做清单/小文件核对。42未核实，38已移出 |
| Flash W8A8 | 另一套网络记录 `172.16.10.166:/data/tiankuan/wio/GLM-5.3-Flash-w8a8`；97文件、332,970,578,967 bytes | 来自166/170文本1P1D功能验收的本地历史记录，不是新14/15/46/47就绪证据；本次166 SSH认证失败，没有重新确认远端或下载权重 |
| 当前六节点SGLang日更 | `sglang-ascend:nightly-cann9.1.0-a3-20261014-27b68303`；imageID `88d9988736fd0e0ba51b5f5e5b4b0274b97e57d0e3500a0daf4bcb242cde68d0` | 33/34/35/36/37/41实际容器与imageID一致；包元数据0.5.22.dev919+g27b68303d；保留运行栈和回退 |
| SGLang B103候选 | `swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.1.0-a3-B103`；ARM64 manifest `a99166c124ffe3c8750855fbeb998647aec1fb8757aace87e9cabffb592b2357` | Mac已完整下载和层校验；15台检查的Docker镜像列表均未见B103；未上传/导入/切换。Flash KDA专用Ascend路径仍有适配缺口，不是已就绪Flash环境 |
| vLLM Flash参考镜像 | `quay.io/ascend/vllm-ascend:v0.30.0rc1-a3`；digest `sha256:218626e4a0ec511c13b0bf70b3de3cb1d9f6898c4f3dee5b5d25a34a464d5892`；imageID `090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83` | 166/170等另一环境的历史记录已有；本组15台镜像列表均未见此ID。14/15已有的是0.27.1等旧镜像，不能当Flash固定参考已装好 |
| 原模型vLLM 4P2D镜像 | 尚未给新SDMA组冻结资格通过的完整镜像/补丁组合 | 不能直接继承旧RDMA基线或以Flash文本单次通过代替；先固定权重、框架提交、SDMA/PD、量化、图/MTP/缓存兼容 |
| 原生MemCache/MF | 当前六节点使用私有配套1.3与MF LOCAL；实际映射SHA在旧快照 | 本次只读包元数据不等于实际映射库复核；vLLM参考镜像自带MF/MC1.2.0，不能假设已经带上现有1.3修复 |
| AISBench客户端 | 43既有`sglang-aisbench-43-20261014`保留 | 若43转D需另选独立CPU客户端；未搬迁或重跑 |

Flash参考镜像来源和边界见 [flash-source-record.json](inventory/20261016/flash-source-record.json)。没有把该RC镜像称为全项目稳定认证；当前六机SGLang镜像和B103是不同候选，比较阶段必须固定一个明确环境，不能混跑后计算框架收益。

Mac B103离线文件：`/Users/wio/work/sglang-glm53-33-41-deploy/stable-migration-b103-20261016/sglang-a3-b103-arm64-verified.tar`，17,106,358,784 bytes；已验收归档SHA256 `9c6e36ec870dedf51d28e3ce7310c9204bf3a525beff0d92ff78fe4947722b92`。本轮只核对文件存在/大小与已有21层验收receipt，没有重新解包或重读17GB。旧镜像现场归档位置见ASSETS，不要调用依赖已清理Mac旧tar的传输脚本。

本地权重路径：14/15/30/31/32用`/data/wio/GLM53/GLM-5.3-w8a8`；33/34/35/36/37/41用`/data/wio/glm53-w8a8-tuning-local`；43/45/46/47用`/data/wio/glm53-w8a8-local`。这些路径实际落本地`/dev/nvme1n1` ext4。另一个`/data/wio/sglang-glm53-pd-20261013/model`在43/45/46/47是NFS源，不能把它当本地副本启动新实验。新Flash的模型架构/processor/视觉权重必须单独审计，原764GB文本模型不满足Flash要求。

新Flash目标已查到本地盘余量约2.0–2.8TB，容量能够容纳约333GB权重及镜像；这不表示已经传好。全节点精确磁盘余量、大小/小文件SHA、索引检查、镜像ID和采样时间见readiness.json。新结果只在同合同、同权重精度和已证明的SDMA链路后产出；没有新性能数据。

## 尚需解决的项目条件

1. 确定Flash SGLang D节点：47已被用户指定替代38作为原模型vLLM P；45未列新计划但有旧作业，不自行抢占。
2. 核对14/30/31/32/42/45的资源申请归属，以及43/45/46/47旧推理作业的去向；不因瞬时AICore0便停用。
3. 获取42宿主机可用的只读访问/资产证据；此前不把其权重、镜像或NPU健康写成已验证。
4. 取到Flash实际manifest并在新节点完成本地权重校验；离线准备固定vLLM参考镜像及SGLang候选、必要配套依赖。当前Mac只保留B103候选，避免重复占盘。
5. 经本机ZCode DeepSeek部署后，先验证文本/真实PD/SDMA/已知答案；多模态、图/MTP及MemCache按方案逐项准入。只读检查不能替代这些资格测试。

本轮0模型请求、0容器/服务变更、0部署/性能执行、未改任何自动化。原始全量只读输出留在Mac审计目录；公开JSON只保留必要白名单字段，不包含环境凭证或业务请求文本。
