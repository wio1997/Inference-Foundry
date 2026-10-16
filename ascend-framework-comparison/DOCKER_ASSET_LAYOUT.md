# Docker资源入口与就绪条件

截至2026-10-17本次快照，15台均已验证Docker27.1.2、Linux ARM64、Ascend默认runtime和16个davinci设备；统一入口已经建立在本地NVMe的`/data/wio`。详细时点及验证边界见[机器可读快照](inventory/20261017/docker-asset-checkpoint.json)。本页是资源准备状态，尚无新Flash/2P2D模型服务或性能验收。

| 用途 | 统一宿主机路径 | 实际状态 |
|---|---|---|
| 原GLM53本地权重 | `/data/wio/models/GLM-5.3-w8a8` | 15台真实符号链接指向原本地NVMe副本，避免额外复制764GB；核对193文件大小及config SHA，未再次完整哈希所有payload |
| Flash本地权重 | `/data/wio/models/GLM-5.3-Flash-w8a8` | 14/15/46/47完整复制，97文件逐文件SHA通过后原子改名；父独立检查文件名/大小、config SHA和本地NVMe，不把它称为第二次全量payload重哈希 |
| 离线镜像 | `/data/wio/images/` | 按角色准备，见下表 |
| 启动配置/源码/日志/实验 | `/data/wio/ascend-framework-comparison/{config,source,logs,runs}` | 全15台目录已建立；代码快照位于`source/snapshot-6efbc385`，它是资产准备前的记录，不等于后来GitHub文档提交或新适配实现 |
| 本轮资源状态 | `/data/wio/ascend-framework-comparison/assets/unify-20261016/STATUS.json` | 每台独立进度、错误和最终身份receipt |

Docker实际镜像层仍存于`/docker_data/data`。没有迁移daemon数据根；统一的`/data/wio/images`指离线归档，不能把它误解为Docker运行层存储目录。权重以只读方式挂入新容器，不能启动指向旧NFS模型入口的实验。

## 固定候选镜像

| 框架 | 固定来源/身份 | 本次状态 |
|---|---|---|
| vLLM-Ascend | `quay.io/ascend/vllm-ascend:v0.30.0rc1-a3-arm64`，imageID `sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83` | 14/15/30/31/43/45已导入；父独立核实实际imageID及linux/arm64。它仍是RC候选，不能称两模型均已稳定认证 |
| SGLang | `swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.1.0-a3-B103`，imageID `sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263` | Mac到源1仍上传；完整SHA通过后才分发导入34/35/36/41/46/47，尚不能称已安装 |

B103是`cann9.1.0-a3-B103`镜像构建标签的一部分，不是SGLang的语义版本号。其Ascend源码来自`release/2026930`的`ce0a8fea5f4245839d174b2b8c55692be1810a7f`；Flash A3 KDA/混合状态等适配仍需真实验证。候选基础镜像到位只允许开始功能适配，不等于Flash支持完成或性能达标。

vLLM新归档：源1及目标`/data/wio/images/vllm-ascend-v0.30.0rc1-a3-arm64.tar`，21,824,162,304bytes，SHA256 `b26c61a4ea40dcd5de17a771523bccca33ff8ca92d5d174cba6ec8ee613456fb`。源1旧`/data/wio/old/vllm-ascend-v0.30.0rc1-a3-arm64.tar`至少一层blob校验不符，已在任何目标导入前阻止；旧文件保留，当前使用源1已固定imageID重新docker save并验证每层的归档。不能重用旧包或放松校验。

B103新归档：`/data/wio/images/sglang-a3-b103-arm64-verified.tar`，17,106,358,784bytes，预期SHA256 `9c6e36ec870dedf51d28e3ce7310c9204bf3a525beff0d92ff78fe4947722b92`。上传中为`.incoming/sglang-a3-b103-arm64-verified.tar.partial`；存在partial不代表成功，需最终归档SHA与每台Docker身份验证。

## Flash统一权重来源

本次重新确认源1（172.16.12.1）的`/data/wio/GLM53/GLM-5.3-Flash-w8a8`：97文件、332,970,562,139bytes，`glm5_next` / `Glm5NextForConditionalGeneration`。83个索引引用safetensors分片完成header/offset/张量索引存在检查，源97文件全部payload SHA完成；目标四台逐文件复制后SHA校验。完整清单见[flash-source-sha256.json](inventory/20261017/flash-source-sha256.json)。

两框架统一使用这一精确快照。它与另一网络166历史记录的332,970,578,967bytes不同，不能称同一份逐字节快照，也不能把166的既往功能测试当当前四台验证。

## 什么时候可以开始

每条轨道可以独立推进，无需等待另一条全部完成。负责该轨道的对话先检查实际状态/进程/锁，只有对应节点镜像ID、架构、本地权重、文件/配置身份和容量通过资源门控，且设备独占归属/端口/共享服务冲突核实后，才能开始启动和功能验证。`ROLE_RESOURCES_READY_NO_MODEL_STARTED`仅代表文件与镜像准备完成。当前15台基础Docker/目录均通过，vLLM六台角色资源完成，SGLang六台镜像仍待上传导入；备用32/33/37未启动推理。

随后按单机已知答案、PD/SDMA、图/MTP/缓存正确性，再做正式性能。原2P2D每角色32die须重算并冻结并行与缓存布局；Flash须完成混合状态适配。原有K8s申请未删除，空闲NPU进程列表不能代替独占授权。所有构建/启动/模型/profile/压测经[本机ZCode DeepSeek执行规则](EXECUTION_CONTRACT.md)，不自动恢复定时任务。
