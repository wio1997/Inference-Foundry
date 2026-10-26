# 现有日更环境操作与恢复记录

本记录针对切换候选 B103 之前的实际环境。保存记录和下载新镜像不修改正在运行的服务。

## 基础镜像

- 来源：`quay.io/ascend/sglang`，2026-10-14 A3/CANN9.1.0 日更。
- 引擎来源提交：`27b68303d043ddbddf9861c55c26bc8617ea7c35`。
- 六台实际 Docker 镜像 ID：`sha256:88d9988736fd0e0ba51b5f5e5b4b0274b97e57d0e3500a0daf4bcb242cde68d0`。
- 原镜像归档 SHA256：`1ca08879d0e65d2b4c5c28c7325583a45eb066666139aad01508ef4168019d38`。本机副本按用户要求清理；服务器中的镜像和容器保留。

## 部署与实际运行改动

- P：34、33、35、37；D：36、41。权重为各节点本地只读目录挂载 `/model`。
- P：TP64、attention DP4、EP64、内存比例0.75、MTP1/draft2、每DP chunk4096。
- D：TP32、attention DP16、EP32、内存比例0.90、MTP5/draft6、FULL decode graph；原生 D radix 前缀复用开启。
- P HiCache：buffer_only、wait_complete、kernel_ascend、page_first_kv_split、每rank主L2配置32GiB，原生 npu_memcache。
- 8个实际80GiB provider，共640GiB；独立 Meta 服务34:25340。私有SDK目录和provider保持在服务器。
- 私有配套 MF/MC 1.3源码构建覆盖层：`/data/wio/sglang-glm53-33-41-20261015/warm93-memcache-aisbench-20261015/sdk-official-13-a085-0649-v2/probe/overlay`。用于修复pinned HOST误判，不使用Python额外CPU拷贝，也没有LD_PRELOAD。
- P MLA兼容补丁（官方PR43031/41028移植）：`ccd5ab4f85f16bd8c8c9ac88df7c152612f34e9e8bc7c83b9d0ebfd74915cffc`。D MLA保持各自原文件，以逐节点快照为准。
- P预取生命周期修复（官方PR43108/44698方法移植）：UnifiedRadixCache SHA `b58a067b25f291df9650d418e40dedcf2427da8f3cb52d1df9df9e93e0d58a65`。
- 原生MemCache backend SHA `875ae02e8cad22d89c05fe44d75776373849e198abd7d133421f8cf728bda6ba`。
- 正式网关为官方SMG1.13 ARM64预编译包，36:31480，native SHA `f2d6e3bfdb665037a4fa144e1ae8835051343bd3469d00ba668465c47fd0507e`。旧0.3.2回退包仍保留。
- 当前规范启动脚本位于服务器任务根目录的 `scripts/warm-rank-wrapper.sh`、`warm-launch-p.sh`、`warm-launch-d.sh`、`warm-env.sh`、`warm-router.sh`。快照保存实际文件及SHA，不使用旧上午交接包中的过时网关脚本。

## 测试与未解决事项

最新C64测试固定64个独立74400-token前缀预热，4轮共256个不同80000输入/600输出请求；预热不计吞吐，正式保留自然缓存。

最终252条成功、4条超时；成功请求中3条缓存明细缺失，其余249条确认74368 cached/storage tokens。原始256条、DB和采样已导出，归档SHA `4ba0f9ea7c391d528cf472102b334a8a70a835fd8c3e37ec39a0201b094a78fb`。此轮不作为合格性能基线。

超时后队列恢复空闲，原进程身份保持。因此，先前不同rank线程栈中的不同collective位置只能作为线索，不能据此认定永久TP死锁。P L2在阻塞期间接近占满，已发现预取在D bootstrap之前占用L2的资源循环风险。

`warm93-memcache-aisbench-20261015/optimizations/c64_liveness_fix_v1` 中的bootstrap后预取候选仅通过实际方法体的CPU容量fixture，**尚未部署到任何模型容器**。其结果和候选源码另存，不能与已上线补丁混称。

## 恢复依据

- `records/node-N/sources`：六节点实际引擎关键源文件；`host_files`：实际启动脚本和MemCache配置。
- `records/node-N/runtime.json`：容器/镜像ID、挂载、96个scheduler身份、安装包版本、实际映射SDK二进制路径与SHA。
- 原镜像在六个服务器Docker中保留；原模型容器、权重、官方SDK私有目录和provider没有删除。当前仍可使用36:31480。
- 若以后重建，使用保留的基础镜像ID、该节点源文件和启动配置、原SDK覆盖目录；不得只使用基础镜像而省略这些运行改动。
- 本包保存源码/配置/恢复依据，未复制模型权重或运行中HBM内容。

## 新镜像选择

候选为昇腾官方已验证发布线 `cann9.1.0-a3-B103`，具体来源、ARM64 manifest/config摘要和组件配套见同目录 `IMAGE_PIN.json`、`OFFICIAL_B103_DEPENDENCIES.json`。固定digest下载，不能因标签名认定本机完整GLM5.3/4P2D/MTP/MemCache组合已经通过验证。

所有定时任务保持暂停。新镜像未通过兼容及实际服务验证前，保留当前环境。部署和测试通过用户指定的本地ZCode DeepSeek执行。
