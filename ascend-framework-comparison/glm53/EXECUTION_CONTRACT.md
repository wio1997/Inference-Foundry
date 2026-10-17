# 原 GLM-5.3：本机ZCode DeepSeek执行规则

本文件只约束project_id=`glm53`；本项目自己的GPT6.1 Sol High对话独立决策和验收，没有另一个模型的前置任务、统一排队主控或共同基线合同。

主控可做只读核查、源码/文档研究、本地代码草拟和审查。实际资源准备、离线构建、镜像导入、容器/服务创建启停、部署、SDK/框架overlay、NPU/模型请求、性能/profile采集均交用户Mac本机已有DeepSeek配置的ZCode执行。保留配置，不读取/输出provider或凭证。

入口`/Users/wio/work/sglang-glm53-33-41-deploy/zcode.sh`。每次显式指定本项目独立执行目录`--cwd`，提示包含project_id、唯一目标和本次已冻结SHA的精确launcher；不能省略cwd而把另一项目记忆或默认公共任务目录当当前工作。

1. 读取本项目真实状态/receipt、进程身份和锁；无重复任务才继续。先确认模型身份/路径与本项目资源白名单。
2. 审查具体脚本/参数/目标/边界/回退，冻结本次SHA。服务变更前准备可恢复配置；历史一次性launcher不可重放。
3. 本机ZCode只执行精确launcher一次；后台正常长任务保留，按有意义检查点或用户问进度时读取，不连续盯、不建定时。
4. 主控独立核对实际imageID、库/源码、模型/缓存/PD正确性以及原始失败；正确性通过后才出本项目性能基线。局部微测不冒充端到端收益。
5. 只更新本项目STATE/HANDOFF/run与结果；独立合同版本不依赖另一模型。实际物理争用如实记录、选择合适窗口，不操作对方服务，不发送跨对话消息。

运行根为`/data/wio/inference-foundry/glm53`，配置/脚本/数据/日志/locks/Meta/provider分别使用本项目名称。该根已在本项目八个白名单节点创建；当前实际实例、阶段与证据见`STATUS_BRIEF.json`和`STATE.json`。Docker层仍在`/docker_data/data`，权重从本项目RESOURCES的本地目录只读挂入。

P→D数据面强制使用**SDMA**，同时适用于本项目vLLM-Ascend与SGLang；不允许自动回退RDMA传输模型状态。按本目录[SDMA合同](PD_TRANSPORT.md)核配置、实际加载路径和真实请求传输证据；TCP控制/握手与PD数据面分开。此项为用户指定要求，尚不是运行验证结论。

## 2026-10-17 用户补充：MemCache 适配先查 Ascend 部署文档

进一步修改 MemCache 适配前，先核对应框架/模型/SDK版本的 Ascend 官方部署指导与配置说明，保存实际来源及适用边界。当前查证索引为 `runs/20261017T1340-official-memcache-docs-review/index.json`，结论见同目录 REVIEW。优先核原生租约TTL、A3可建池内存扫描、独立预分配池与客户端0GB模式等受支持机制，再决定框架补丁必要性。文档中的系统调优示例不覆盖用户现有禁止系统SDK/驱动/BIOS/时钟及其他系统变更的边界；不重写已冻结运行文件。


### 当前部署配置的适用边界

- 对照 [vLLM-Ascend v0.30.0rc1 的 Step 6](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0rc1/docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md#memcache-and-vllm-separated-deployment)：Meta → 独立贡献进程建池成功 → 模型进程；模型客户端 `dram.size=0GB`，两类进程的连接/协议一致，`max.dram.size` 覆盖贡献量。此处“分离”指进程，不要求把同一 NPU 拆到两个非特权容器。
- SDK Python API 使用已保存的官方 1.3.0 提交 `06495e8a381c65b7f3b2615687e6b93e5672e242`，见 `runs/20261017T1458-official-memcache-api-review/index.json`。`init_bm=False` 仅作元数据查询；真实 payload 读回使用 `init_bm=True`、`dram_size=0GB`。元数据存在不能充当读回证据。
- 当前 vLLM 原生配置是 `config/vllm-correctness-v10/local.conf`：Meta `30:29680`、ConfigStore `30:29681`、common max `192GB`；SGLang 使用另一组端口和 `160GB`。检查工具须直接复用对应原生配置并校验 SHA，禁止照抄另一框架的端口。1774 检查助手的错误端口导致初始化失败，实际原因保存在 `results/vllm-shared-long-attempt-v1.json`，没有据此修改 SDK。
- 当前外部池两框架均为 568GiB，实际分配和真实读写仍分别验收；文档示例容量不等于现场可分配容量。官方只读内存扫描、原生分配与 Meta mount 证据分开记录。系统调优示例不改变本项目禁止系统变更的边界。

### 文档配置与项目补丁的边界（当前 1954 部署）

| 项目 | 当前实现及依据 |
| --- | --- |
| 建池与客户端接入 | 官方 Meta → 独立贡献进程 → 模型客户端顺序；项目 provider/supervisor 只封装已固定的 SDK1.3 setup/init/close 调用。模型客户端 dram=0GB/common max 覆盖贡献量。 |
| 容量与缓存读回 | 外部池 568GiB 用现场容量证明验收；P 的 buffer_only staging 另计。键存在和真实 payload 读回分开验证。 |
| 强制 PD SDMA | transfer_engine/conn 项目 overlay 用于用户要求的传输约束及类型/返回码审计，不冒充官方 MemCache 部署要求。 |
| DeepEP L04 优化 | 独立的 D 路由元数据批量下载优化，不改 MemCache 池配置或存储后端。 |

当前 staged overlay 清单没有 MemCache 存储后端替换文件；这不等于本次重新逐字节核验了镜像内所有 SDK/后端文件，实际兼容性仍依据既有源码/运行身份、真实缓存读回与模型 SDMA 验收。vLLM-Ascend 的官方模式可指导进程部署，不能单独充当 GLM-5.3 SGLang 支持证明。完整来源和适用边界：`results/memcache-docs-current-deployment-reconciliation-v1.json`。

### C32 恢复实例的对应关系（2029）

受控恢复只改变运行身份：26 个 staged 文件中，22 个逐字节相同，4 个 supervisor 文件仅 run_id 不同。2048 当前启动和 2050 只读计数快照分别独立通过；P 实验缓存补丁未启用。先前程序语义资格可按来源复用，当前 C32 前缀的真实读回与模型 SDMA 仍分别验收。新来源核对见 `results/memcache-docs-C32-L04-restore-reconciliation-v2.json`，不改写历史 1954 证据。官方另列的灵衢网络版本与 SDK Python 包版本分开记录，本次文档核对不新增系统版本认证或系统变更。
