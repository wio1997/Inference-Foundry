# 36点安装路径与支持性审计

Static installed-tree anchors only. No task PD runtime, numerical or performance support is inferred.

观察安装提交：`ce0a8fea5f4245839d174b2b8c55692be1810a7f`。本任务运行路径全部仍待实例化与计数；ENTRY_LOCATED不是PATH_CONFIRMED。

| ID | 机制 | 安装路径入口 | 当前边界 |
| --- | --- | --- | --- |
| M001 | 聊天模板、分词、reasoning parser | `python/sglang/srt/managers/tokenizer_manager.py:974` | 前端CPU入口存在；模板与tokenizer及动态批处理配置待本任务运行合同确认。 |
| M002 | 进程分工、IPC 与流式背压 | `python/sglang/srt/managers/detokenizer_manager.py:173` | 反分词/IPC入口存在；背压和网络消费者需服务时间线。 |
| M003 | CPU affinity / NUMA / 线程数 | `python/sglang/srt/utils/numa_utils.py:169` | NUMA工具存在；NPU设备映射、绑核和线程设置尚未冻结。 |
| M004 | Continuous batching 与请求状态机 | `python/sglang/srt/disaggregation/decode.py:2869` | 专用D组批循环存在；实际调用次数未测。 |
| M005 | 多资源准入、输出预留与抢占重算 | `python/sglang/srt/managers/schedule_policy.py:796` | 预算与抢占入口存在；P/D资源和推测预留分别核验。 |
| M006 | Chunked Prefill 的游标与预算 | `python/sglang/srt/disaggregation/prefill.py:665` | P续块与发送入口存在；chunk与传输粒度不能混为一个开关。 |
| M007 | Mixed batch 与 ragged / packed 输入 | `python/sglang/srt/managers/scheduler.py:4122` | 专用PD分派到独立P/D循环；P/D文件无mix_with_running或is_mixed_chunk引用。仅混批候选在专用PD下不适用，packed/ragged输入单独保留。 |
| M008 | Cache-aware scheduling：LPM / DFS-weight | `python/sglang/srt/managers/schedule_policy.py:237` | 缓存感知策略存在；实际policy、缓存类型与D角色收益未知。 |
| M009 | Paged KV、请求映射与预分配池 | `python/sglang/srt/mem_cache/memory_pool.py:18` | 页映射与MLA池存在；实际Ascend池/dtype及复制或分片方式待确认。 |
| M010 | Radix 树 vs 前缀关联哈希块 | `python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:788` | Unified树匹配入口存在；是否实际选择Unified而非旧Radix待确认。 |
| M011 | 引用保护、释放、淘汰与 Copy-on-Write | `python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:670` | 释放/淘汰入口存在；索引clone不等于KV载荷Copy-on-Write。 |
| M012 | Unified cache 组件 vs Hybrid KV 分组 | `python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:493` | 组件路径存在；模型状态类型、cache实现和组别待确认。 |
| M013 | HiCache：HBM / 主机 / 存储分层 | `python/sglang/srt/managers/cache_controller.py:56` | 分层控制器存在；当前主线是否启用及NPU后端支持需独立探针。 |
| M014 | 预取、写回与 layer/page 布局 | `python/sglang/srt/managers/cache_controller.py:362` | 预取/回写入口存在；布局、event和后端能力不能由接口存在推断。 |
| M015 | CPU / 设备 overlap 调度 | `python/sglang/srt/disaggregation/decode.py:2892` | D overlap循环存在；MTP保留，按真实batch回退与时间线诊断。 |
| M016 | Future token / placeholder 与依赖解析 | `python/sglang/srt/managers/overlap_utils.py:29` | 真实新长度和bonus等字段仍需发布；普通decode删长度副本不适用于MTP。 |
| M017 | 持久状态、增量更新与 GPU metadata | `python/sglang/srt/managers/overlap_utils.py:26` | CPU长度镜像已有按backend门控；缺失声明保守保留，TBO/ngram强制需要。 |
| M018 | Pinned buffer、异步 H2D 与写读竞态 | `python/sglang/srt/managers/overlap_utils.py:87` | 延后H2D入口存在；私有D2H流分配明确CUDA-only，不能直接当Ascend实现。 |
| M019 | 多 stream、event 与资源竞争 | `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py:432` | NPU Q投影副流/事件分支存在；模型和实际shape命中未知。 |
| M020 | TBO / SBO vs vLLM DBO | `python/sglang/srt/batch_overlap/two_batch_overlap.py:363` | TBO入口存在；Ascend实际通信后端和spec组合待探针，不能套CUDA文档。 |
| M021 | 图捕获、形状桶与 replay | `python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py:16` | NPU graph runner存在；draft/target图覆盖及桶填充需运行计数。 |
| M022 | 算子融合、编译与中间张量复用 | `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py:5` | 现有融合与opt-out门控存在；实际模型映射未闭合，不能强开融合或宣称数值等价。 |
| M023 | 权重 / 激活 / KV 量化与尺度 | `python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_w8a8_int8.py:21` | W8A8方案文件存在；目录名C8不证明全部KV是INT8，量化描述与池需交叉确认。 |
| M024 | Attention：MLA、布局、融合与历史读写 | `python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py:21` | NPU MLA预处理文件存在；调用者、KV布局及实际shape待记录。 |
| M025 | DSA indexer / Top-K / HiSparse | `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py:66` | DSA短序列跳选属于已有能力；HiSparse shared-index prefetch安装文档明确要求no speculative decoding，仅该子候选排除。 |
| M026 | MoE 路由、重排与 grouped GEMM | `python/sglang/srt/hardware_backend/npu/moe/fuseep.py:29` | NPU FuseEP文件存在；路由/重排/计算分段与实际专家分布未测。 |
| M027 | 后端选择、形状特化与冷启动 | `python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py:165` | 后端和图路径存在；选择合法性、冷启动和形状分派需安装配置确认。 |
| M028 | DeepEP dispatch / combine 的模式选择 | `python/sglang/srt/layers/moe/token_dispatcher/deepep.py:14` | DeepEP源码存在不表示Ascend主线使用；应与NPU FuseEP分派分开核验。 |
| M029 | TP / EP / DP / PP / CP 的切分边界 | `python/sglang/srt/disaggregation/prefill.py:313` | PD并行分组相关代码存在；TP/EP/DP/PP/CP和跨节点拓扑仍UNKNOWN。 |
| M030 | 专家负载均衡、冗余与最慢 rank | `python/sglang/srt/eplb/eplb_manager.py:42` | 重平衡入口存在；实际后端支持、冗余与最慢rank数据未测。 |
| M031 | MTP / EAGLE：草稿、树验证与接受 | `python/sglang/srt/speculative/eagle_worker_v2.py:9` | V2推测worker和NPU graph导入存在；具体MTP参数、接受长度分布及总成本仍未知。 |
| M032 | 采样、grammar mask 与结构化输出 | `python/sglang/srt/speculative/spec_info.py:165` | 实际算法能力判定比decode旧注释更细；EAGLE族等可声明支持，host-draft类按batch同步。CPU谓词不证明Ascend FSM/collective正确。 |
| M033 | 增量反分词、SSE 合并与结束回收 | `python/sglang/srt/managers/detokenizer_manager.py:75` | 增量反分词存在；SSE合并和结束回收需端点/取消测试。 |
| M034 | Cache-aware routing 与副本亲和 | `docs/docs/advanced_features/sgl_model_gateway.mdx:72` | 仅取得固定安装文档；实际gateway安装版本和副本数未知。单副本不做路由A/B，副本饱和不排除路由价值。 |
| M035 | PD 分离、KV connector 与异步传输 | `python/sglang/srt/disaggregation/ascend/conn.py:95` | Ascend连接器和DCP分支存在；复用PD-CHUNK索引，实际backend与发送时间线待证。 |
| M036 | 会话 / 混合模型 / LoRA / 多模态扩展 | `docs/docs/advanced_features/lora.mdx:2` | 仅取得固定文档；会话/LoRA/多模态逐子功能核验，不默认开启非文本工作量。 |
