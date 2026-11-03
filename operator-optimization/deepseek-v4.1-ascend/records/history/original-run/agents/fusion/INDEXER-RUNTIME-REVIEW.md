# Loop-009：Indexer 运行分支和 copy_ 的现有证据

## 实际采样能说明什么

ZCode `operator-ceiling169-v2-092-indexer-events` 退出码 0、verified `deepseek-flash`，只对 rank EP8 的既有 profiler SQLite 做只读查询。数据库 SHA、查询 SHA、完整聚合和局限在 `indexer-host-event-aggregate.json`，查询与远端 wrapper 在本目录的 `query_indexer_host_events.py`、`run_indexer_host_events.py`。这个数据库里：

- PYTORCH_API 记录 `_C_ascend::npu_quant_lightning_indexer_v3` 24 条，`aclnnVllmAscendQuantLightningIndexerV2` 24 条，设备 TASK 类型 `VllmAscendQuantLightningIndexerV2` 80 条。**它们是全库各自计数，不能因次数相同或名字接近就认定逐条一一对应**。冻结 `indexer.py:275` 非 packed 正常路径确有该 `_C_ascend` V3 调用，因而采样与此路径一致；但只凭全局名称不能确认是哪一个模块、哪次调用、来源张量或前述自复制路径。
- PYTORCH_API 有全库 `aten::copy_` 6029 条，但未按堆栈、张量地址或图节点与目标语句匹配。它们包括大量其他对象的 copy，绝不能把计数或耗时归到 Indexer。按照本次 SQL 的两个路径字符串模式，未匹配到 Indexer 或 dsa_v41 的 Python 函数名；这只表明这些模式无匹配，不证明函数未运行。
- `captured-layer-attribution/operator_metrics.csv` 的 `INTERIOR_RUNS_1_2_3` 中，模型配置的八个 index source 层 2、8、14、20、24、28、32、36 均归属 48 次 `VllmAscendQuantLightningIndexerV2`。这是 16 rank × 3 个已有窗口的角色/时间约束归属，不能代替 tensor identity 和 callchain，亦不能断言完整负载的所有轮次。

这里不是一项新 NPU 测试。host API 事件与设备任务之间、捕获时和图重放时之间的对应尚未逐一证明；图内部是否存在、短路或重排 copy_ 以及其 HBM 流量、耗时、tensor version、事件与 capture side effect 均保持 UNKNOWN。

## 设备 profile 限定与待核验

ZCode `operator-ceiling169-v2-090-device-profile` 退出码 0、verified `deepseek-flash`，`indexer-device-profile-provenance.json` 验证 baseline 和 stage 容器镜像一致、三份源码的目标路径未被挂载覆盖且内容一致。

冻结 `device/hardware.py:25-46` 把 build SoC 字符串 `Ascend910_9381`、`Ascend910_9382`、`910c` 映射 A3，`device/device_config.py:26-47` 先使用包 `_build_info.__device_type__`，缺失才根据 `__soc_version__` 推导；`hardware_profile.py:286-307,399-405` 表明 A2 和 A3 都选择 STANDARD，`device_op.py:50-54,2032-2041` 表明 STANDARD 的 `get_dsv41_packed_cache_ops()` 返回 None。`model.py:711` 与 `indexer.py:50` 在构造对象时各自使用该函数。**因此若对应运行时加载的 _build_info 判为 A2/A3 且对象未被替换，则两个对象进入非 packed 配置**；这仍不证明每次 index source、非空缓存及 copy_ 条件命中。已有平台 ini 或机器型号信息不能直接代替 Python 进程所加载的 `_build_info`。

最小缺口是只读取证基线镜像中 vllm_ascend 实际加载位置的 `_build_info.py`、核对两个容器哈希/挂载，再检索历史启动日志是否记录安装路径、构建身份或 resolved profile。即便同一文件存在，也不能仅此证明运行过程中没有替换对象或某次调用实际命中；精确的 tensor 别名和 copy 代价仍须相应 callchain/存储 identity 数据，若现有 trace 没有就保持 UNKNOWN，不增加无目的测试。

无推理代码、Task 状态、服务或设备变更。

## 补充：两容器冻结 build 信息为 A3

ZCode `operator-ceiling169-v2-093-build-info` 退出码 0、verified `deepseek-flash`。`indexer-build-info-provenance.json` 证实两个容器在相同镜像、目标路径无覆盖挂载的条件下 `/vllm-workspace/vllm-ascend/vllm_ascend/_build_info.py` 内容一致，SHA-256 为 `65edd0c22127fac498174576a99758e422052ed0180ed332570c19cddab0197a`，内容 `__device_type__ = 'A3'`。主控可复核对应源码和当前实际包加载路径。按上面的冻结代码正常导入及不替换对象这两个前提，A3 → STANDARD → `get_dsv41_packed_cache_ops() is None`，两个 Indexer/attention 采用 non-packed 配置的证据链已从假设缩为当前镜像配置推导。历史进程当时实际加载路径、补丁及每次 source/非空分支仍未直接核实；用这个推导无法证明某一个 `copy_` 产生设备 task 或能安全删除。
