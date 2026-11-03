# Loop 002：末层缓冲分支的冻结配置证据

冻结启动命令明确包含 `--speculative-config '{"method":"dspark","num_speculative_tokens":5,...}'`，同时设置 DP=2、TP=8、启用 EP，并请求 `enable_flashcomm1=true`、`enable_dsa_cp=true`；见 `evidence/frozen-serve.sh:28`。原始基线记录有 speculative draft 计数，但该计数不直接证明 MTP 缓冲调用。

在冻结镜像的 `vllm/config/speculative.py:1909-1912`，`use_eagle()` 明确包含 `dspark`。结合 `model.py:975-986`，如果进程的 resolved speculative config 仍为 dspark，并且该进程是 PP 最后一组，那么 `needs_mtp_hidden_states=True`，模型初始化路径会分配 `_mtp_hidden_buffer`；接着该模型 forward 的源级路径会对最后一层残差执行 `flatten` 与 `copy_`，与另一路 `hc_collapse` 一同消费残差。源文件的双容器哈希一致记录在 `speculative-source-provenance.json`。

**仍需核实**：启动脚本未显式传 PP size；目前可用快照没有证明 PP 默认值或实际 resolved PP 值，也没有进程的 `get_pp_group().is_last_rank` 对象证据。FlashComm1 被请求并不单独证明 `use_sequence_parallel_moe` 为真：`ascend_config.py:751-806` 仍检查该属性。Ascend patch 的说明指出其重新定义该属性，但具体实现、实际应用及 resolved backend 尚需对应源码或进程证据。故只可将 MTP buffer 存在判定为 **条件推导**，SP 分支为 **UNKNOWN**，不能称运行时已经逐次观测到 copy / gather。

若 SP 且 buffer 存在，源码为 gather 残差和 pre_mix → 复制原残差 → collapse；SP 且 buffer 不存在时为 collapse → gather 输出；若 SP 不存在则不产生这两条 gather。存在缓冲时，融合分析必须同时保留复制缓冲的正确值和最终 collapse 消费；运行时具体 caller、buffer 生命周期、图优化与别名仍未知。

结构化证据、输入 SHA 和判定域在 `frozen-configuration-evidence.json`。只做 CPU、源码与已有结果分析，没有新设备负载。

### 追加：并行配置源码（Loop-002）

冻结 `parallel.py:122` 的 `ParallelConfig` 默认 PP=1；启动脚本未显式传 PP，故**在 CLI、环境、配置和 patch 没有覆盖默认值的条件下**，单个 PP 组中的每个 TP/DP rank 都是该组最后一组，结合 DSpark 谓词可推导 MTP 缓冲被分配。并未直接读取基线进程的 resolved 对象，不能把“默认值”写成实测。

`parallel.py:197,715-729` 的原始默认 backend 为 `allgather_reducescatter`；源属性在该 backend、EP 开启、TP>1、DP>1 时为 true。冻结命令所列 EP、TP8、DP2 满足后三项，并请求 FlashComm，但属性可被 Ascend patch 改写、backend 等可能被运行时覆盖，尚未获得当时进程中有效属性的证据。故 SP=true 仍是写明前提的配置推导。双容器源码 SHA 见 `parallel-source-provenance.json`；候选融合仍不能据此宣称运行 trace 已验证。

### 追加：Ascend 并行 patch

补充的 `patch/platform/patch_parallel_config.py:4-28` 将 SP 谓词改成支持 EP、TP>1、且 backend 在允许集合时返回 true，不再要求 DP>1，并增加 `deepep_v2`。在本次声明的 EP、TP8、DP2 和默认 `allgather_reducescatter` backend 条件下，上游与 patch 两套谓词**都会返回 true**；因此是否打上该特定 patch，不改变这些条件下的逻辑结果。该 patch 文件本身不设 PP size。源文件在两个容器的哈希相同，见 `parallel-patch-source-provenance.json`。

但这仍不能排除当时 backend、配置或属性被其他路径改写；现有快照中的 `patch/__init__.py` 只以注释描述加载方式，不能单独证明原进程实际执行了该 patch。进程 resolved backend/SP 值、原进程调用和当前样本的状态仍 UNKNOWN。
