# M019：Q/K 副流适用性与 Stream/Event 参数合同

固定 SGLang 安装 HEAD `ce0a8fea5f4245839d174b2b8c55692be1810a7f`，8 个新采集源文件匹配 HEAD；安装包为 `torch_npu 2.10.0.post6`。**本轮取得布局/分支和 Python 包装层证据，未验证原生事件、模型数值或性能。M019 运行/性能 INCONCLUSIVE，整个 PD＋MTP 任务仍为 E0。**

## 原 Q/K 候选的门控

原候选针对 DSA prepare 中“副流 Q 投影、主流 KV norm”这对工作，比较并行与串行。它位于特定布局分支，不能由模块中出现 alt_stream 推断实际重叠。

| 步骤 | 固定源码行为 | 边界 |
| --- | --- | --- |
| 模型映射 | GlmMoeDsaForCausalLM 继承 DeepseekV2ForCausalLM | 实际加载类未 introspect |
| 布局 | `is_neox_style = not getattr(config, "rope_interleave", True)` | 观察 raw config 的 rope_interleave=True，赋值得 False；实际 HF override 和 rotary 对象未知 |
| MLAPO | 开关为 True 且不是 extend/draft-extend/mixed 时先走 preprocessing | 进程开关取值未知 |
| neox + alt_stream | 副流 wait_stream(main)，Q 投影后 record_stream/record_event；主流做 KV norm 后 wait_event(q_event) | 原生操作本轮未执行；正确的 q_event 等待未修改 |
| interleaved | 可能调用已有 fused_split_qk_norm，再做 Q 投影 | 与该对 Q/K 副流不同；融合已存在 |
| 副流分配 | CUDA/MUSA 或 NPU/ROCm 指定开关触发 | NPU 多流/MLAPO 的声明默认值 False 不等于实际进程值 |

[布局探针](layout-cpu-result.json)的 16 组分支、3 组布局、16 组分配条件和 2 组 factory 传参 spy 均通过，没有导入 torch。若沿用[观察配置](observed-layout.json)且实际对象保留构造标志，课程引用的 neox Q/K 副流子候选**条件不适用**。不为适配候选改变 rope 布局；其他 MLAPO/indexer/MoE 流仍独立。

## 独立 MLAPO 包装层反例

另一个 `npu_mla_preprocess` 分支在不使用 mlaprolog 且存在 alt_stream 时，副流 preprocessing、主流计算 q_lora，最后执行 `current_stream().wait_event(m.alt_stream)`。它不是上述 q_event 分支。

安装 `streams.py` 的真实 Python 方法：wait_event(event) 调 event.wait(self)；wait_stream(stream) 先执行 stream.record_event()，再交给 wait_event。将这两个原样方法放到 object 基类，提供显式最小接口夹具，得到[三组包装层反例及 Event 正向控制](wrapper-contract-result.json)：

| 调用 | 实际 Python 结果 | 证据范围 |
| --- | --- | --- |
| wait_event(Stream-like，无 wait 方法) | AttributeError，未记录 event | 该夹具不满足包装层参数合同 |
| wait_stream(Stream-like) | alt.record_event → event.wait(main) | 包装层安排生产流记录标记、接收流等待 |
| wait_event(Event-like) | event.wait(main) | 正常 Event 参数控制通过 |

标签不代表设备执行时序。没有真实 Native Stream/Event 实例、tensor 或数值计算，不能写成真实 NPU 已复现错误，也不能认定观察服务已经故障。

## 原生类接口核验：UNAVAILABLE

在精确镜像 `sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263` 的新 CPU 容器中，校验 4 个安装包文件哈希后尝试只读取类描述符，导入因缺少 `libascend_hal.so` 失败。因此 native Stream 的基类方法、动态属性和实际别名仍 UNKNOWN。

[原始结果](native-api-unavailable.json)保留 `probe_status=UNAVAILABLE`。其中 correctness=PASS 表示流程捕获并记录了不可用状态，不能当接口测试通过；Task Run correctness 为 not-applicable。没有添加 driver 挂载重试。

容器 runc、无网络/设备/bind、只读根、cap-drop ALL、1 核/2 GiB/64 PID，已按精确 ID 与任务 label 删除。[容器回执](cpu-container-receipt.json)显示已有服务 ID、启动时间和重启计数前后相同。没有模型请求。不同机器时钟不能混算耗时，本轮没有性能比较。

## 可审阅候选与验证门槛

保存[一行候选补丁](candidate/mlapo_wait_stream.NOT_APPLIED.patch)：只把独立 MLAPO 调用改为 wait_stream(m.alt_stream)，通过 Python 语法解析。**未应用、未接受为已验证修复。**

应用前须确认实际 m.alt_stream 类型、真实 API、分支可达性及事件完成语义，再测 graph/eager、buffer lifetime、取消复用、数值和 accepted length。候选保留副流到主流的依赖，不改原 neox Q/K 的 q_event/record_stream。该独立分支受 MLAPO、非 extend、uses_mlaprolog=False 等条件限制；raw 配置与默认开关不足以证明当前服务会走这里。

## 必要工作、上界与后续

适用 Q/K 分支必须完成 Q 投影、KV norm 和必要依赖。两支会竞争计算与 HBM，并付出 event/切流成本，不能把孤立耗时简单取 max 当实际并行时间。需要在同 NPU、dtype、布局、shape、graph 条件下测串行/并行段长、资源能力及每个有效输出 token 的完整成本。所有设备时间、可达带宽与算力仍 UNKNOWN。

若原布局排除了这对副流，删它的 event 或串行化不会带来该分支的主线收益。后续跟随实际 interleaved/preprocessing 路径，不开启原本不走的分支制造对照。

资源归属仍待确认。独立推进 M001 tokenizer/template 身份及输入夹具，为真实 PD 运行合同补齐输入；Ascend、PD、MTP 的最终约束保留。

## 出处与复现边界

[source-index.json](source-index.json)和 source-inputs 保存固定源码与原文件头。[zcode-audit.json](zcode-audit.json)核验五个精确 session、deepseek-flash 配置别名、冻结命令各一次、哈希与内外退出码；后台 release UNKNOWN。没有发布认证设置或 raw model I/O。

脚本默认依赖本地 Task 布局。布局探针的原始机器盘点未发布，只公开相关模型字段与 raw/config 哈希；原始入口见 [artifact-index.json](artifact-index.json)。CPU 夹具不能充当原生 NPU 回归。
