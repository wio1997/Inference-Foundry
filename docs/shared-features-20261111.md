# 原生共同优化接入与公平对照状态

此前约1.7倍来自DeepInfer开启KDA捕获而参考保留KDA eager边界，不能作为双方同优化的框架优势。旧数据保留，原验收结论不适用于新的公平对照目标。

## 架构及源码

DeepInfer继续拥有原生LLMEngine、C++ scheduler、Worker与请求回调。固定SDK仅负责模型计算、物理缓存与设备采样；没有创建vLLM引擎或请求调度器。

- `sdk_schedule.py`：请求hasher、真实前缀命中、延迟发布；现有原生分块检查点保存中间递推状态。
- `sdk_async_schedule.py`：按固定SDK的计数语义管理最多两个在途批次、输出占位和缓存释放。
- `sdk_async_dispatch.py`：下一步计算先提交，随后回收上一输出；不猜测设备上尚未回到主机的token。
- `kda_graph_capture.py`：使用SDK稳定索引缓冲区修复align模式捕获；参考与原生使用同一文件，不能算独有收益。

已通过探测的源码提交：`5ba306cfd29839bb6f80b996ecb51ab01b2f1e8a`。
相关源码位于[模型接入目录](../DeepInfer/deepinfer/model_executor/models/glm5_next/full_model/)。

## 已实测范围

相同8die/TP8/DP1/PP1/EPoff、W8A8/BF16。更强参考开启async、prefix/align、PIECEWISE/breakable与同一KDA捕获，每rank缓存4GiB：C4为123.78067421 token/s，C8为196.88095174 token/s。每档32请求、1024入/256出、2次排除的预热；64条完整输出与旧参考一致。原始SQLite计时独立重算通过。

原生集成探测完成6请求，前缀实际命中640 token；28次设备token反馈，23次观察到回收上一输出时下一步NPU仍未完成。43个提交批次全部回收，8rank exit0，固定输出通过。EOS感知请求当步回收；声明的greedy/ignoreEOS负载实际使用两步重叠。这是共同能力接入验证，不是性能收益验收。

复用同一镜像及权重，原生CLI新增`--prefix-caching --async-scheduling`，与已有`--graph-decode --breakable-graph --kda-decode-capture`一起显式开启。独立重建仍需合法权重、固定SDK环境和DeepInfer本地扩展，未声称干净环境复现已通过。

## 波动及当前边界

参考每档仅一批，不据此声明稳定框架收益；旧原生g47的预热、缓存容量和共同优化不同，不能与新参考直接计算公平倍率。新原生正式C4/C8仍在执行，当前没有独有≥30%收益。

FlashComm1/SP-MoE在固定SDK的TP8/DP1/EPoff下不能激活。FULL、MTP组合及其他8die拓扑仍待审查；现场权重含MTP层，固定Ascend源也包含Glm5NextMTP，不能以缺少实现排除它。正式通用质量、长上下文及独立重建仍未验收。

[脱敏结果与证据哈希](../validation/shared-features-20261111/summary.json)。历史报告保留原数字，但其约1.7倍不满足新目标。
