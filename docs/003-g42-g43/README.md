# 003：G42 完整四档与 G43 行分片初测

## 原理介绍

G42 在 G37 已修复草稿 Markov 权重的基础上，复用 CANN RMSNorm 返回的 FP32 逆均方根统计，保留原 FP32 投影、Sinkhorn 与 lagged combine。它减少归一化临时张量；实际 NPU 组件 93 个用例通过原先的严格阈值，16384 行分配峰值由 672292352 降至 337139712 字节。这是组件分配量，不是端到端收益。

G43 进一步在 attention TP4 内按连续行保存 HC 残差与 MoE 输入，每 rank 处理约四分之一的行。例如每 DP 的 8192 行变成每 rank 的 2048 行。先把全局行数补齐到四的倍数，按相等块分片；在原始完整 attention、Engram 和模型尾部需要完整顺序时 gather，attention 原有 all-reduce 不变。未开启通用 SP/CP，也未删除其兼容性断言。

## 源码与补丁

- [G42 RMS 统计复用源码](../../compat/runtime/g42/native/sglang/srt/models/deepseek_v4_mhc.py)，[累计补丁](../../patches/sglang-a3-1017-g42.patch)，[逐文件清单](../../patches/g42-manifest.json)。
- [G43 HC 行布局辅助模块](../../compat/runtime/g43/hc_row_shard.py)，[完整模型接入源码](../../compat/runtime/g43/native/sglang/srt/models/deepseek_v4.py)，[累计补丁](../../patches/sglang-a3-1017-g43.patch)，[逐文件清单](../../patches/g43-manifest.json)。
- [G43 实际16-rank组件验收](../../evidence/g43/sglang_G43_HC_row_component_acceptance.json)：1536 个 HC 相关实际用例，FP32 系数最大差 2.2649765014648438e-6；给定相同系数时 BF16 combine/post、尾部行顺序及变输入图路径通过。MoE 使用 identity-expert INT8 传输夹具，不能替代真实专家 GEMM、共享专家或整模型质量测试。
- [组件源码](../../benchmark/g43/component/probe.py)、[16-rank原始结果](../../evidence/g43/component)、[G42组件源码](../../benchmark/g42/component/probe.py)与[原始结果](../../evidence/g42/component/result.json)均入库。

累计补丁相对固定上游 commit `d0e354bb7a0a75f2bd72f3724431198abcd9304b`，只涵盖实际绑定的上游文件；SDK 覆盖、PYTHONPATH 辅助模块及厂商算子仍须使用完整 Docker 命令和依赖配方。实际载荷原始 SHA 见各版本 runtime manifest，脱敏副本 SHA 见[本次来源索引](../../evidence/g43/publication-provenance.json)。

## 实测范围

2026-11-13，在同一台 aarch64、8 张物理910C/16逻辑die、每die64GiB、驱动25.5.0的服务器测量。模型为 Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8，本地 SSD 权重、Engram DRAM。每档独立预热16条、正式32条，输入32768/65536/131072/262144，输出1024，配置和实际峰值并发均为16。预热与正式使用不同输入、每阶段前清缓存，radix cache 开启；吞吐为正式整批总输出token数除以墙钟耗时。

| 输入token | G42 输出token/s | 原生vLLM 输出token/s | G42 平均/P99 TTFT秒 | G42 平均/P99 TPOT毫秒 |
|---:|---:|---:|---:|---:|
| 32768 | 247.7817 | 253.6514 | 16.2821 / 30.5593 | 43.1 / 79.1 |
| 65536 | 130.5071 | 217.9197 | 32.7670 / 69.2011 | 84.0 / 170.8 |
| 131072 | 84.2510 | 163.8013 | 60.5079 / 129.1180 | 124.1 / 265.1 |
| 262144 | 39.5832 | 101.1776 | 127.4165 / 281.1498 | 267.7 / 662.5 |

G42 正式128/128、预热64/64成功，四档均未对齐vLLM。完整 TTFT/TPOT/E2EL、SQLite并发、输入输出及SHA合同见[G42验收](../../evidence/g42/sglang_G42_all4_acceptance.json)、[最终记录](../../evidence/g42/final_records)和[同机对照](../../evidence/g42/G42_all4_comparison.md)。

G43 已接受一次32K初测：**259.8607 token/s**，正式32/32、预热16/16、实际峰值C16；平均/P99 TTFT为13.8580/28.0987秒，TPOT为42.5/97.1毫秒，E2EL为57.3371/114.0335秒。原生vLLM首轮为253.6514、同实例一次重复为259.7673 token/s。G43与后者仅差约0.036%，平均TTFT比原生首轮仍高约104%；不能据此宣布稳定获胜。见[G43验收](../../evidence/g43/sglang_G43_retry2_screen_acceptance.json)。

## 波动噪声

原生32K单次重复较首轮高2.4111%，不能作为噪声界。G42早期32K筛选曾为217.3597、227.6248，最终四档中为247.7817 token/s；这些数值表明运行间变化，不能把最终值与G37的差直接归因于RMS改动。G42长输入相对G37分别约-0.63%、-0.88%、+0.61%，没有长输入改善证据。

G43相对G42最终32K吞吐观察值高4.87%，同时P99 TPOT高22.76%、P99 E2EL高15.85%。G43没有完成重复性验收。首次启动与第二次启动的源码和启动命令相同，但自动random_seed分别为821228692和827671776，不能声称所有有效配置完全相同；[配置差异核对](../../evidence/g43/inspection/G43_retry2_same_configuration/proof.json)保留该差异。

## 中止、失败与适用边界

首次G43筛选在预热期检测到另一容器的独立NPU上下文后保护中止。第二次启动的32K初测有效，但后续四档任务在32K CLI结束后再次检测到独立外部NPU上下文，整体未验收、64K未开始。上下文出现的确切起点未知，不能断言它必然与正式请求计分时间重叠；保护中止后的那批结果不作为有效分数。见[中止说明](../../evidence/g43/G43_resource_interruptions.md)及[失败证据](../../evidence/g43/interrupted)。本分支不声称仍有在线服务。

G38/G39 的DP预填充协调实验遇到18/20行元数据不一致，未取得有效性能；G40/G41完整HC融合未通过既定严格数值门槛，未采用。G44仅离线候选准备，未部署、未测收益，不纳入本分支已运行适配。

整模型质量、数值等价性和稳定收益未验证；约781GB权重未计算全量payload SHA。固定镜像加本分支适配不等于官方原镜像开箱支持1017，也不适用于910B或x86_64。两引擎共同优化尚未全部一致：vLLM启用DSA CP，SGLang未启用prefill CP；内部准入为SGLang每DP4、vLLM每DP32，外部并发均为16。不得据此称“所有优化完全一致”。

## 环境、依赖与启动参数

镜像、依赖沿用[精确环境说明](../ENVIRONMENT.md)：SGLang `quay.io/ascend/sglang:cann9.2.0-a3-20261111`，ImageID `sha256:08367522c4f9803fb876808c2cd4afe459e7e2b0adaab5a0daf36061edf8e517`，Torch/TorchNPU2.13、CANN9.2.0 B237；上游源码包与完整观察依赖均已入库。依赖构建复用[原配方](../REPRODUCE.md)，二进制从固定公开镜像获取后按原清单核验。

两版均为普通TP16、attention TP4/DP4、EP16、context270336、实际KV1099904/DP、BF16 KV、W8A8、DSPARK5/verify6、全局prefill32768、HCCL1024、AscendOps prefill alltoallv/decode low_latency且无padding、QKV融合、radix/overlap、多流、CPU亲和与ZBAL graph/VMM。Engram为cpu/per_node，配置lookup=all_reduce，但实际NPU适配直接读取映射DRAM，不能据配置名推断有该collective。

完整环境和参数以[G42实际Docker命令](../../deployment/g42/docker_command.json)、[G43实际Docker命令](../../deployment/g43/docker_command.json)、[G42最终server_info](../../evidence/g42/final_records/server_info_final.json)、[G43实际server_info](../../evidence/g43/readiness/server_info.json)为准。G42开启`SGLANG_NPU_MHC_FUSED_RMS=1`；G43额外开启`SGLANG_NPU_MHC_ROW_SHARD=1`。原始manifest中“pending”等字段保留创建时原值，验收范围以最终acceptance为准。

便携启动器默认G42，因为它完成了四档；G43须显式选用。以下仅打印命令，不占卡：

```bash
python3 scripts/launch.py --generation g42 --model-path /path/to/local-ssd/model
python3 scripts/launch.py --generation g43 --model-path /path/to/local-ssd/model
python3 scripts/run_benchmark.py --generation g42 --model-path /path/to/local-ssd/model
```

真实执行前须协调稳定独占窗口，并使用该机器现有设备锁、端口锁及租约机制。便携脚本`--execute`参数见[复现说明](../REPRODUCE.md)；它在本次只做离线命令检查，未在新机器重新部署验证。历史守护及验收脚本见[审计脚本](../../benchmark/g43/historical)，不得直接重放旧PID/状态。

## 原始数据与离线核验

[G42完整四档原始数据](../../results/sglang/20261113T124625Z_ascops42_full)、[G43有效32K初测](../../results/sglang/20261113T140855Z_ascops43r2_screen)保留输入引用、预测、逐token时序、SQLite、CLI输出和配置。与早期归档共计1056条请求记录（包含预热），均可逐字恢复并按原始SHA核验。输入没有改写或重新生成；引用仅去重。部署日志脱敏压缩保存，原始SHA与公开副本SHA分开记录。

```bash
python3 scripts/unpack_data.py --with-results
python3 scripts/validate_archive.py
```

上述离线检查只核对归档完整性、数据恢复和指标算术，不是新的硬件性能或质量验收。
