# 002：修复草稿 Markov 权重加载并保留完整四档对照

## 原理介绍

DSpark 草稿模型应读取 checkpoint 中训练好的 Markov embedding/head。原 NPU 名称转换没有把 `mtp.2.markov_head.embed.weight`、`mtp.2.markov_head.head.weight` 映射到实际参数 `markov_w1`、`markov_w2`，加载日志出现跳过。G37 复用模型原有通用别名，并在两份参数任一缺失时拒绝启动。本次实际服务的16个rank均出现加载成功标记；这证明加载路径，不能替代模型质量验收。

## 源码变更

- [ModelSlim 加载修复](../../compat/runtime/g37/modelslim.py)与[缺失即报错检查](../../compat/runtime/g37/native/sglang/srt/models/deepseek_v4_dspark.py)。实际索引143324个名称的CPU检查仅改变两个Markov名称，其余143322个不变，见[检查记录](../../evidence/g37/probes/markov_g37/result.json)。
- 同时保留此前已经测试的16头attention路径、[FP32 mHC按4096行分块](../../compat/runtime/g32/native/sglang/srt/models/deepseek_v4_mhc.py)、[Engram门控按1024行分块](../../compat/runtime/g33/engram_npu_uva.py)、[原生AscendOps启动路径](../../compat/runtime/g35/a3_deepep_startup.py)以及[完整基础辅助源码](../../compat/runtime/g31)。
- [相对固定上游提交的G37累计补丁](../../patches/sglang-a3-1017-g37.patch)和[逐文件清单](../../patches/g37-manifest.json)。补丁覆盖实际上游文件绑定；SDK覆盖和PYTHONPATH辅助模块仍按Docker命令使用。依赖构建配方复用[原说明](../PATCHES.md)，未包含镜像和权重二进制。

## 实测范围

2026-11-13，同一台aarch64服务器，8张物理910C、16个逻辑64GiB设备，驱动25.5.0，Engram DRAM。每档warm16/formal32，输入32768/65536/131072/262144，输出1024，实际峰值C16。两引擎各自正式128/128、预热64/64成功；原始输入逐条一致。正式结果：

| 输入token | SGLang 输出token/s | 原生vLLM 输出token/s | SGLang/vLLM |
|---:|---:|---:|---:|
| 32768 | 174.9960 | 253.6514 | 68.99% |
| 65536 | 131.3399 | 217.9197 | 60.27% |
| 131072 | 85.0005 | 163.8013 | 51.89% |
| 262144 | 39.3416 | 101.1776 | 38.88% |

TTFT、TPOT、E2EL均值/P99和SQLite并发证据完整保留于[SGLang验收](../../evidence/g37/sglang_G37_all4_acceptance.json)与[vLLM验收](../../evidence/g37/native_vllm_baseline_acceptance.json)。吞吐为整批成功输出token除以正式墙钟耗时，预热不计分。修复前不同配置的历史数字不作为修复收益对照。

## 波动噪声

原生vLLM在同一实例、相同配置下仅额外重复过32K一次，为259.7673 token/s，较首轮高2.4111%。一次重复不构成噪声界；其他长度和G37没有重复性验收。逐请求离散不能替代同配置重复测试。

## 结论与适用边界

四档均未对齐或超越vLLM。G37成功记录不代表数值等价或整模型质量通过；781GB权重未重新计算全量payload SHA。固定镜像加本分支适配不代表官方原生支持，也不适用于910B或x86_64。

共同优化尚未完全对齐：原生vLLM开启DSA CP，SGLang未开启prefill CP；SGLang共享MoE传输分支对AscendOps的CP声明存在冲突，实际兼容性未验证。内部准入上限也不同：SGLang全局16、实际每DP4，vLLM每DP32；外部并发同为16。本报告不声称全部共同优化一致或内部配置同构。

## 环境、启动参数与数据入口

SGLang镜像为 `quay.io/ascend/sglang:cann9.2.0-a3-20261111`，ImageID `sha256:08367522c4f9803fb876808c2cd4afe459e7e2b0adaab5a0daf36061edf8e517`，源码commit `d0e354bb7a0a75f2bd72f3724431198abcd9304b`；Torch/TorchNPU2.13、CANN9.2.0 B237。完整上游源码、镜像身份与观察到的Python依赖见[环境版本](../ENVIRONMENT.md)。其依赖快照对应同一ImageID；G37新增运行载荷以[原始manifest](../../evidence/g37/configs/sglang_G37_runtime_manifest.json)及[公开副本来源索引](../../evidence/g37/publication-provenance.json)为准。

vLLM镜像为 `quay.io/ascend/vllm-ascend:v0.30.0rc2-a3`，ImageID `sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b`；vllm-ascend commit `8fde6d4e335fe300e2b949aab0e31659cce0811d`，vLLM commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`。原生引擎未修改源码。

G37实际参数为attention TP4/DP4、普通TP16/EP16、context270336、完整KV每DP1099904、BF16 KV、W8A8、DSPARK5/verify6、全局prefill32768（每DP8192）、HCCL1024、AscendOps无padding、QKV融合、radix/overlap、多流、planstream、CPU亲和、ZBAL graph/VMM，Engram cpu/per_node/all_reduce。全部参数及环境变量以[实际Docker命令](../../deployment/g37/docker_command.json)、[最终server_info](../../evidence/g37/final_records/server_info_after_262144.json)及[服务门禁](../../evidence/g37/service_gate_sglang_ascops37.json)为准；原始manifest中早期“pending”等字段保留历史原值，不能覆盖最终验收。

先按[依赖配方](../REPRODUCE.md)构建，再打印G37命令：

```bash
python3 scripts/launch.py --generation g37 --model-path /path/to/local-ssd/model
```

G37需显式使用`--generation g37`（当前默认G42）；便携启动器只完成离线语法和命令检查，尚未在另一台机器重建验收。实际执行需明确现有公共锁及租约规则；不自动抢占设备：

```bash
python3 scripts/launch.py --generation g37 --model-path /path/to/local-ssd/model \
  --device-lock-dir /path/to/existing/pool-device-locks \
  --legacy-device-lock-dir /path/to/existing/locks \
  --port-lock-dir /path/to/existing/port-locks \
  --pool-policy /path/to/existing/npu-pool-policy.json --execute
```

原实测[16份AISBench配置](../../configs/g37)、[历史runner及守护脚本](../../benchmark/g37/historical)、[SGLang原始四档结果](../../results/sglang/20261113T103258Z_ascops37_full)、[原生vLLM结果](../../results/vllm/20261113T080217Z)及[32K重复](../../results/vllm/20261113T082828Z_v2_repeat32k)均保留。历史脚本有当时的前置状态，作为审计材料，不能直接当作新机器入口。便携AISBench入口见[复现说明](../REPRODUCE.md)。

G37便携压测入口默认只打印计划；资源与服务检查通过后才显式执行：

```bash
python3 scripts/run_benchmark.py --generation g37 --model-path /path/to/local-ssd/model
python3 scripts/run_benchmark.py --generation g37 --model-path /path/to/local-ssd/model \
  --server-container sglang-dsv41-910c --benchmark-lock /path/to/existing/benchmark.lock --execute
```

便携客户端不映射NPU，限8CPU/32GiB。新压测结果仍须检查请求成功、token、时序和实际并发，CLI退出不能自动视为性能验收。

恢复数据与离线验档：

```bash
python3 scripts/unpack_data.py --with-results
python3 scripts/validate_archive.py
```

该历史归档阶段共816条原始请求记录可逐字恢复并按原始SHA核验。文本部署证据中的内部路径与地址已脱敏；原始字节与公开副本SHA分开保存，历史manifest中的原始SHA不应拿来校验脱敏副本。权重、镜像及第三方算子二进制未入库；公开分支不声明当前有在线服务。
