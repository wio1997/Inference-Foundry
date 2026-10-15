# GLM-5.3 MemCache AISBench 测试脚本 v1

## 脚本作用

这是 GLM-5.3 W8A8 / Ascend SGLang 4P2D 环境的可复用测试版本。入口为 `run_glm_aisbench_prefix.sh`，负责把“独立前缀预热→核验真实MemCache对象→清L1/L2保留L3→AISBench正式计时”串起来。它不部署模型、不重启服务、不安装SDK。

固定测试口径：

| 客户端并发 | 正式请求数 | 含义 |
|---|---:|---|
|16|64|4倍并发量|
|32|128|4倍并发量|
|64|256|4倍并发量|

AISBench按并发上限连续补充请求；“4批”指每档请求总数为并发数的4倍，不设置4次整批同步屏障。

共448条不同请求，每条服务实际80000输入、600输出。每条先预热自身74400-token前缀，128-token页对齐后要求实际L3命中74368（92.96%）。禁止一条数据复制成多并发，也不拿完整请求重复测试冒充93%命中。

**预热和冒烟的时间/tokens不计入正式吞吐。正式请求自身的KV加载、排队、P计算、PD传输及解码时间正常计入。**

## 使用范围

这是当前GLM-5.3/4P2D配置的具体版本，复用已有且审计过的80K文本数据，不重新生成随机文本。默认源目录 `/work/glm53-prefix-data`，需要其中的dataset-audit.json、dataset-manifest.jsonl、datasets/、prefixes/及needle-manifest.jsonl。输入数据及token-ID二进制文件不入库，需另外保存。原始审计要求 `formal_requests=768`、`per_cc=256`；本脚本从中选取448条。`prefix_ids_file` 是绝对路径，迁移容器时需保持原路径挂载，或重新审计路径和SHA。

`SOURCE_ROOT`可替换数据根目录，但源文件必须保持同样的审计结构和SHA；不是任意JSONL直接输入器。源目录应在AIS容器可访问，P审计环境中要有匹配的前缀ID文件。

服务地址和现场路径由本地 `GLM_BENCH_PROFILE` JSON 提供，示例见 [profile.example.json](profile.example.json)。现场地址、凭据、数据和运行结果不入库。`extra_backend_tag` 和 `meta_service_url` 必须与实际P服务严格一致，供给本任务独立的Meta/缓存；清L3接口的影响范围需由部署者确认，不能只凭tag推断共享池隔离。

拓扑检查固定为 P：TP64/attention-DP4/EP64、内存比例0.75；D：TP32/attention-DP16/EP32、内存比例0.90。attention-TP分别为16、2。适用当前GLM模型、分词器、128-token页、原生MemCache和流式usage接口，不是任意模型通用测试器。

## 如何运行

先把整个目录放到已有AISBench容器，例如 `/work/tools/aisbench_memcache_prefix`。需已有可用AISBench、mmengine、transformers、本地可离线读取的GLM tokenizer，以及能报告原生cache counters的SGLang服务；不执行pip安装。

```bash
cd /work/tools/aisbench_memcache_prefix
cp profile.example.json profile.local.json
# 编辑 profile.local.json，填写你自己的地址、独立缓存tag和路径。
export GLM_BENCH_PROFILE="$PWD/profile.local.json"
bash run_glm_aisbench_prefix.sh plan
bash run_glm_aisbench_prefix.sh check --out /work/aisbench-kit-offline-check-01
```

`plan`只打印测试口径。`check`使用合成SSE片段验证当前安装AISBench的配置/缓存字段/流式usage捕获，不发送模型HTTP请求。`run`准备数据后，还会通过安装的AISBench数据加载器逐组加载全部448条并核对SHA，成功后才开始任何冒烟/预热请求。

真实MemCache元数据审计需要在有相同SDK与SGLang源码的P环境执行。把 `scripts/audit_prefix_storage.py` 和 `scripts/bench_profile.py` 放到该环境同一目录，并保存一份对应的本地profile。`p_audit_root` 下需已有 `bench-audit/{dataset-manifest.jsonl,needle-manifest.jsonl,prefixes/}`、与实际服务一致的 `conf/model-memcache.json` 及可写 `evidence/`。复制 `audit_hook.example.sh` 为自己的可执行hook，设置以下变量（也可使用你现有的SSH/kubectl包装脚本）：

```bash
export AUDIT_HOST=<P主机SSH别名>
export AUDIT_CONTAINER=<本任务P服务容器>
export AUDIT_ENV_SCRIPT=<容器内SDK环境启动脚本>
export AUDIT_PYTHON_SCRIPT=<容器内audit_prefix_storage.py路径>
export AUDIT_PROFILE_PATH=<审计容器内profile.local.json路径>
export AUDIT_HOOK=/work/tools/my_memcache_audit_hook.local.sh
chmod +x "$AUDIT_HOOK"
export SOURCE_ROOT=/work/glm53-prefix-data
export RUN_ROOT=/work/aisbench-glm53-memcache-run-01
bash run_glm_aisbench_prefix.sh run
```

hook会收到 `--cc`、`--start-row`、`--audit-label`、`--manifest` 参数，必须执行实际K/V/index对象存在性检查并返回 `STORAGE_PREFIX_AUDIT_PASS` JSON。SDK环境、数据审计根目录和模型backend配置必须对应；不能用sleep替代。

`profile.local.json` 的 `model_path` 必须指向AIS容器内GLM tokenizer；`benchmark_cwd` 为AISBench源码/运行目录。`lock_roots` 必须包含所有使用同一服务的压测工具共同遵守的锁目录；复制到新路径、不同锁目录不等于能够并行运行。入口持有矩阵锁，每步另有全局 `benchmark.lock`，并检查实际服务空闲。

每次使用新的RUN_ROOT。失败保留所有原始记录，不覆盖、不自动重发失败请求。此入口不修改已有后台测试；不要在其他压测运行时同时run，脚本会检查全局锁和服务空闲。

## 执行流程与容量

640GiB池以最多64条独立前缀为工作集：16并发1组、32并发2组、64并发4组，共7组。派生数据保留GSM8KDataset要求的空 `train.jsonl`；ZeroRetriever使用零示例，不增加请求数。各组之间清理仅本任务缓存并重新预热。正式数据从原始数据的不同source_row选择，后续组不复用前一组请求。

先做已知答案L3读回600输出和独立80K/600冒烟；成功后才正式跑7组。两个冒烟不计入448条正式请求，也不计吞吐。

## 输出和如何看结果

- `contract.json`：本次口径。
- `logs/`：每步日志；失败先看对应日志。
- `dataset-audit.json`：448条数据分组及来源SHA。
- `cohorts/<组名>/warmups/`：独立预热记录与耗时。
- `cohorts/<组名>/runs/`：AISBench原始trace与数据库。
- `cohorts/<组名>/reports/cc*/result.json`：该组正式全程与AIS原生稳态指标。
- `MATRIX_COMPLETE.json`：7组完成；仍需独立导出/审计及按下述规则汇总，不能把它视为独立审计完成。
- `SUMMARY.csv`、`SUMMARY.json`：入口自动重新审计448条原始请求后生成的正式吞吐汇总；JSON另保留每组稳态及原生报告路径。

主要看正式输出tokens/s、请求/s、TTFT、TPOT及失败数。输入token吞吐需区分每条逻辑80000与本轮剩余prefill的5632，不能把已命中的74368当作本轮重新计算量。每档聚合吞吐=该档正式输出tokens总和÷各组正式时长之和；预热耗时单列，不能把间隔拼成连续稳态。64并发单组只有64条时原生AIS稳态可能N/A，如实保留；实际D运行/排队数见原始采样，客户端64不等于同时解码64。

[run_cc.py](scripts/run_cc.py) 调用的是真实 `ais_bench --mode perf --num-warmups 0 --num-prompts 64`。`prewarm.py`通过真实router/generate预热tokenID，`audit_prefix_storage.py`验证MemCache对象，`cache_admin.py`做严格空闲/命名空间检查后的缓存管理，`summarize.py`审核每条实际80000/600及74368 storage后计算指标。

## 验证状态

核心测试流程来自当前冻结的实际运行版本；发布时把现场地址/路径抽为本地profile，保留原拓扑、缓存来源与token数检查。语法、plan、配置生成及统计回归可离线验证；已有AIS环境的零HTTP兼容检查结果记录于 `VALIDATION.json`。其中合成回归数值只验证统计口径，不能当作模型性能。完整448条暖缓存性能结果须以真实原始trace和独立审计为准，未随此脚本发布。

当前运行环境使用 SGLang `0.5.22.dev919+g27b68303d`（源 `27b68303d043ddbddf9861c55c26bc8617ea7c35`）和配套官方 MemFabric/MemCache 1.3开发源码（分别为 `a085a7c905a51e8cd55c2ef16c74ffe95ded31bc` / `06495e8a381c65b7f3b2615687e6b93e5672e242`）。这不是“1.3正式版”的声明，也不保证其他版本的AIS内部统计接口兼容，迁移后先运行 `check`。

本脚本使用原生SDK指针接口，无额外Python CPU中转适配。官方SDMA HOST/池DRAM路径内部仍有临时HBM staging，不能把此配置称为一次物理传输或用功能校验推算带宽。
