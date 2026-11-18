# vLLM G4：混合 KV 容量估计与历史 D 采样范围

**状态：历史来源的只读摘要和条件性算术，不是现场重采、逐请求资源实测或跨框架性能根因。** 本文不包含原始服务日志、机器地址、私有路径或输入文本。原始日志和固定镜像源码未复制入本增量，其身份和 SHA 留在[脱敏证据](../evidence/vllm_g4_capacity_034.json)。

## 1. 固定镜像中的块数和容量口径

既有历史 G4 D 日志记录 vLLM `0.30.0`，固定镜像 digest 为 `4d683043e4bdb7056bf69c2064acbd535d7c9587ccf76c14f70c467e4c24e823`。原 D 服务日志 SHA-256 为 `0aeb762b860a48e4a88956f390f1318ade28b65aad7d24efc6636132293973ee`、2,953,708 字节；八个 rank 的 connector 注册记录均显示 `num_blocks=2060`。TP0 可用 KV 显存记录为 16.04 GiB。**2060 是单 rank 的配置/注册块数，不能乘 8 冒充一个共享物理池。**

既有只读固定镜像源码观察及 SHA 如下；这些是历史观察，本文没有重新启动容器或重新读取镜像：

| 源文件标识 | SHA-256 | 涉及语义 |
|---|---|---|
| `kv_cache_utils.py` | `2666c9f113584e52e7521058efd2c0d9598544559dc942d3a3da87001a3fedf2` | 多组容量、按 rank 统一配置、scheduler 配置生成 |
| `kv_cache_interface.py` | `1cf202f1a44d5bc5c3832b70b41507687ecceea3784e67a4f003bc0210d6eecb` | attention、Mamba 和统一类型组的最大内存/页语义 |
| `base_worker.py` | `42e6dc76ae5cc59cb6d15ac0d428e62689c2a500289f4b625e847abb4b95f91c` | connector 注册使用配置中的块数 |

既有源码分析指出，connector `register_kv_caches` 在记录 2060 前，从 `kv_cache_config.num_blocks` 赋给自身块数；配置生成会把 rank 块数归一到最小值，scheduler 配置要求各 rank 块数一致并复制首个配置。八 rank 注册值一致与这条源码链相符；**可见材料没有 scheduler 运行时配置的独立内存转储，不能把源码推导写成该转储已验收。**

启动日志记录四个 hybrid KV group 的 block size 均为 640，LCM 为 640；MLA、recurrent state、compressed indexer 等组的页和 alias 布局与单一 full-KV 池不同。日志给出最大上下文 263296 时的 **group-aware 折算 1267265 token / 4.81x**。这里 `1267265` 是按混合组需求估出的容量，不是直接记录的某个单组物理 token 池；不能与 SG 的 full-KV 345728 token 直接作池大小比例或性能结论。[SG 容量观察边界](033-256k-diagnostic-stage-evidence.md#3-kv-资源准入有条件的候选不是根因)

## 2. `428` 是有前提的反推，不是单组实测

既有源码计算逻辑：对每个 device group 计算 `ceil(group_max_memory_bytes / group_page_size_bytes)`，各组相加得到一个最大上下文请求的理论 group-block 总需求；device 并发上限为 `num_blocks / 每请求 group-block 总需求`。如存在 host group，再取其上限与 device 上限的较小值；最终 token 估计为 `int(有效并发上限 × max_model_len)`。

如果 **device 上限是实际主导者、没有更紧的 host 上限**，并使用日志的 `2060`、`263296` 和 `1267265`，则唯一满足向下取整算术的正整数分母为 **428**：

```text
floor((2060 / 428) × 263296) = 1267265
2060 / 428 ≈ 4.813084，日志以 4.81x 显示
```

这是对**全部 group 合计理论最大上下文需求**的条件性反推，不是某一组的页数、实际 256K 请求块分配，或两个 256K 请求已并发驻留的证据。四组的实际类型、每组 page bytes、逐组 max-memory/page 比值及 host 标志未逐项验收；不能将猜测的某组分解当成观测。`2060 × 640 = 1318400` 是简单相乘，**不能替代 group-aware 公式**。若 host 上限更紧，428 也就不能由该日志值唯一认定。

## 3. 历史 256K cold/hot 周期采样

已有 vLLM G4 [256K 原始测量审计](../evidence/G56_256K/audit_results.json)记录 cold/hot 各 32/32 成功、客户端配置及观察最大并发均为 16，输出 1024。以下窗口是依据客户端结果目录时间和持续时间推得的 2026-11-17 **近似** UTC 窗口；跨容器时钟偏差未校准。D 的周期日志观察不等于请求级时间线。

| 窗口 | 样本 | Running=0 | Running=1 | Running≥2 样本 | Waiting=0 样本 |
|---|---:|---:|---:|---:|---:|
| cold，约 19:06:19–19:42:02 | 119 | 62 | 57 | 0 | 119 |
| hot，约 19:42:50–20:18:36 | 115 | 62 | 53 | 0 | 115 |

因此只能说这 234 个周期采样点中 D 的 `Running` 最大观察值为 1；**不能排除采样间短时并发，也不能证明每个 Running=1 都是完整 256K 请求或两条请求从未同时驻留**。日志缺少可配对的请求 ID、长度和每组实际块分配/释放证据；客户端 C16 与 D 状态不是同一口径。vLLM `Waiting=0` 也不能与 SG D 的 prealloc queue 定义直接等同或互相替代。

## 4. 复算和后续证据

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/reconstruct_034.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/test_reconstruct_034.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_archive.py
```

[复算脚本](../scripts/reconstruct_034.py)根据[脱敏摘要](../evidence/vllm_g4_capacity_034.json)独立计算满足整数下取整的分母区间、记录采样计数一致性，并显式标记 device 上限主导与实际逐请求块数**尚未验证**。[测试](../scripts/test_reconstruct_034.py)包含小整数穷举、采样开区间和不一致记录的合成测试。它只能验证记录数字和算术，不能在不持有原日志和固定镜像源码时独立复算历史日志筛选、源码观察或实际运行行为。

要判断两个长请求是否同时拥有完整状态，需要请求 ID/长度、每 group 块分配与释放、共享和可驱逐前缀、时间同步及 P/D connector 状态；现有历史日志若无这些字段，应明确“无法证明”。此结论不要求现在重启服务或发新请求。
