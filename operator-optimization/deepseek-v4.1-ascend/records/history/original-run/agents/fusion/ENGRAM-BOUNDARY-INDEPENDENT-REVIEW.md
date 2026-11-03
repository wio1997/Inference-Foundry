# Engram 层间矩阵归属：独立审查

审查 `agents/diagnosis/ENGRAM-BOUNDARY-EVIDENCE.md` 的源码、配置、参考权重索引及 144 个已保存区间。结论：**形状、位置与算子序列强烈支持 Engram wkv + 普通 gate 的候选解释，但不足以将某条 `MatMulV2` 精确绑定到权重或排除全部同形算子。** 不改变原报告。

## 独立检查

- `engram/layer.py:23–29` 用模型配置构造 `ReplicatedLinear((4−1)×8×256, (4+1)×5120)`，逻辑维度为 **6144 → 25600**。`model.py:1565–1592` 中它在配置层 1、14、Engram 存在且 mask 非空时先于对应 decoder 执行。这与两个边界 `MatMulV2` 的 `[3,6144]`、`[25600,6144]` 匹配。EP8 的六个精确 task 元数据均报告第二输入转置；这与线性权重布局相容，但要以此认定 WKV 仍需权重或 tensor 身份，不能单凭 shape 和转置认定参数绑定。
- 169 当前 checkpoint 索引已只读哈希核对，SHA `c32977056a3e8bb17d2330d566020f49ff65facb022c25035b8d3fcf5d2af3a2` 与本地 172 参考索引字节一致；两层名称各自指向 `quant_model_weights-engram-extra-{1,14}.safetensors`。stage128 只读这两份 169 分片头部，均记录 BF16 `[25600,6144]`；索引和头部仍不能证明历史进程中某个指针来自哪个分片。详见本报告末尾与 `engram-169-header-evidence.json`。
- 两层 `wkv` 的参考量化描述标签均为 `FLOAT`，该描述文件已有 169 checkpoint 文件当前 SHA 比对一致。**FLOAT 标签不等于 BF16 权重或计算，也不是历史已加载对象证明**。trace 中的两输入 `DT_BF16` 是该记录的导出类型，与之相容但不会确认对象身份。
- `engram/common.py:61–72` 的 rotation 分支在输入 `[3,4,5120]`、rotation block 32 时，逻辑分块为 `[3,4,160,32]`；展成 1920 行可与 `[1920,32]×[32,32]` 的另一 MatMulV2 形状相容，但代码没有保证 profiler 一定以此方式 lowering；同形矩阵也可能有其他来源。该 gate 的规约、sigmoid 与 mask 等同窗名称支持关联，但仅名称/顺序不能将各 task 全部绑定到该函数，rotation 是否真正打开仍未知。
- 独立重读现有 144 个样本：0→1、13→14 各 48/48 个窗口均各有恰好一条目标大矩阵和一条 `[1920,32];[32,32]` 小矩阵；12→13 对照各 48/48 为零。这验证保存数据中的计数，**不扩大 profiler 覆盖、不证 tensor 身份或 causal attribution**。

## 其他候选与排除范围

当前快照中对硬编码 `6144/25600` 和 Engram 公式的检索主要定位 Engram 定义；这不是全代码、运行时插件或所有权重 header 的完备 shape 索引，也不能排除另一表达式、变形/融合后的其他同形算子。按源码定义且不考虑额外 lowering/重排，普通 attention 的 Wq_a 和 Wkv 逻辑输入分别为 hidden_size=5120，输出分别是 1280、512；MoE gate 逻辑为 5120→384，与 6144→25600 不同，因此**这些特定原始线性调用的标准逻辑尺寸**不匹配。不能据此排除某些其它投影、不同执行分支或编译生成的同 shape GEMM。

证据可排除“0→1 整段无任何已导出任务”，也可否定“这些区间完全只记录了 HCCL 高层导出行”；但不能排除 profiler 未记录的通信、kernel 内部等待、host 提交/事件依赖、其他独立瓶颈。逐 tensor storage/权重 identity、任务准确 callchain、全部 rank 的 `transpose` 属性和历史运行时分支仍缺失；EP8 六条任务的 transpose 属性已有记录。不能把大矩阵 task 时长直接称作全部 Engram 开销或可消除时间。

本审查仅读本地已保存资料，未访问 172、运行 NPU 或修改服务。

## stage128：169 当前 checkpoint header 与 trace 对照

只读结果 `engram-169-header-evidence.json` SHA-256 `3b2efae1e148066dc96fb1c18484e69b2a9a98ec3d3a7f56abda7d76afc00fcf`；其内部脚本 SHA 与本机 `inspect_169_engram_headers_inside.py` 一致。ZCode stage128 的 exit 为 0、`model_verified=true`（`deepseek-flash`）。receipt 核对历史 baseline 容器 ID、镜像 ID，并确认三个目标文件的 mount 指向 169 `/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8` 内同名文件。索引 13,843,894 字节，SHA 与本地参考索引一致；仅查询两个白名单分片的 safetensors 头部，不读 tensor payload（`payload_bytes_read=0`），也不运行设备。

- `layers.1.engram.wkv.weight` → `quant_model_weights-engram-extra-1.safetensors`；`layers.14.engram.wkv.weight` → `quant_model_weights-engram-extra-14.safetensors`。两个文件头均声明该 tensor 是 **BF16 `[25600,6144]`**，各自声明的 tensor 数据区为 314,572,800 字节（300 MiB）。这只是文件头声明，不是已测载入显存或实际读取量。
- 冻结 `engram/layer.py:23–29,41` 使用 `(engram_max_ngram_size−1)×engram_n_heads×engram_head_dim = (4−1)×8×256=6144` 输入、`(hc_mult+1)×hidden_size=(4+1)×5120=25600` 输出的 `ReplicatedLinear`；`model.py:837–840,1565–1592` 在 Engram 条件开启且 mask 非空时将之插入配置层 1、14 的 decoder 前。
- `agents/diagnosis/engram-matmul-task-links.json` 的 EP8 三轮、0→1 和 13→14 边界共六条精确任务记录：两个输入分别为 BF16 `[3,6144]`、BF16 `[25600,6144]`，属性 `transpose_x1:0|transpose_x2:1`，故与输入乘以文件头所示矩阵转置的尺寸相容。该六条精确属性不自动推广到全部 16 rank 和所有样本。

这些互相独立的证据共同强化 Engram WKV 的归属候选，**仍没有证明具体历史执行的矩阵第二输入就是这两个文件中的权重**：读取是事后当前磁盘元数据；运行时可能存在加载、转换、替代或其他同形算子，trace 未给出相应 tensor storage 地址或加载映射。也不能从 300 MiB 文件头声明推断实际 HBM 读量或对应可消除时间。
