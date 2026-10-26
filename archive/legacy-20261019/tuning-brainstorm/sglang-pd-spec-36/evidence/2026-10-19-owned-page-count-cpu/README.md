# M005/M017：producer 显式页计数候选的 CPU 调用链差分

新版候选通过了 819 项组件级 CPU 检查：59 组 producer/bridge 配对、3 组直接 allocator stale 回归、4 个 private-helper gate 检查。原 NPU allocator 类 AST 不变；计数由已核验的 ordinary、PD、EAGLE producer 显式选择传入。候选仍未接受为原生优化，整体 serving 保持 E0，已接受 Ascend PD+MTP 基线为空。

源码固定为 `ce0a8fea5f4245839d174b2b8c55692be1810a7f`。R001 生成四文件隔离 patch 并检查语法/AST 修改范围。R002 完整导入 baseline 与候选 NPU 模块；调用方执行完整选定函数体，未导入完整 allocation、PD、EAGLE 模块。测试临时绑定候选 NPU 模块以解析新增 lazy helper import，完成后恢复 canonical module binding，安装源码未改变。

## 改法与范围

候选在 NPU 模块增加 private helper，只有 exact NPUPaged 类、page>1、无 c128 标记时返回 CPU 页计数。原 NPU allocator 的构造、默认 num_new_pages=None 算法、显式 count、kernel 选择、free、decode 方法全部保持原类 AST。

ordinary alloc_for_extend 使用自己构造的 CPU prefix 和同批配对 seq CPU；EAGLE 在 H2D 入队后使用自己刚创建的 cur/next CPU；PD prealloc 从本次 Python prefix/fill 标量构造 CPU 数组。计数使用实际 allocator.page_size，包括参考 DCP 缩放单位。

ordinary helper 与 alloc_for_spec_decode 增加默认 None 的内部可选参数，只有非 None 时向 allocator 转发 count。legacy helper、非 NPU context、其他 exact type、子类、page1、c128 不自动选用该 helper。已有 allocator optional-count 能力属于基线；本轮新增的是 owner 侧选择与传递。

L024 的默认方法候选仍保留为反例档案，未被覆盖。本轮直接 allocator stale 回归中，device 200 页、CPU 201 页仍遵循旧默认 device 计算，两侧 loc 与页池一致，未再出现 L024 的额外一页消耗。

## 检查结果

| 类别 | 记录数 | 结果与边界 |
| --- | --- | --- |
| ordinary producer | 18 | 完整 alloc_for_extend 函数体；短/长、前缀、B2/3、page128/256/512、容量拒绝、negative context、strict adapter、page1 |
| PD prealloc producer | 17 | 单请求 prefix0/127/128/129、512/32768 输入及负向 context/strict adapter |
| EAGLE producer + spec bridge | 15 | reserve8 参考与大 reserve 诊断；B2/3、page128/256/512、strict/context/page1 |
| EAGLE 已有余量无需分配 | 3 | 无 allocator call，原请求元数据更新保持一致 |
| legacy helper | 3 | 未显式 opt in，仍不传新 count 关键字 |
| 直接 NPU non-DSV4 wrapper | 3 | 验证其对新 helper 显式 count 的转发；不是实际 NPU device dispatch |
| allocator stale 回归 | 3 | 原默认 API 不自动读取 stale CPU 值，真实 Torch loc/页池一致 |
| private helper gates | 4 | page1/subclass/c128/unsupported；仅 gate fixture，不代表真实替代 backend 支持性 |

前 59 组配对中，16 组返回、3 组容量拒绝、40 组在 native 模块 import 前停止。16 组返回含 page1 与无需分配用例，不能一律计作大页 alloc_extend 成功。大 reserve 仅为执行 Torch metadata 路径的诊断，不是实际 MTP reserve 配置。

每组比较 outcome、真实 allocator 剩余页 tensor/容量、loc 的 dtype/字节摘要、req-table fixture、aux 调用及源函数更新的请求字段。在 allocator entry 记录实际 num_new_pages，核对独立整数页 oracle；只有合格 owner route 才出现显式 count。strict-signature forwarding adapter 不接受新关键字，用于检查 unknown allocator 的默认参数边界；其后端仍为 CPU metadata，不宣称测试了其他真实硬件 backend。

## 实验隔离与未执行的部分

req slot 分配、eviction、int32 req-table write/aux、DSV4 hook 是明确 fixture。源 owner 函数的 allocated/committed/decode index 等字段逻辑实际执行；这不证明真实 req pool、complete scheduler、PD 网络或设备写入。EAGLE CPU device 使用真实 dispatch map 的 CPU 默认 helper；NPU wrapper 单独检查，不把 CPU 执行称为 NPU dispatch。

baseline 与 candidate 使用真实 NPU allocator 类在 Torch CPU 上管理 slot/page metadata，kvcache=None、debug=False，无 KV payload。带标记 CPU Tensor 统计逻辑 device `.item`；不是硬件等待次数或耗时。小页路径在真实 sgl_kernel_npu import 请求处抛出观测异常，发生在 import/launch 之前；没有 fake kernel 或伪造 native 输出。

page512、raw32768 的 B2/B3 仍低于 200 新页，本轮保留该边界，没有把它们改造成大页 Torch 成功。原生小页成功、kernel 编译、完整输出、物理 KV、PD 消费者和设备读结束均未验证。

R002 任务 CPU 容器已删除，参考服务 ID、启动时间/restart count、四个只读 metadata/tokenizer 文件保持一致。无 accelerator 设备、权重或目标模型推理请求。

## 接受门槛与下一步

本轮支持组件级调用和元数据等价性，未测性能，也未接受生产改动。owner 自己持有配对镜像是前提，不能推广成“任何 CPU 参数均新鲜”。新增 host tensor 算术、类型判断、lazy import 和参数转发有成本，native wait 已隐藏或 device 已就绪时可能无收益。

更关键的是，原默认 item 可能提供隐含主机等待。显式传 count 会绕过它，CPU 差分不能证明 H2D buffer 寿命、req-table WAR、stream event、PD transfer/free 与其他设备读者的顺序仍安全。下一轮先追踪真实 plan/schedule stream 与 allocation call 的 source 依赖，再由任务自有 Ascend PD+MTP 原生 trace 和同条件 warmed baseline 验证。

任何 loc、dtype、页身份/容量、None/OOM 行为、请求输出或生命周期不一致均否定候选。没有原生同条件性能与噪声，不能填写加速比或宣称逼近物理边界。Loop INCONCLUSIVE，candidate_accepted=false。
