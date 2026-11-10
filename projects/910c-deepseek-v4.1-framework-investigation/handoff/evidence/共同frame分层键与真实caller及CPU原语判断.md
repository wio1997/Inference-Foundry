# 共同frame分层键与真实caller及CPU原语判断

介绍：沿目标 runner→CP 包装→global/local base→原 metadata 回调，确认同代 frame 可以共享哪些事实，保留哪些字段、写入和稳定地址。不是按 shape 缓存整对象；每个替换原语逐位一致。

到了哪里：12 当前磁盘源全 SHA 匹配，11 份读 29 个限定行段；原 helper AST 的标量整数模型 1728 用例、46656 比较与 8 个错误 key/合并配方。另只读已有 16 worker CPU 日志，用进程内 parent IDs 追 caller。随后在原 owned image 的私有目录执行实际 Torch2.10.0+cpu 原函数与缓存原型，1728 合成整数用例、44928 次字节比较，0 差异；可选 out 的写入旁路一致。它没有安装到服务，也不证明完整框架/NPU 结果。

真实 caller 已校正：每 rank 的 11 个 runner frame 各有 26 次 CP/helper、52 次 base；另有每循环 1 次 draft CP/helper、2 次 draft base。原“27/54”是两个角色混合计数，不能当一个兼容 frame。日志没有参数值，caller 计数不代替输入/alias 证明；原 2 个 FT/body 未匹配范围仍保留。`prepare input` 的源区间调用 runner metadata，不包住 draft propose，不能把 draft 的一次工作也认作重复目标工作。

|边界|共同事实与必要条件|必须保留|
|---|---|---|
|raw CPU helper|封闭调用代次/角色/实际 TP 上下文，numReq、numInput，所选 CPU QSL/长度的内容与 dtype、causal、optional-out 模式|CP118 的 out 均为 None；其它 Legacy caller 写稳定 qsl/length/start-pos，必须旁路或完整重放写入|
|post CPU|再加入 actualTokens、裁剪 start/end、目标 QSL dtype 与原标量结果|不能用 raw key 缓存 actual=0 与 actual>0 的裁剪结果|
|device projection|原设备 seq_lens 的生产代次/依赖，原 QSL 与原整数计算顺序|CPU mirror 可为上界，不能替代设备权威长度|
|每资源交付|cache kind、ratio、物理 table/slot、ring policy、native attrs、输出 group/id 和图描述符|原独立对象和稳定地址、COMPRESSOR/ATTENTION/INDEXER 发布顺序与全部读者|

原 CPU helper 的 TP 区间只读取真实 get_tp_group；选中 CPU 分支没有可选输出时产生新的 qsl/length 结果，包装层 clamp/cast 是 out-of-place。目标 runner 每次新建 batch 字典、按 KV group 浅拷贝公共输入；资源表和 slot 各自保留。这支持把原首次 CPU 计算放入一个有明确输入租约的 frame。需要调用方保证这段期间没有写入；Tensor identity/_version 并不能单独证明这一点，真实 Torch 的 NumPy 视图写入已显示 version 不变。

两个长度字段不能合并：runner 的 `_seq_lens_cpu` 是 optimistic/corrected host mirror，async 时公开 `seq_lens_cpu` 可以为 None；CP 先读内部字段，`common.replace(seq_lens_cpu=localCPU)` 只更新公开字段，其它 dataclass 字段仍浅继承。替换时把二者都设为 localCPU 会改变原接口。CPU correction 等待上一轮接受计数 D2H 完成后才写 optimistic buffer；共享不得提前到这个修正前，也不得跨迭代只按 pointer/shape 复用。

DSpark 的边界也不同：`set_inputs_first_pass` 会克隆并扩长 host mirror、重建 QSL、改 token 数/slot/causal 等。LLMBase 虽准备 shared_dsa_draft_cache，当前 V4.1 `build_for_drafting` 却直接 `self.build(0, common)`，丢弃 kwargs；不能引用另一个 Legacy DSA 的 `_cpu_local` 缓存就说本路径已共享。观察窗中 draft 每轮本来只有一次，不为该角色计 26 倍省时。跨角色、其它模式与 forwarding 变更须另判。

base 已共享 batch 坐标；SMLA/QLI 回调仍捕获各次 common.query_start_loc 和 max attrs，`_publish_task` 按 key 保留首次 buffer/id。只看到 base 命中，不能删各 builder QSL 写入或更换捕获地址。CPU 原型仅省 raw helper 重算，所有后续 pin/H2D、设备长度、slot/ring、回调和输出都未替换。更大范围应继续查：在原主流只计算一次局部整数 device projection、保留每 builder 稳定 QSL 复制，是否已覆盖全部实际读者。

实际 CPU 核验在已安装 Torch2.10.0+cpu 上执行，版本与冻结 catalog 相符。明确关闭后端 autoload，Python -S 后只加已核 site 路径；前后均无 torch_npu/vllm 和 libtorch_npu/libascendcl/libhccl。只导入 Torch CPU，不加载模型或启动 NPU。程序 12.60s 完成，原服务未动；文件仅在 `/data/tiankuan/wio/.../runtime/cpu-frame-run-20261110T104319Z`。刻意关闭 TorchNPU 的隔离原语测试，不等于服务插件状态下的整框架验收。

六种合成形状/秩的 26-call CPU 批次，各 15 轮交替 A/B、每臂 10 个 frame：原函数中位 2.423–2.650ms，带首输入快照和 key 检查的原型约 0.196–0.206ms；逐轮配对中位净省 2.226–2.445ms/frame。完整正负样本保留。这是本地 CPU 调用成本，没有 NPU/device 发布、真实请求级关键路径，也不把 11 轮乘到原 C8 请求。

原诊断价另按真实 26 caller 更新，并更保守地支付首次 helper 整个 body、首次 CP 去掉已知子区间后的 body 余部，加时钟余量；整个 base builder 仍保留。新增每循环 0.5ms、暴露转移系数 0.5 时，helper-only 条件净为新诊断窗口 1.964%；CP 自身另有 25% 可共享时为 3.275%。它们取代“27 次均值首调用”的当前排序口径，旧 1.995%/3.622% 原件作为敏感性保留。144 情景含负值，不能与 CPU 微基准或其它窗口相加。

结果：**已确认可实现，限定原语**——封闭不可变输入租约内的原 CPU helper 缓存复用，已有实际 CPU 原型、逐字节与局部成本证据。**待确认**——服务集成、全部保护负载，以及更大的 post/device common projection、原生 packet/事件/末读和请求净收益。**已停止**——全局 helper cache 跳过 out 写入、合并两个 CPU mirror、遗漏 TP/causal/dtype/actualTokens、仅凭 Tensor version 跨 NumPy 写入，以及把 Legacy draft 缓存或 26 次重复套到当前 V4.1 draft。

框架替换准入仍为 0，隔离 CPU 原语原型通过 1。全链路未扫全。下一条继续原主流公共 device projection→base 首 miss 与所有 closure 读者、稳定 QSL 输出和图事件；不能只把 common writer 挪到异步流而保留 eager 读者。证据见 common-frame-key-{dependencies,cpu,source-review,source-excerpts,frontier}、caller-{logs,price}、torch-cpu 和 import-guard；首次 SSH banner timeout 原件保留，后续 12 源只读重取成功，没有重启任何服务。
