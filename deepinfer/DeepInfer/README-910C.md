# DeepInfer 910C W8A8 适配入口

默认保留已验证的 keep140 路径。官方 YOCO/ZOS 是可选配置，功能通过与性能收益分别判断。现有结果未证明整套 ZOS 稳定快于原生路径。

| 配置 | 源码与作用 |
|---|---|
| `native140`（默认） | 未修改的原生源码，保留现有有效优化与 keep140 |
| `official-yoco` | 本源码树；原 C++ YOCO metadata、COMPACT 与 Q_b 合并，原生调度入口 |
| `official-zos` | 本源码树；原 Scheduler/Worker、持久请求池、晚到准入、跨代换槽、异步采样整数元数据与 DP 局部 YOCO |
| `native256` | 未修改的原生源码，显式选择已有 256 回退 |

原生回退源码固定为 `/data/tiankuan/wio/910c-deepinfer-172/src/DeepInfer-official-fp4-index-v1-20261102`，启动前按独立清单核对。本源码树不修改该目录或权重。

## 启动与停止

在本源码根目录执行。`plan` 只输出命令，不操作服务器；`start` 要求设备和端点空闲，不会停止现有服务。服务名称每次使用新值。

```bash
python3 ascend910c/serve.py plan --service my-deepinfer --profile official-zos
python3 ascend910c/serve.py start --service my-deepinfer --profile official-zos
python3 ascend910c/serve.py inspect --service my-deepinfer
python3 ascend910c/serve.py stop --service my-deepinfer
```

`stop` 先核对 PID 出生钟、设备归属、健康、客户端和已完成队列，只停止该服务的进程。原生回退通过新服务名及 `--profile native140` 启动；256 回退显式使用 `native256`。操作记录和所有服务数据写入 wio 项目目录。

当前部署固定为 172、16 个逻辑设备、DP2×TP8/EP16；权重使用 wio 下的只读挂载。统一扩展放在官方 `_so/ascend-aarch64` 目录，通过标准包导入；桥接模块位于 `deepinfer.engine.zos910c`。运行不依赖诊断 bundle、动态模型覆盖或额外 `sys.path`。依赖标识见 `ascend910c/runtime.json`，绑定构建来源见 `ascend910c/build`。

## 已验证范围与限制

官方框架参照固定提交 `553741557a57df44a46fb3ab72b6aafeaf945337`。保留已验证 W8A8/INT8 路径，当前不重建 FP4、通信或注意力底层精度后端。精度验收采用匹配 vLLM Ascend 的实际 demo 效果相近、无明显退化，不设 CPU/FP64 或逐 token 一致门槛。

连续入口目前从 8 条符合热缓存与 ignoreEOS 条件的请求初始化；初始小批、冷请求、EOS 等剩余范围保留原生回退。晚到请求支持已验证的变长输出与多代槽位。混合 DP 的官方 YOCO 要求关闭全局 query-padding，原因是现有原生分支的条件 collective；这是软件接口组合限制。

标准包的 CPU 导入、IPC 和双 DP Scheduler 检查不等同于整模通过。当前整模验证、性能和交付状态以 `/data/tiankuan/wio/910c-deepinfer-172/diagnostics/deepinfer-stable-package-20261106/v1/RESULTS.md` 为准。已有特性逐项测试证据保留，本整理不新增算子或调度优化收益声明。
