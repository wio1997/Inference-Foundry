# vLLM 原生图端点诊断代码候选（未发布，待 root 验收）

这是实际 gen81 短诊断的任务自有代码子集与 CPU 审计入口。它给 64 层 pre/post 事件附唯一 JSON 标签，以同一 profile 的 raw `EVENT_RECORD` 标签取得 Model/Stream/Task，再使用显式 flow 关联回放边界。它是诊断插桩，未作为推理优化采用；设备端点区间不是独占 kernel 或全 stream 关键路径。四 rank 不相加为墙钟。

## 文件与字节依据

`runtime/` 是从历史部署清单递归选择的 13 个任务模块，根为 `q27_fx_model_bootstrap`。所有静态 Import/ImportFrom 的本任务模块依赖均包含，逐件 SHA 与 gen27 已绑定的实际 gen81 字节相等；动态 import 的框架插件由原冻结环境提供，此处不声称完整服务可复现。`source-bindings.json` 记录每件来源、SHA、角色及依赖图。

`historical-build/` 保存原 gen78 C++ 与原 CPU 构建脚本原字节。原脚本、bootstrap 的 `/task/...` 路径和模块名是历史事实；某些注释仍含 gen53 字样，不是当前模块名。原脚本不能在任意解压目录直接运行。`build_extension.py` 是本轮新增的路径可配置入口，保持原 C++、编译参数、链接库与精确模块初始化名；本轮未执行该入口编译或加载，不能称为已验证可移植构建。重建 SO 的 SHA 不假定等于历史 SO。

`raw-audit/` 保留标签到 raw 图的精确身份审计与反例；`executor-run-gen5/` 和 `native-sg-recovery/gen66/` 保留回放审计及其固定 SHA helper 的原相对布局。此处没有原始 profile、模型权重、上游源码树、二进制 SO、SSH transport、凭据或服务/SDK/controller 启动作业。

## 可直接执行的 CPU 核验

在本目录执行以下命令，输出目录必须不存在：

```bash
python3 check_cpu.py --output PRIVATE_PATH_9804bbe696a2
```

只需 Python 标准库。入口先检查原文件 SHA、语法和静态本任务依赖闭合，再用 `-I -B -S` 的独立进程执行现有 CPU fixture；不会隐式加载 sitecustomize、torch 或 NPU。CPU stub/合成身份反例不能证明实际模型正确性、greedy 等价、端到端收益或实际 204 完成。

## 冻结环境与构建入口

历史设备环境是 aarch64、910B1、TP4、CANN 9.2.0、torch_npu 2.10.0.post7（git `1d88b8608fd1e44545b5804026c7a57c44240bb4`），bucket1、512/C1/out32。精确镜像、启动 argv、来源 SHA 与历史 SO SHA 见 `environment.json`。依赖包括匹配的 PyTorch/torch_npu/pybind/编译器、ninja、CANN 头文件和 ascendcl，以及原安装的 vLLM/vllm-ascend/npugraph_ex；完整版本应以原镜像和构建回执为准，不安装 latest 替代。

仅在匹配冻结环境、独立构建目录上，以下 CPU 编译入口才适用：

```bash
python3 build_extension.py --output PRIVATE_PATH_ca3d43e7bbf2 --cann-root /usr/local/Ascend/cann-9.2.0
```

该入口断言 Host Architecture、torch_npu 版本/git、原 C++ SHA、头文件 API 存在及前后 NPU 未初始化；它记录实际 torch/header/SO SHA。此命令本轮未运行，也不启动服务。动态 glue 仍需用户原来授权的 workflow 锁、唯一作业目录、精确镜像/源码/容器核对及 EOF 清理协议，不能从这个包直接启动模型。若后续重定位历史 bootstrap，必须产出单独派生 SHA 和配置回执，保留原件，不直接替换历史证据。

## 原生审计入口与边界

```bash
python3 raw-audit/audit_raw_identity.py --help
python3 executor-run-gen5/audit_tagged_native.py --help
```

真实审计需要另取完整 SHA 绑定的原始图、回调和 SDK trace 输入；本包只保留代码。无 flow、歧义 tuple、重复标签、跨边界 rows 均保留 unknown，不按时间或序号补全。不能通过调整测试把历史缺口变成已闭合。

当前只交付 vLLM 标签诊断组件。SGLang gen67 的公共 Event/cleanup 及 Q06 shadow observer 是独立组件，尚未纳入本包；最终地图合并器也使用既有 gen19→gen26→gen28 交付，不在这里复制。两份保存的插桩诊断文本存在差异，其根因与 greedy 等价仍未确认。完整私有仓归档、实际 204 SHA 门禁、最终 HTTP/browser QA 和 root 总体验收仍待完成。本候选未编号、未提交、未发布。
