# DeepSeek V4.1 Flash · 172

目标：先让 vLLM-Ascend 在单台 A3 的 8 张物理卡上完成启动与文本接口验证。
SGLang 固定候选已选定，暂未启动。权重复制和本地结构检查已完成；普通容器 NPU 访问预检未通过，vLLM 尚未启动。

| 项目 | 固定值 |
|---|---|
| 服务器 | 172，`<TARGET_HOST>`，主机名 `<TARGET_HOSTNAME>` |
| 数据根目录 | `<WIO_ROOT>` |
| 本项目目录 | `<WIO_ROOT>/projects/deepseek-v41-172` |
| 模型 | `<WIO_ROOT>/DeepSeek-V4.1-Flash-1017-w8a8` |
| 模型架构 | `DeepseekV41ForCausalLM`，Ascend W8A8 |
| 卡数及并行 | 8 张物理卡 / 16 个逻辑设备；DP4×TP4、EP16 |
| Engram | CPU offload、DP shared memory |
| vLLM-Ascend 镜像 | `quay.io/ascend/vllm-ascend:v0.30.0rc2-a3` |
| SGLang 候选 | `quay.io/ascend/sglang:main-cann9.1.0-a3`，`0.5.23.dev315+ge9b22a081` |
| 容器 | `wio-dsv41-vllm-s172` |
| API | `http://<TARGET_HOST>:8172`，model=`dsv41`；通过验收前不可视为可用 |
| 首轮参数 | 8192 context、2048 batched tokens、4 seq/DP、eager、无 DSpark |

实际状态见 [STATUS.json](STATUS.json)，镜像和权重身份见 [inventory.json](evidence/inventory.json)。

## 启动与验证

连接方式由用户提供，认证不进入本仓库。登录后：

```bash
# Export WIO_ROOT, TARGET_HOSTNAME and PUBLIC_MODEL_SOURCE privately first.
cd "$WIO_ROOT/projects/deepseek-v41-172"
cp config/serve.env.example config/serve.env
python3 scripts/preflight.py
bash scripts/start.sh
tail -f logs/current.log
# 等服务就绪后执行
python3 scripts/smoke.py
```

启动脚本拒绝错误主机、镜像漂移、已有同名容器、端口冲突或可见 NPU 进程。
已知其他调优任务即使暂未出现在 npu-smi 中，也应先协调。
脚本只启动自己的容器，不停止其他容器。重启前须检查自己的旧容器状态并手动处理。

2026-10-26：普通 runc 与 Ascend runtime 的显式设备映射预检均返回 NPU 不可用；内核记录 `Conflict open udevid`。这表明存在设备命名空间访问冲突，不能用 npu-smi 的空闲进程列表推断新容器一定可以访问设备。`start.sh --privileged` 是等待用户明确授权的备选命令，尚未运行，也未证明可解决该问题。

布局来自镜像内 `/vllm-workspace/vllm-ascend/docs/source/tutorials/models/DeepSeek-V4.1-Flash.md` 的 A3 单机配置；首轮缩小 context/batch/seq，并改 eager，先验证功能。
镜像中 CANN 实际为 9.2.0，按实际元数据记录。SGLang 的 cann9.1.0 标签属于另一套镜像。

权重目录只读挂载。项目缓存、HOME、临时目录、Ascend 日志和服务输出都指向本项目目录。
Docker 的已有镜像层和容器元数据仍由宿主 Docker daemon 管理在 `<DOCKER_DATA_ROOT>`；没有迁移全机 Docker 根目录。共享内存属于运行内存。

## 验收边界

142 个索引引用分片共 781,343,665,008 bytes，143,324 个索引张量；所有分片头、索引张量存在性和数据 offset 已检查通过。
另已对从 public 来源复制的全部 153 个文件（781,477,434,273 bytes，含配置和辅助文件）逐文件计算源 SHA256，并独立读取目标文件校验 SHA256，全部一致后才将临时目录原子改为最终目录。复制耗时约 300.47 秒；该数字是数据准备时间，不是推理性能。
smoke 包含 `/health`、`/v1/models` 与一次文本请求；它不能代替准确率、多模态、并发或正式性能验收。
只有已产生对应结果文件的项目才能在 STATUS 中标记通过。

公开仓库仅保存脱敏模板。真实服务器地址、主机名、部署路径和完整现场证据只保存在授权服务器本地。
