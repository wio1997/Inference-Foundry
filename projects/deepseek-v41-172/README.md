# DeepSeek V4.1 Flash · 172

用户当前要求：**仅使用逻辑设备 0–7，DP1×TP8；仅 Engram CPU 卸载，不使用主干权重 CPU prefetch 卸载。**

**当前结果：启动失败，8 个 worker 在创建主干 MoE 权重时均发生 NPU OOM，容器已退出，当前没有本项目模型服务。** 设备约 61.28 GiB 可用容量，示例已分配 60.62 GiB，再申请 1.06 GiB 时仅余 464.17 MiB。错误发生在 KV cache 分配前。见 [本轮结果](runs/dp1tp8-engram-only-summary.json)。

权重头部统计：排除 Engram 与 MTP 后，原始主干及其余参数 payload 约 520.31 GiB；8 个设备报告容量总计约 490.24 GiB，尚未计入运行时和 KV。此统计是文件 payload，不等于精确的 NPU 格式化后占用。实际 OOM 已由运行日志确认。

当前现场状态以 [STATUS.json](STATUS.json) 为准；旧 16 设备的启动和文本结果不代表当前 8 设备配置通过。

| 项目 | 当前配置 |
|---|---|
| 目标 | 172，`<TARGET_HOST>`，主机名 `<TARGET_HOSTNAME>` |
| 数据根目录 | `<WIO_ROOT>` |
| 项目目录 | `<WIO_ROOT>/projects/deepseek-v41-172` |
| 模型 | `<WIO_ROOT>/DeepSeek-V4.1-Flash-1017-w8a8` |
| 架构 / 量化 | `DeepseekV41ForCausalLM` / Ascend W8A8 |
| 设备 | 逻辑设备 0–7，共 8 个；对应物理卡 0–3 |
| 并行 | DP1×TP8，EP8 |
| Engram | `cpu_offload=true`，`dp_shared_memory=false`；DP1 不允许 DP shared memory |
| 主干权重卸载 | 关闭；用户明确不使用 CPU prefetch |
| vLLM-Ascend | `quay.io/ascend/vllm-ascend:v0.30.0rc2-a3` |
| SGLang 固定候选 | `quay.io/ascend/sglang:main-cann9.1.0-a3`；仅准备镜像，未启动 |
| 容器 | `wio-dsv41-vllm-s172` |
| API | `http://<TARGET_HOST>:8172/v1`，model=`dsv41`；以当前配置验收结果为准 |
| 其他参数 | 8192 context、2048 batched tokens、4 seq、eager、async、无 DSpark |

## 启动和验证

```bash
# Export WIO_ROOT, TARGET_HOSTNAME and PUBLIC_MODEL_SOURCE privately first.
cd "$WIO_ROOT/projects/deepseek-v41-172"
cp config/serve.env.example config/serve.env
python3 scripts/preflight.py
# 本次 privileged 方式已取得用户明确授权。
bash scripts/start.sh --privileged
tail -f logs/current.log
# 健康就绪后执行
python3 scripts/smoke.py
```

启动脚本检查目标主机、镜像 ID、同名容器、端口和所选设备是否有活动 NPU 进程。
不停止其他任务。旧的本任务容器和日志被保留以便追溯；当前只看同名活动容器、STATUS 和本轮日志。

权重只读挂载，容器根目录只读；缓存、HOME、临时目录、Ascend 日志和服务输出写入本项目目录。
现有 Docker 镜像层和容器元数据由宿主 daemon 管理在 `<DOCKER_DATA_ROOT>`，未迁移全机 Docker 根目录。
普通容器预检报设备命名空间冲突；用户明确授权后，privileged 预检成功识别设备。

## 资产身份与验证边界

权重来源已按原 173 shell history 的 cp 命令确认。源 public 目录与 173 的 config、index 哈希一致。
153 个文件、781,477,434,273 bytes 全部完成源 SHA256 和独立目标 SHA256 比较；临时目录仅在所有文件通过后原子改名。
本地 142 个索引分片、143,324 个索引张量的存在性、safetensors header 和 offset 也通过。
详见 [inventory](evidence/inventory.json) 和 [copy summary](runs/copy-summary.json)。

smoke 只验 `/health`、`/v1/models` 和一次文本生成，不能代替准确率、多模态、并发或正式性能验收。
每轮结果必须标注实际并行与设备数量，不继承旧配置的 PASS。

公开仓库仅保存脱敏模板。真实服务器地址、主机名、部署路径和完整现场证据只保存在授权服务器本地。
