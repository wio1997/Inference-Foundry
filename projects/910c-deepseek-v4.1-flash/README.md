# 910C · DeepSeek V4.1 Flash

当前用户配置：**DP2×TP8，逻辑设备 0–15，共16个；仅 Engram CPU 卸载，不使用主干 CPU prefetch。**

**本轮 DP2×TP8 已通过验证：健康接口 HTTP 200、模型注册成功、文本问题“1加1等于几”返回“2”；16个逻辑设备均检测到本服务 worker。**

本轮运行状态见 [STATUS.json](STATUS.json)。旧 DP1×TP8 的 OOM 和早期 DP4×TP4 的文本结果仅作历史，不替代当前配置验证。

| 项目 | 配置 |
|---|---|
| 目标 | 172，`<TARGET_HOST>`，主机名 `<TARGET_HOSTNAME>` |
| 数据根目录 | `<WIO_ROOT>` |
| 项目文件夹 | `<WIO_ROOT>/projects/910c-deepseek-v4.1-flash` |
| 模型 | `<WIO_ROOT>/DeepSeek-V4.1-Flash-1017-w8a8` |
| 架构 / 量化 | `DeepseekV41ForCausalLM` / Ascend W8A8 |
| 设备与并行 | 16逻辑设备 / 8物理卡；DP2×TP8、EP16 |
| Engram | `cpu_offload=true`，`dp_shared_memory=true` |
| 主干权重卸载 | 关闭 |
| vLLM-Ascend | `quay.io/ascend/vllm-ascend:v0.30.0rc2-a3` |
| SGLang 候选 | `quay.io/ascend/sglang:main-cann9.1.0-a3`；未启动 |
| 容器 | `wio-dsv41-vllm-s172` |
| API | `http://<TARGET_HOST>:8172/v1`，model=`dsv41`；以本轮验收为准 |
| 其他参数 | 8192 context、2048 batched tokens、4 seq/DP、eager、async、无 DSpark |

## 启动与验证

```bash
# Export WIO_ROOT, TARGET_HOSTNAME and PUBLIC_MODEL_SOURCE privately first.
cd "$WIO_ROOT/projects/910c-deepseek-v4.1-flash"
cp config/serve.env.example config/serve.env
python3 scripts/preflight.py
# 当前用户已明确授权本次 privileged 方式。
bash scripts/start.sh --privileged
tail -f logs/current.log
# 等待健康就绪后
python3 scripts/smoke.py
```

启动脚本检查主机、镜像 ID、设备占用、同名容器和端口，不停止其他任务。
模型和容器根目录只读；HOME、缓存、临时文件、Ascend 日志和服务日志均在本项目目录。
Docker 已有镜像层和容器元数据仍由宿主 daemon 管理在 `<DOCKER_DATA_ROOT>`，未迁移全机 Docker 根目录。
项目文件夹由原目录整体改名，历史文件保留；历史原始日志中的旧路径按当时事实保留。

## 已完成的资产检查

153 文件、781,477,434,273 bytes 已全部完成源 SHA256 与独立目标 SHA256 校验。
本地142个索引分片、143,324个索引张量的存在性、safetensors header 和 offset 通过。
见 [镜像与权重身份](evidence/inventory.json)、[复制验收](runs/copy-summary.json)。

当前 smoke 仅检查健康、模型注册和文本生成，不代表正式准确率、性能或多模态验收。
历史8设备结果见 [DP1 TP8 OOM](runs/dp1tp8-engram-only-summary.json)，不能将它当16设备的运行结论。

公开仓库仅保存脱敏模板。真实服务器地址、主机名、部署路径和完整现场证据只保存在授权服务器本地。
