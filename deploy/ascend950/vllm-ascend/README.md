# vLLM Ascend：8×950DT 部署

模型：DeepSeek-V4.1-Flash。镜像：官方 `vllm_cann:ds41`。源码、镜像 ID 与归档 SHA256 固定在 [manifest.json](manifest.json)。

1. 从 [CANN 官方部署说明](https://gitcode.com/cann/cann-recipes-infer/blob/master/models/deepseek_v4_1/README.md) 获取镜像，核对 manifest 中的归档哈希后导入。
2. 准备原始模型、驱动和本机通信拓扑。将本目录复制到部署主机的独立目录，修改 `deploy.env` 的目录、容器名、网卡和端口。
3. 在主机依次执行：

```bash
bash 01_create_container.sh
source ./deploy.env
docker exec "$CONTAINER_NAME" bash /workspace/deploy/02_prepare.sh
docker exec "$CONTAINER_NAME" bash -lc 'cd /workspace/deploy; source ./cann_env.sh; python3 check_routing.py'
bash 03_start.sh
bash 04_verify.sh
```

停止本部署：`bash 05_stop.sh`。脚本核对容器的部署标签，不停止其他容器。

`02_prepare.sh` 使用固定版本的官方 Ascend C 源码构建路由算子，安装到容器挂载目录；`patch_dsa.py` 检查原文件 SHA 后启用原生混合批次重排初始化。脚本不修改宿主机驱动和全局 CANN。

完整参数说明见 [HTML 部署手册](../../../docs/ascend950/vllm-ascend-deployment.html)。这些脚本从实测环境整理，已做语法、补丁与路径检查；整理后的整套新主机部署尚未重跑。性能测试使用的内部配置和数据未包含在本次代码发布中。
