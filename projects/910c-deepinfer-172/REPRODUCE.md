# 复现和原vLLM恢复

已有172服务保留运行；以下是交付入口，不在交付时重跑。简单执行工作按用户要求交给本机ZCode中配置的DeepSeek，
本轮实际rollout模型已核验为`deepseek-flash`。认证从当前Mac钥匙串直接进入SSH，不写入这些文件。

## 源码和依赖

补丁相对官方`deepseek-ai/DeepInfer@553741557a57df44a46fb3ab72b6aafeaf945337`，
包括170继承内容及172完整修改；本地最终checkpoint为`55ae7d193425fe8f20531809ab7f9aebaee52c08`，未推送官方origin。
在独立源码目录checkout该基线后执行 `git apply /path/to/source/deepinfer-910c-w8a8.patch`。
保留Apache-2.0许可；补丁源码中的原始引用/注释随源保留。该原生入口不需要编译170的`native_C`或重建950扩展。

使用本次已有固定image ID；标签可能更新，禁止用不同image替代后仍声称复现。
[容器清单](config/container-manifest.json)和[补充元数据](config/container-extra.json)保存实际runtime=`ascend`、entrypoint、
readonly、privileged、binds、设备映射、环境覆盖、64GiB shm、ulimit和log-driver=none。
驱动来自同一宿主只读挂载；依赖版本见package-versions和aisbench-package-versions。

AISBench源固定`7bcb6440da11bef020943b2af9ed7c1ac8362306`（https://gitee.com/aisbench/benchmark）。
现有源/venv分别位于项目`tools/aisbench-7bcb6440da11`、`tools/aisbench-venv-7bcb6440da11`，不修改旧vLLM项目。
重建工具环境时按该提交的requirements及`plugins/prefix_cache`安装，并使用交付的版本清单和`sitecustomize.py`；
同一镜像的CPU执行环境不初始化NPU。不要安装不同版本的tokenizer代替固定镜像内的deepseek_v41 tokenizer。

## 恢复数据布局

1. 先验证[evidence/archives.json](evidence/archives.json)的两个archive SHA256，再解包检查各自SHA256.json。
2. `final-native-v8.tar.gz`解包根为`native-v8/`：AISBench原始数据、worker/API控制记录、源码、实际执行controller、
   container元数据。放回项目`final/native-v8`，保持这一冻结副本不变。
3. `reproduction-v8.tar.gz`根为`reproduction-v8/`：原始冻结输入/种子/参考预测、AISBench适配器、历史状态和日志。
   将其`workload/frozen`恢复为项目同名路径，`tools/aisbench-adapter`恢复为同名路径；其余文件作证据保留。
4. 将本目录放在`/data/tiankuan/wio/910c-deepinfer-172/delivery`，源码放`src/DeepInfer`，
   既有权重按config中只读挂载。准备项目`tmp`、`home`、`cache`、`build`、`logs`、`runs`、`tools`目录。
   不重复复制/校验全部权重；本次153个payload文件已通过源/目标SHA验证，权重一直只读。

## 启动/基准入口

所有wrapper默认只打印计划，只有`--execute`实际操作，且只接受172主机。

```bash
cd /data/tiankuan/wio/910c-deepinfer-172/delivery
python3 scripts/create-containers.py
python3 scripts/start-native.py
# 仅在确实需要恢复启动、设备空闲且配置已核对时执行：
python3 scripts/create-containers.py --execute
python3 scripts/start-native.py --execute
```

已有健康native服务会保留；端口占用或NPU忙时拒绝启动，不停止别人进程。新启动使用新control/status/log路径，
校验实际已部署源码SHA清单，以16个torchrun worker加载40层；后台dispatch不代表READY。
实际加载与API命令原文见`controllers/native_service_v8_job.py`和最终archive的`serving-v8-status.json`。
wrapper只更改新run目录和日志名，并显式固定prompt lookup63；本次实际参数来自同一默认值。

如未来确实需要重复正式基准，使用 `python3 scripts/run-benchmark.py --execute`。
它使用新输出目录、同一输入/参考预测和AISBench规则；种子预热后做C8/32输出的必要图/正确性预热，再做两轮C8/1024正式测量。
预热不计时；每轮核对完整文本、输入/输出/235904实际命中计数，失败即拒收。
该脚本未在交付时执行，避免额外测试；本次记录的测量由随包原始controller实际执行。
vLLM基线的排除预热是完整C8/1024一轮，与native预热长度不同；二者正式计时前实际图均已预热。

## 回切原vLLM

```bash
python3 scripts/restore-vllm.py          # 只看计划
python3 scripts/restore-vllm.py --execute
```

该入口检查原容器完整ID与image ID、外部占用、native进程归属；在任何停止前备份原日志，避免`docker start`中`>`截断历史。
只向此项目的torchrun/API父进程发SIGTERM，等所有8物理设备和8172端口空闲，随后启动保留的原容器。
身份变化、外部任务、进程不唯一或无法确认空闲时拒绝操作，不强杀、不重建原容器。
启动容器后仍需读health/日志确认就绪；此wrapper不会追加推理测试。
原始停止前配置/日志备份在项目`baseline/before-stop-20261026T233954`；原基线commit保持不变。

交付wrapper只做离线语法/默认计划检查，未为了验证脚本重启服务或回切vLLM。
