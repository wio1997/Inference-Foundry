# 最新服务器分工：两套2P2D（2026-10-16）

用户最新决定：原GLM-5.3两种框架都改为**2P2D**；**38和42排除**。45纳入本批并补原模型vLLM D；47恢复Flash SGLang D，原角色冲突已解除。旧SGLang停止已经执行，新组仍未部署。15台Docker和/data/wio统一目录已验证；Flash四台权重复制及逐文件SHA完成，vLLM六台已导入固定候选镜像，SGLang六台B103也已完成上传、导入及父独立imageID/别名验收，15台角色资源准备完成；见[最终验收](DOCKER_RESOURCE_ACCEPTANCE.md)。见[资源路径](DOCKER_ASSET_LAYOUT.md)，本次未启动新模型。

| 实验线 | P节点 | D节点 | 物理规模 |
|---|---|---|---|
| 原GLM53 · vLLM-Ascend | 30、31 | 43、45 | 4台，P32 die、D32 die |
| 原GLM53 · SGLang | 34、35 | 36、41 | 4台，P32 die、D32 die |
| Flash · vLLM-Ascend | 14 | 15 | 2台，1P1D |
| Flash · SGLang | 46 | 47 | 2台，1P1D |
| 备用 | 32、33、37 | — | 暂不启动推理 |
| 排除 | 38、42 | — | 不在本批部署/测试范围 |

12台计划承担推理，3台备用，共15台；管理IP来自已观察到的Kubernetes信息。节点角色为分配，**不表示新环境已部署或已通过性能验收**。机器可读表为 [RESOURCES.json](RESOURCES.json)。

## 实际停机结果

通过本机ZCode DeepSeek执行父审查冻结的完整ID白名单，停止**33个SGLang相关容器（12模型、20辅助、1Meta）**；没有使用全机pkill，也没有删除镜像、权重、源码、配置、容器对象或旧日志。43的AISBench工具容器、所有访问Pod和原K8s资源声明保留。

父代理随后独立读取16台宿主机（含当时尚未排除的42）：256die健康OK、没有NPU进程。42原本未发现SGLang模型级占用，没有对其原作业执行停止。42诊断只临时建立管理Pod打通root只读访问，用户排除42后已撤除该Pod，未改SSH或宿主机文件。38未执行任何停止/修改。

最新停机与重分配证据：[stop-and-reallocation.json](inventory/20261016/stop-and-reallocation.json)。[早一轮只读快照](inventory/20261016/readiness.json)保留当时的运行容器与P47冲突，属于历史，不覆盖本页。

| 节点 | 管理IP | 最新用途 | 停机后实际观察 | 原K8s申请（历史快照，不代表今天空闲） |
|---|---|---|---|---|
| 14 | 172.16.12.14 | Flash vLLM P | 16 die OK；无NPU进程；HBM 2.82–3.08 GiB/die | 先前K8s申请16 die，保留待重核 |
| 15 | 172.16.12.15 | Flash vLLM D | 16 die OK；无NPU进程；HBM 2.81–3.07 GiB/die | 先前无NPU申请；部署前重核 |
| 30 | 172.16.12.30 | 原GLM53 vLLM P | 16 die OK；无NPU进程；HBM 2.81–3.06 GiB/die | 先前K8s申请1 die，保留待重核 |
| 31 | 172.16.12.31 | 原GLM53 vLLM P | 16 die OK；无NPU进程；HBM 2.82–3.06 GiB/die | 先前K8s申请16 die，保留待重核 |
| 32 | 172.16.12.32 | 备用 | 16 die OK；无NPU进程；HBM 2.81–3.06 GiB/die | 先前K8s申请2 die，保留待重核 |
| 33 | 172.16.12.33 | 备用 | 16 die OK；无NPU进程；HBM 2.82–3.09 GiB/die | 先前K8s申请16 die，保留待重核 |
| 34 | 172.16.12.34 | 原GLM53 SGLang P | 16 die OK；无NPU进程；HBM 2.82–3.08 GiB/die | 先前K8s申请16 die，保留待重核 |
| 35 | 172.16.12.35 | 原GLM53 SGLang P | 16 die OK；无NPU进程；HBM 2.82–3.09 GiB/die | 先前K8s申请16 die，保留待重核 |
| 36 | 172.16.12.36 | 原GLM53 SGLang D | 16 die OK；无NPU进程；HBM 2.81–3.08 GiB/die | 先前K8s申请16 die，保留待重核 |
| 37 | 172.16.12.37 | 备用 | 16 die OK；无NPU进程；HBM 2.82–3.09 GiB/die | 先前K8s申请16 die，保留待重核 |
| 41 | 172.16.12.41 | 原GLM53 SGLang D | 16 die OK；无NPU进程；HBM 2.81–3.07 GiB/die | 先前K8s申请16 die，保留待重核 |
| 43 | 172.16.12.43 | 原GLM53 vLLM D | 16 die OK；无NPU进程；HBM 2.81–3.05 GiB/die | 先前无NPU申请；部署前重核 |
| 45 | 172.16.12.45 | 原GLM53 vLLM D | 16 die OK；无NPU进程；HBM 2.82–3.06 GiB/die | 先前K8s申请16 die，保留待重核 |
| 46 | 172.16.12.46 | Flash SGLang P | 16 die OK；无NPU进程；HBM 2.82–3.06 GiB/die | 先前无NPU申请；部署前重核 |
| 47 | 172.16.12.47 | Flash SGLang D | 16 die OK；无NPU进程；HBM 2.82–3.06 GiB/die | 先前无NPU申请；部署前重核 |

AICore瞬时0不能证明无模型；本次判断使用实际NPU进程列表和容器退出状态。无NPU进程也不要求HBM变成0，表中为真实剩余占用。Kubernetes allocatable仍不是free；旧资源申请未删，部署前须重核独占归属，避免预留容器随后启动作业造成冲突。

新2P的物理规模为32die，不能继续照搬4P时的EP64/全局TP64。TP、attentionDP、EP、MTP/图兼容、每rank真实KV容量与MemCache池需要重新冻结。备用节点不默认参与缓存供给或计算，不能偷偷借其资源提高一侧成绩。正式验证仍只用SDMA/HCCS的已证明路径，不能由同superpod标签推断带宽。

43原AISBench容器保留只作工具资产。实际压测客户端搬到独立CPU主机，不能与D43/D45竞争。166处于另一网络且已有Flash作业，不默认其可用或可直达本组。

当前没有新模型请求、2P2D性能数据或自动化恢复。离线镜像/权重仍见 [资产表](ASSET_READINESS.md)。后续构建/部署/测试继续由本机ZCode DeepSeek执行新审查的launcher；不要重放旧4P2D broker或自动回退重启已获授权停用的服务。
