# 本项目现有访问入口

Mac到资源源节点1：`ssh -S /Users/wio/work/sglang-glm53-pd/ssh-control -p 60011 -o BatchMode=yes -o ConnectTimeout=15 root@61.241.77.34`。源节点预期`172.16.12.1`、`A3-5U-Atlas900RCKA3-1`；先验证已有control连接，不读取凭证/provider或配置密码登录。源1仅供读资产/入口，不作为本项目计算节点。

现有资源核查使用源节点上的Kubernetes hostctl容器进入对应宿主mount namespace。下表是2026-10-16资源批次的访问记录，先核当前pod→节点IP/hostname映射和权限，禁止把旧名字当实时身份。只读示例：`kubectl exec -n default POD -c hostctl -- nsenter -t 1 -m -- hostname`。缺失或权限不匹配时先只读诊断，不重建access pod或修改集群策略。

|节点|IP|预期hostname|既有pod|
|---|---|---|---|
|30|172.16.12.30|B4-15U-Atlas900RCKA3-30|sglang-node-access-30|
|31|172.16.12.31|B4-23U-Atlas900RCKA3-31|sglang-node-access-31|
|34|172.16.12.34|B5-15U-Atlas900RCKA3-34|sglang-tuning-access-34-20261013|
|35|172.16.12.35|B5-23U-Atlas900RCKA3-35|sglang-tuning-access-35-20261013|
|36|172.16.12.36|B5-33U-Atlas900RCKA3-36|sglang-tuning-access-36-20261013|
|41|172.16.12.41|B7-5U-Atlas900RCKA3-41|sglang-tuning-access-41-20261014|
|43|172.16.12.43|B7-23U-Atlas900RCKA3-43|sglang-container-access-43-20261013|
|45|172.16.12.45|B8-5U-Atlas900RCKA3-45|sglang-container-access-45-20261013|

实际宿主操作与模型/性能仍必须经本机ZCode DeepSeek、父审具体脚本SHA和精确launcher一次，见EXECUTION_CONTRACT。只管理本项目自有任务，使用独立容器/端口/锁/Meta与结果目录；不重放旧资源传输或导入launcher。Docker store `/docker_data/data`不变，权重从RESOURCES所列本地目录只读挂载；不改系统SDK/驱动/BIOS/时钟，不启用定时。
