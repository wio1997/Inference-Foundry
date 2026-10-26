# 本项目现有访问入口

Mac到资源源节点1：`ssh -S /Users/wio/work/sglang-glm53-pd/ssh-control -p 60011 -o BatchMode=yes -o ConnectTimeout=15 root@61.241.77.34`。源节点预期`172.16.12.1`、`A3-5U-Atlas900RCKA3-1`；先验证已有control连接，不读取凭证/provider或配置密码登录。源1仅供读资产/入口，不作为本项目计算节点。

现有资源核查使用源节点上的Kubernetes hostctl容器进入对应宿主mount namespace。下表是2026-10-16资源批次的访问记录，先核当前pod→节点IP/hostname映射和权限，禁止把旧名字当实时身份。只读示例：`kubectl exec -n default POD -c hostctl -- nsenter -t 1 -m -- hostname`。缺失或权限不匹配时先只读诊断，不重建access pod或修改集群策略。

|节点|IP|预期hostname|既有pod|
|---|---|---|---|
|14|172.16.12.14|A6-15U-Atlas900RCKA3-14|zhixing-dockerctl-25xz2|
|15|172.16.12.15|A6-23U-Atlas900RCKA3-15|zhixing-dockerctl-42pbf|
|46|172.16.12.46|B8-15U-Atlas900RCKA3-46|sglang-container-access-46-20261013|
|47|172.16.12.47|B8-23U-Atlas900RCKA3-47|sglang-container-access-47-20261013|

实际宿主操作与模型/性能仍必须经本机ZCode DeepSeek、父审具体脚本SHA和精确launcher一次，见EXECUTION_CONTRACT。只管理本项目自有任务，使用独立容器/端口/锁/Meta与结果目录；不重放旧资源传输或导入launcher。Docker store `/docker_data/data`不变，权重从RESOURCES所列本地目录只读挂载；不改系统SDK/驱动/BIOS/时钟，不启用定时。
