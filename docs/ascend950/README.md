# Ascend 950DT 部署操作手册：复制命令，直到真实模型回答成功

这份手册部署 **8×Ascend 950DT + DeepSeek-V4.1-Flash-1017 + DeepInfer + deepseek-recipe HTTP API**。固定布局为 **DP8 / EP8 / CP1**，上下文4096，关闭MTP和图捕获。

**执行位置：已经登录目标950DT服务器的 Bash 终端。** 首次部署按1→9执行，每个代码块成功后再进入下一步。代码块可以整体复制；目录、镜像和网卡会自动保存，重新登录后也能继续。

服务器应已具备950DT驱动、固件及可用Docker；账号有Docker权限，创建数据目录时有sudo权限。`/data`所在磁盘建议至少1TB可用空间，0–7八卡供本任务使用。本手册从下载开始，不安装或升级宿主驱动。

**不要把约510GB权重放到容量不足的 `/home`。** 本手册把所有大文件放在 `/data/inference-foundry-950-当前用户名/`，只在家目录保存一个很小的环境文件。

成功终点是第9步打印 **`DEPLOYMENT_VERIFIED`**，并能看到真实回答 `2` 和中文答案 `28`。启动一个容器或看到 `/healthz` 200都不算完成。

## 1. 创建数据目录，下载操作脚本，保存环境

整段复制到服务器终端执行：

```bash
bash <<'BASH'
set -Eeuo pipefail
for command in docker curl tar python3 ip; do
  command -v "$command" >/dev/null || { echo "缺少服务器基础命令：$command"; exit 1; }
done
[ "$(uname -m)" = aarch64 ] || { echo '本手册需要ARM64服务器'; exit 1; }
python3 -c 'import sys; assert sys.version_info >= (3, 8), "宿主Python需要3.8或更新版本"'
docker info >/dev/null

DI950_DATA="/data/inference-foundry-950-$(id -un)"
if [ ! -d "$DI950_DATA" ]; then
  if [ "$(id -u)" = 0 ]; then
    install -d -m 0755 "$DI950_DATA"
  else
    sudo install -d -m 0755 -o "$(id -u)" -g "$(id -g)" "$DI950_DATA"
  fi
fi
[ -w "$DI950_DATA" ] || { echo "数据目录不可写：$DI950_DATA"; exit 1; }
export DEPLOY_ROOT="$DI950_DATA/work"
export MODEL_DIR="$DI950_DATA/models/DeepSeek-V4.1-Flash-1017"
mkdir -p "$DEPLOY_ROOT" "$MODEL_DIR"

DI950_UNPACK=$(mktemp -d "$DI950_DATA/.bundle.XXXXXX")
trap 'rm -rf -- "$DI950_UNPACK"' EXIT
curl -4 -fL --retry 5 --connect-timeout 20 \
  https://codeload.github.com/wio1997/Inference-Foundry/tar.gz/refs/heads/950 \
  -o "$DI950_UNPACK/950.tar.gz"
mkdir -p "$DI950_UNPACK/source"
tar -xzf "$DI950_UNPACK/950.tar.gz" --strip-components=1 -C "$DI950_UNPACK/source"
cp -a "$DI950_UNPACK/source/deploy/ascend950/." "$DEPLOY_ROOT/"
mkdir -p "$DEPLOY_ROOT/logs" "$DEPLOY_ROOT/runtime-20261021"

# 已有本任务模型容器时，只读复用其权重目录；第3步仍做完整SHA校验。
DI950_EXISTING_MODEL=$(python3 - <<'PYCODE'
import json, pathlib, subprocess
for name in ['deepinfer-950-engine','deepinfer-engine-20261020']:
    p=subprocess.run(['docker','inspect',name],capture_output=True)
    if p.returncode:
        continue
    for mount in json.loads(p.stdout)[0].get('Mounts',[]):
        src=pathlib.Path(mount['Source'])
        if str(mount['Destination']).endswith('/DeepSeek-V4.1-Flash-1017') and (src/'config.json').is_file() and (src/'model.safetensors.index.json').is_file():
            print(src.resolve())
            raise SystemExit(0)
PYCODE
)
if [ -n "$DI950_EXISTING_MODEL" ]; then
  export MODEL_DIR="$DI950_EXISTING_MODEL"
  printf '[INFO] 复用已下载模型：%s\n' "$MODEL_DIR"
fi

HCCL_SOCKET_IFNAME=$(ip -o -4 route get 1.1.1.1 | awk '{for(i=1;i<=NF;i++) if($i=="dev") {print $(i+1); exit}}')
[ -n "$HCCL_SOCKET_IFNAME" ] && [ -d "/sys/class/net/$HCCL_SOCKET_IFNAME" ] || { echo '无法确定IPv4出口网卡，请先恢复服务器网络'; exit 1; }
export HCCL_SOCKET_IFNAME
export BASE_IMAGE=quay.io/ascend/vllm-ascend@sha256:10f4e41479604545b74f3684e1a98e9e7116eff6adc72ecaec765d2dbecc7e81
export DEEPINFER_IMAGE=local/deepinfer:950dt-rebuild-20261021
{
  printf 'export DEPLOY_ROOT=%q\n' "$DEPLOY_ROOT"
  printf 'export MODEL_DIR=%q\n' "$MODEL_DIR"
  printf 'export BASE_IMAGE=%q\n' "$BASE_IMAGE"
  printf 'export DEEPINFER_IMAGE=%q\n' "$DEEPINFER_IMAGE"
  printf 'export HCCL_SOCKET_IFNAME=%q\n' "$HCCL_SOCKET_IFNAME"
} > "$DEPLOY_ROOT/deploy.env"
cp "$DEPLOY_ROOT/deploy.env" "$HOME/ascend950.env"
printf '[PASS] 工作目录：%s\n[PASS] 模型目录：%s\n[PASS] 通信网卡：%s\n' "$DEPLOY_ROOT" "$MODEL_DIR" "$HCCL_SOCKET_IFNAME"
BASH
```

**成功标志：** 最后出现三行 `[PASS]`，显示工作目录、模型目录和网卡。若sudo提示密码，输入当前服务器账号的sudo密码；不需要把密码写入脚本。

后面的代码块都会读取 `$HOME/ascend950.env`，不需要再次填写路径。

## 2. 检查磁盘、设备和主机环境，拉取基础镜像

```bash
bash <<'BASH'
set -Eeuo pipefail
source "$HOME/ascend950.env"
cd "$DEPLOY_ROOT"
python3 - "$MODEL_DIR" "$DEPLOY_ROOT/locks/model-manifest.json" <<'PYCODE'
import json, pathlib, shutil, sys
model = pathlib.Path(sys.argv[1])
manifest = json.load(open(sys.argv[2]))
already = sum((model/x['path']).stat().st_size for x in manifest['files']
              if (model/x['path']).is_file() and (model/x['path']).stat().st_size == x['bytes'])
needed = max(0, manifest['total_bytes'] - already) + 30*1024**3
free = shutil.disk_usage(model).free
assert free >= needed, '数据盘空间不足：剩余%.1fGiB，至少还需%.1fGiB' % (free/1024**3, needed/1024**3)
print('[PASS] 磁盘空间：可用 %.1f GiB' % (free/1024**3))
for n in range(8):
    assert pathlib.Path('/dev/davinci'+str(n)).exists(), '缺少设备节点 /dev/davinci'+str(n)
for name in ['/usr/local/Ascend/driver','/etc/hccl_rootinfo.json','/lib/route.conf']:
    assert pathlib.Path(name).exists(), '缺少交付环境文件：'+name
print('[PASS] 八卡设备与驱动/拓扑文件存在')
PYCODE
lscpu | head -25
free -h
cat /usr/local/Ascend/driver/version.info

docker pull "$BASE_IMAGE"
[ "$(docker image inspect "$BASE_IMAGE" --format '{{.Architecture}}')" = arm64 ]
printf '[PASS] 基础镜像已就绪\n'
BASH
```

**成功标志：** 显示八卡文件存在，镜像拉取完成，最后打印 `[PASS] 基础镜像已就绪`。当前参考环境是driver25.6.rc1/firmware9.0.0.200.200/CANN9.2；不同驱动环境先按供应方兼容要求处理，勿直接套用本次成功结论。

如果这里设备或拓扑文件缺失，应先修复服务器交付环境；后续安装Python包不能补出这些硬件能力。

## 3. 下载模型，并校验全部84个文件

```bash
bash <<'BASH'
set -Eeuo pipefail
source "$HOME/ascend950.env"
cd "$DEPLOY_ROOT"
python3 -u download_model.py --dest "$MODEL_DIR" --workers 8 \
  2>&1 | tee "$DEPLOY_ROOT/logs/03-model-download.log"
python3 - "$DEPLOY_ROOT/model-verification.json" <<'PYCODE'
import json, sys
d = json.load(open(sys.argv[1]))
assert d['phase']=='complete' and d['file_count']==84
assert d['sha256_verified_files']==84 and d['weight_shards']==48
print('[PASS] 模型：84文件、48分片、84/84 SHA256通过')
PYCODE
BASH
```

**成功标志：** 最后打印模型校验通过。模型大小510,311,045,870字节，下载和校验可能持续数小时；终端仍有curl进程或文件持续增长时不要误判为卡死。

下载中断后**原样重跑本步骤**：已正确文件会复用，`.partial`文件继续下载。坏文件处理命令在后面的“常见故障恢复”中。

此步骤按每个文件固定的ModelScope revision下载，校验清单随部署包提供。无需配置Hugging Face token，也不需要手工拼接48个分片。

## 4. 下载固定源码和Python依赖

```bash
bash <<'BASH'
set -Eeuo pipefail
source "$HOME/ascend950.env"
cd "$DEPLOY_ROOT"
python3 -u fetch_sources.py 2>&1 | tee "$DEPLOY_ROOT/logs/04-sources.log"
python3 -u download_wheels.py 2>&1 | tee "$DEPLOY_ROOT/logs/04-wheels.log"
printf '[PASS] 固定源码和带SHA校验的wheelhouse已就绪\n'
BASH
```

**成功标志：** 源码下载成功或显示 `PINNED_SOURCES_ALREADY_PRESENT`，wheel下载显示 `WHEELS_VERIFIED 75`，最后打印 `[PASS]`。

中断后原样重跑；已经下载完整的固定源码会保留，**不会覆盖已打的补丁**。源码归档包含必要的vendor与嵌套依赖，不需要再执行其他递归clone命令。

## 5. 自动修复已知兼容问题，编译并生成运行镜像

```bash
bash <<'BASH'
set -Eeuo pipefail
source "$HOME/ascend950.env"
cd "$DEPLOY_ROOT"
bash build_image.sh 2>&1 | tee "$DEPLOY_ROOT/logs/05-build.log"
python3 apply_patches.py --check-only
python3 - "$DEPLOY_ROOT/image-lock.json" <<'PYCODE'
import json, sys
d = json.load(open(sys.argv[1]))
print('[PASS] 运行镜像：'+d['image'])
print('[PASS] image ID：'+d['image_id'])
PYCODE
BASH
```

**成功标志：** 出现 `BUILD_IMAGE_READY`、`PATCH_SET_VERIFIED` 和运行镜像ID。

该步骤自动完成以下操作，**无需手工编辑源码**：

- 在不传NPU设备的专用构建容器中安装编译依赖。
- GCC13编译主机代码，Bisheng配GCC11头文件编译设备代码。
- 应用DeepSelect构建平台、FlashMLA RINT、Engram直接存储、本机UBMEM、NPU from_blob、DeepGEMM断言头文件和短上下文TopK七个补丁。
- 创建独立engine-venv与recipe-venv，编译四个算子组件及DeepInfer主机扩展。
- 确认构建容器退出码0，提交本地运行镜像并记录image-lock。

**编译中断：原样重跑本步骤。** 脚本会校验并复用同一个构建容器。若此前已成功构建且镜像/源码锁一致，直接复用结果。

`local/deepinfer:950dt-rebuild-20261021`由这一步生成，不需要、也不能从公共仓库直接拉取这个local标签。

## 6. 生成本机八卡通信配置

```bash
bash <<'BASH'
set -Eeuo pipefail
source "$HOME/ascend950.env"
cd "$DEPLOY_ROOT"
bash runtime.sh --topology 2>&1 | tee "$DEPLOY_ROOT/logs/06-topology.log"
python3 - "$DEPLOY_ROOT/runtime-20261021/topology" <<'PYCODE'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
d=json.loads((p/'hccl_rootinfo-local-only.json').read_text())
assert d['rank_count']==8
assert {int(x['device_id']) for x in d['rank_list']}==set(range(8))
assert all(x['level_list'] and all(y['net_layer']==0 for y in x['level_list']) for x in d['rank_list'])
assert all((p/'hixlep'/('ub_endpoint_npu_'+str(n)+'.json')).is_file() for n in range(8))
print('[PASS] 本机八卡layer0拓扑及8个endpoint文件已生成')
PYCODE
BASH
```

**成功标志：** 最后打印本机八卡layer0拓扑检查通过。文件只生成在本任务目录，运行时只读挂载进容器；不覆盖宿主全局拓扑。

本次HCCL超时和大尺寸AllGather问题已经通过任务内layer0配置及源码补丁处理。换服务器时执行本步骤重新查询，不能把旧机器的endpoint地址复制过来。

## 7. 在空闲八卡上运行算子正确性检查

```bash
bash <<'BASH'
set -Eeuo pipefail
source "$HOME/ascend950.env"
cd "$DEPLOY_ROOT"
bash runtime.sh --operator-checks 2>&1 | tee "$DEPLOY_ROOT/logs/07-operator-checks.log"
python3 - "$DEPLOY_ROOT" <<'PYCODE'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
for name,count in [('flashmla-correctness.json',4),('dg-layout-correctness.json',3),('candidate-short-context-correctness.json',5)]:
    cases=json.loads((p/'runtime-20261021'/name).read_text())
    assert len(cases)==count and all(x['passed'] for x in cases), name
lines=(p/'logs/07-operator-checks.log').read_text().splitlines()
for fp8 in ['True','False']:
    selected=[x for x in lines if 'PASS: rank ' in x and 'FP8='+fp8 in x]
    assert len(selected)==8, 'Engram八rank检查不完整，FP8='+fp8
print('[PASS] FlashMLA 4组、DeepGEMM 3组、TopK 5组、Engram 16条rank检查全部通过')
PYCODE
BASH
```

**成功标志：** 最后打印全部检查通过。脚本会先确认0–7八卡空闲；被其他任务占用时拒绝运行，不会自动清理别人的容器。

首次JIT编译可能较慢。此步骤包含真实NPU运算，检查成功之后再启动完整模型。

## 8. 启动八卡模型和CPU API，自动等待就绪

```bash
bash <<'BASH'
set -Eeuo pipefail
source "$HOME/ascend950.env"
cd "$DEPLOY_ROOT"
bash runtime.sh --start
bash runtime.sh --start-api
python3 -u wait_ready.py --timeout 3600 2>&1 | tee "$DEPLOY_ROOT/logs/08-ready.log"
curl --fail-with-body -sS http://127.0.0.1:18110/readyz
printf '\n[PASS] 八卡真实引擎已就绪\n'
BASH
```

**成功标志：** 出现 `ENGINE_READY`，返回JSON中 `ready=true`、`ranks=8`。

两个容器名称固定为：

```text
deepinfer-950-engine    八卡模型，后端18111
deepinfer-950-api       CPU协议API，端口18110
```

等待期间打印 `MODEL_LOADING` 属于正常状态。容器退出或超时会停止等待并返回错误，不会误报成功。

## 9. 发起真实中文请求，完成部署验收

```bash
bash <<'BASH'
set -Eeuo pipefail
source "$HOME/ascend950.env"
cd "$DEPLOY_ROOT"
python3 -u verify_deployment.py 2>&1 | tee "$DEPLOY_ROOT/logs/09-verification.log"
BASH
```

脚本会真实请求模型：

1. “只回答数字：1+1等于几？”——必须返回 `2`。
2. “小明有3盒铅笔，每盒12支，送出8支后还剩多少支？”——必须包含答案 `28`。

还会检查API确实使用真实后端、八rank输出token哈希一致、API计数与实际生成token一致。**最后必须出现：**

```text
DEPLOYMENT_VERIFIED
```

本次新部署的验收结果保存在工作目录 `runtime-20261021/verification-latest.json`；原始token及八rank证据保存在同目录的 `generations/`。

首次请求可能继续触发编译，请等待。历史功能部署首个简单请求曾耗时约376秒；这不是稳态性能。此时无需再手工处理EOS：API已经包含末尾EOS修复。

## 部署完成后：直接调用API

在服务器上复制执行：

```bash
curl --fail-with-body -sS http://127.0.0.1:18110/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":"请用中文介绍一下你自己。"}],"thinking":{"type":"disabled"},"max_tokens":128,"temperature":0,"stream":false}'
```

当前接口为文本、非流式、单请求功能入口。它把同一请求发送到八个DP rank，不用于独立并发吞吐排名。API只监听服务器loopback。

## 每天用到的运维命令

### 查看是否就绪

```bash
bash <<'BASH'
set -Eeuo pipefail
source "$HOME/ascend950.env"
cd "$DEPLOY_ROOT"
bash runtime.sh --check
BASH
```

### 查看模型/API日志

```bash
docker logs --tail 200 deepinfer-950-engine
docker logs --tail 100 deepinfer-950-api
```

持续跟踪模型日志：

```bash
docker logs --follow --tail 100 deepinfer-950-engine
```

按 `Ctrl+C` 只结束日志查看，不停止容器。

### 停止、恢复和重启本任务

先停止API，再停止模型；下列代码只处理具有本任务标签且挂载本工作目录的容器：

```bash
bash <<'BASH'
set -Eeuo pipefail
source "$HOME/ascend950.env"
python3 - "$DEPLOY_ROOT" <<'PYCODE'
import json, subprocess, sys
for name in ['deepinfer-950-api','deepinfer-950-engine']:
    r=subprocess.run(['docker','inspect',name],capture_output=True)
    if r.returncode:
        continue
    d=json.loads(r.stdout)[0]
    assert (d['Config'].get('Labels') or {}).get('inference-foundry.scope')=='950-deploy', '容器不属于本任务：'+name
    assert any(x['Source']==sys.argv[1] and x['Destination']=='/work' for x in d['Mounts']), '工作目录不匹配：'+name
    if d['State']['Running']:
        subprocess.run(['docker','stop',name],check=True)
print('[PASS] 本任务API与模型已停止')
PYCODE
BASH
```

停止后恢复，或服务器重启后重新启动：

```bash
bash <<'BASH'
set -Eeuo pipefail
source "$HOME/ascend950.env"
cd "$DEPLOY_ROOT"
bash runtime.sh --start
bash runtime.sh --start-api
python3 -u wait_ready.py --timeout 3600
python3 -u verify_deployment.py
BASH
```

不需要重新下载、重新编译或删除缓存。

## 常见故障：直接复制对应恢复命令

### 下载中断，或文件SHA校验不一致

下面命令保留损坏文件为 `.bad-时间戳` 后重下；正确文件继续复用，不删除原权重。保留坏文件会额外占用磁盘。

```bash
bash <<'BASH'
set -Eeuo pipefail
source "$HOME/ascend950.env"
cd "$DEPLOY_ROOT"
python3 -u download_model.py --dest "$MODEL_DIR" --workers 8 --repair \
  2>&1 | tee "$DEPLOY_ROOT/logs/03-model-repair.log"
BASH
```

完成后继续步骤4；若已完成步骤4及后续构建，可直接继续步骤8。

### 编译失败或SSH在构建期间断开

不删除构建容器，直接重新执行：

```bash
bash <<'BASH'
set -Eeuo pipefail
source "$HOME/ascend950.env"
cd "$DEPLOY_ROOT"
bash build_image.sh 2>&1 | tee "$DEPLOY_ROOT/logs/05-build-retry.log"
BASH
```

脚本会复用或重新启动 `deepinfer-950-build`，等待退出并检查退出码。看到 `BUILD_IMAGE_READY` 后继续步骤6。

### readiness长时间503，或模型容器退出

先复制下面命令，日志末尾会给出具体失败位置：

```bash
bash <<'BASH'
source "$HOME/ascend950.env"
docker ps -a --filter label=inference-foundry.scope=950-deploy
docker logs --tail 200 deepinfer-950-engine
docker logs --tail 100 deepinfer-950-api
cat "$DEPLOY_ROOT/runtime-20261021/npu-preflight.txt" 2>/dev/null || true
BASH
```

如果是八卡被占用，需要先按资源安排释放设备，再执行上面的“恢复”代码块。不要用停止所有Docker容器、清空所有NPU进程的命令。

如果出现 `unknown format type`、`ascendc_assert` 未声明、旧RINT入口、Engram初始化OOM或短上下文TopK断言，先确认没有跳过步骤5，并复制下面命令检查补丁状态：

```bash
bash <<'BASH'
set -Eeuo pipefail
source "$HOME/ascend950.env"
cd "$DEPLOY_ROOT"
python3 apply_patches.py --check-only
cat image-lock.json
cat build-state.txt
BASH
```

七个修复在本手册的编译流程中已经自动应用。不要在修过源码后仍运行另一目录里的旧venv或另一镜像。

### 需要强制重新编译并重建运行容器

以下代码校验并停止/删除本任务的两个运行容器，保留模型、源码、venv、缓存和构建容器；然后完成重建、算子检查和真实生成验收。它不删除任何卷，也不处理其他任务容器。

```bash
bash <<'BASH'
set -Eeuo pipefail
source "$HOME/ascend950.env"
cd "$DEPLOY_ROOT"
python3 - "$DEPLOY_ROOT" <<'PYCODE'
import json, subprocess, sys
names=['deepinfer-950-api','deepinfer-950-engine']
owned=[]
for name in names:
    p=subprocess.run(['docker','inspect',name],capture_output=True)
    if p.returncode:
        continue
    d=json.loads(p.stdout)[0]
    assert (d['Config'].get('Labels') or {}).get('inference-foundry.scope')=='950-deploy', '容器不属于本任务：'+name
    assert any(x['Source']==sys.argv[1] and x['Destination']=='/work' for x in d['Mounts']), '工作目录不匹配：'+name
    owned.append((name,d['State']['Running']))
for name,running in owned:
    if running:
        subprocess.run(['docker','stop',name],check=True)
    subprocess.run(['docker','rm',name],check=True)
PYCODE
bash build_image.sh --force 2>&1 | tee "$DEPLOY_ROOT/logs/rebuild.log"
bash runtime.sh --topology
bash runtime.sh --operator-checks
bash runtime.sh --start
bash runtime.sh --start-api
python3 -u wait_ready.py --timeout 3600
python3 -u verify_deployment.py
BASH
```

### 校验文件丢失，但模型已经完整下载

此命令只读取并校验模型，不重新下载，也不需要修改模型目录：

```bash
bash <<'BASH'
set -Eeuo pipefail
source "$HOME/ascend950.env"
cd "$DEPLOY_ROOT"
python3 -u download_model.py --dest "$MODEL_DIR" --verify-only
BASH
```

## 备份：镜像和工作目录一起保存

先按上面的停止代码块停止本任务，再执行：

```bash
bash <<'BASH'
set -Eeuo pipefail
source "$HOME/ascend950.env"
DI950_BACKUP="$(dirname "$DEPLOY_ROOT")/backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$DI950_BACKUP"
docker save "$DEEPINFER_IMAGE" | gzip > "$DI950_BACKUP/runtime-image.tar.gz"
tar -C "$(dirname "$DEPLOY_ROOT")" -czf "$DI950_BACKUP/work.tar.gz" "$(basename "$DEPLOY_ROOT")"
cp "$HOME/ascend950.env" "$DI950_BACKUP/ascend950.env"
printf '[PASS] 备份完成：%s\n[提示] 约510GB权重保留在：%s\n' "$DI950_BACKUP" "$MODEL_DIR"
BASH
```

这份命令备份运行镜像和完整工作目录；权重单独保留。迁移到另一台机器还需传输权重、确认驱动兼容，并重新生成那台机器的拓扑。

## 已自动包含的修复与参考资料

操作主线到此结束。下面资料用于理解问题，不是部署时还需要手动完成的额外步骤：

- [实际报错、原因、补丁与验收依据](TROUBLESHOOTING.md)
- [原始部署记录与版本说明](DEPLOYMENT_RECORD.md)
- [鲲鹏950与上一代的差异](KUNPENG950.md)
- [操作脚本与固定版本清单](../../deploy/ascend950/)
- [历史实卡证据](../../deploy/ascend950/evidence/validated-deployment.json)

历史固定配置已真实完成八卡生成和代表性算子正确性检查。本次整理新增的操作入口做了脚本语法、恢复分支、补丁重放与离线数据核验；未在另一台空服务器上重新执行全部下载与NPU部署。**每次实际交付，以本手册第9步在目标机产生的新验收结果为准。**
