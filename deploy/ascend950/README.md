# 950DT 部署包

从[完整部署手册](../../docs/ascend950/README.md)开始。把本目录复制到目标机器独立的 `DEPLOY_ROOT`；所有容器将该目录挂到 `/work`。

| 文件 | 用途 |
|---|---|
| `download_model.py` | 按锁定文件revision下载84文件并校验SHA256；支持断点续传、只校验和保留坏文件后修复 |
| `download_wheels.py` | 获取带SHA锁的ARM64依赖和recipe wheel |
| `fetch_sources.py` | 固定主仓库、vendor和嵌套依赖源码 |
| `apply_patches.py` | 按依赖顺序打补丁并记录最终源文件哈希 |
| `configure_compilers.py` | 专用容器内分流主机与Ascend C编译器 |
| `bootstrap.sh` | 构建engine-venv、算子、DeepInfer及recipe-venv |
| `build_image.sh` | 按deploy.env构建、续跑、检查退出码及保存镜像锁 |
| `generate_topology.py` | 只读DCMI查询，派生本任务rootinfo与endpoint |
| `runtime.sh` | 拓扑、算子检查、模型启动、CPU API及ready检查 |
| `engine_bridge.py` | 原始验证后的DP8/EP8/CP1单请求功能桥接器 |
| `api.py` | 官方recipe协议适配与真实RPC、EOS/ready处理 |
| `wait_ready.py` | 等待真实八卡ready；容器失败或超时不返回成功 |
| `verify_deployment.py` | 实际发送两条请求，核对答案、token计数及八rank证据 |
| `checks/` | 随历史实卡验证保留的代表性正确性检查 |
| `locks/` | 模型、源码和Python依赖的固定版本/校验清单 |
| `patches/` | 七个构建/运行兼容补丁 |
| `evidence/` | 2026-10-21的脱敏历史验收结果与发布检查记录 |

操作时按完整手册的1→9依次复制代码块；环境会保存到家目录的ascend950.env及工作目录的deploy.env。大文件默认放在/data下的本任务目录。模型校验结果位于工作目录model-verification.json，允许只读复用权重目录。

验证层次：历史原始部署已做实卡生成；整理后的脚本做语法/补丁重放/拓扑离线核对及控制流程检查。完整的新机重建仍需在目标机执行算子和生成验收，不能直接把历史结果复制成新结果。

`api.py`、`engine_bridge.py` 已包含最终EOS和观测修复，不需要再次应用历史API/bridge差分补丁。七个补丁只作用于下载的源码。运行目录、venv、native扩展及缓存与镜像一起保存。

## 来源与许可

本包引用固定版本的 DeepInfer、DeepSelect、FlashMLA、DeepEP-Ascend、DeepGEMM-Ascend 和 deepseek-recipe。源码由下载脚本从官方项目获取，其各自LICENSE仍适用；patch不会替代原项目许可。

FlashMLA检查调用原项目数值参考函数；Engram检查基于DeepEP-Ascend的测试逻辑做直接storage/fetch验证。具体提交见 `locks/source-pins.json`。本包不包含CANN、驱动、wheel二进制、模型权重或私有设备拓扑；这些按对应项目和供应方条件单独获取。
