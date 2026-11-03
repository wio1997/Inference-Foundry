# Engram observer 独立审阅（不启动服务或 NPU）

**版本提示：下方原始问题针对 v3；v4 复审和 v3 优化模式表述修正见文末。**

范围为本目录 `sitecustomize.py`、`engram_bootstrap.py`、`audited_observer.py`、`audit_window.py`、四个 CPU 测试与 v3 receipt；四个主文件 SHA 与 README 冻结值一致。本次没有修改这四个实现文件。优先级按进入隔离诊断前的安全性排序，不代表已在 NPU 上发生。

## P1：安装失败后可能保留 wrapper，状态与实际方法不一致

`audited_observer.py:273–286` 先为类安装 wrapper，再调用 `_observer()`；若该步骤抛异常，外层 `except Exception:return False` 不恢复原方法，`engram_bootstrap.py:56–63` 会把安装视为失败。比如在类方法已替换之后 `_Observer` 初始化中的随机数获取抛错，方法已改变，但安装方得到 False、可能没有 observer status。单独在本机内存中让 `_observer()` 抛合成 `OSError` 的反例输出：`install_return=false`、`wrapper_left_installed=true`、`wrapped_original=true`、`original_return_preserved=true`。最后一项只说明此 mock 的原返回值仍正确，不代表 NPU 模型、生命周期或异常路径行为得到验证。建议调用方不要将 False 解释成“从未修改方法”；修复应保证安装是全有或全无，并测试 post-setattr 的全部异常及恢复失败语义。

## P1：关窗只是快照，不能作为“观测完整”的验收屏障

`audited_observer.py:99–111,184–190` 的 gate 检查、观察线程 checkpoint 和业务线程开始计数没有统一的跨线程门闩：业务线程第一次读取激活 gate 后可被暂停，在它执行第二次检查之前，watcher 可能看到 gate 删除、`inflight=0` 并写下关闭状态；业务线程恢复后发现 gate 已关闭，才标记 loss。若外部 `audit_window.py:37–54` 在 loss 更新之前读取快照，可能得到 `NO_LOSS_DETECTED_CONDITIONAL_SNAPSHOT`。名称和 README 已明确声明 **conditional / not proof**，因此不能把这一快照表述成整个窗口无漏记或所有调用已完整覆盖。要作为完整性门槛，需由外部控制器证明停止提交、全部生产线程跨过明确 barrier 并完成、随后关窗和读取的先后；不能只依靠 sleep 或该 poller。当前没有这样的运行时证明，且图 replay 无 Python 调用仍不能观察。

## P2：写入路径和旧文件权限未按专用根严格验证

`audited_observer.py:234–235,159–162` 只校验输入路径的字符串规范化，未检查父目录真实路径是否位于授权专用根；父目录符号链接仍会被解析。`O_NOFOLLOW` 和 hardlink 检查保护最后的文件对象，但不限制父目录路径。且 `os.open(..., 0o600)` 对已存在文件不更改其权限，也未验证 owner/mode；旧日志若过宽，匿名 shape、stride、offset、调用时序及运行标识可能暴露。`checkpoint:83–96` 若预先存在同名临时文件，异常清理会尝试 unlink 它；在受控目录与随机 process ID 下风险较低，但这依赖控制器独占目录。上线前应验证真实路径、目录所有者和权限，使用全新隔离目录并校验文件 inode/mode；否则不应解释为已证明不影响其他目录。这里未发现读取/打印 tensor 元素、提示词、权重或显式 `data_ptr()`；匿名元数据仍应作为敏感负载信息处理。

## P2：CPU PASS 的实际覆盖与运行环境缺口

`test_cpu_torch_audited.py:19–36` 使用真正的 torch CPU `[13,2,24]` 切片并检查 shape、stride 和 offset；但目标 `embed_gathered` 是原样返回输入的合成类，不执行实际 Engram lookup、NPU、真实 vLLM import、通信或 graph。`test_import_hook.py:5–12` 也是替换 TARGET 的合成模块。`test_audited.py:50–63` 覆盖一个锁丢失及状态文件数量上限，不覆盖跨进程 fork、真实并发关窗竞态、1 MiB 日志上限、设备异步生命周期。v3 stdout 和 exit code 确实记录四项 PASS，ZCode receipt 记录 deepseek-flash；没有证据由此证明 NPU 正确性或无性能扰动。

`run_stage_cpu.py:14–17` 继承环境后只清除开关，没有清除 `PYTHONOPTIMIZE` 或检查 `sys.flags.optimize`；四个测试广泛依赖 `assert`。若上游设置 `PYTHONOPTIMIZE=1/2`，检查可能被解释器移除而仍打印 PASS；当前结果文件未记录该变量或 optimize 状态，不能仅凭现有 receipt 排除。未声称当前采样实际启用了优化模式。应在受控 CPU 测试进程禁用优化并记录解释器 flags；仍需对真实 NPU 场景单独获得证据。

## 其他限制

`audited_observer.py:219–226,252–258` 在 fork 子进程重置本地状态和锁，但继承原 wrapper，状态标记为 `inherited_wrapper_unverified_after_fork`；`audit_window.py:42` 将其识别为 UNKNOWN。这是有意保守的行为，不是子进程已验证。零调用、缺失 log、缺失 worker、状态丢失、设备 graph 和外部 PID/rank 身份仍按 README 的 UNKNOWN 处理。CPU mock 未证明实际 worker 的 import hook 一定安装、线程创建不会影响调度或 NPU metadata API 不会引入额外行为。

## v4 复审（仅源码及既有 CPU 记录，无 NPU）

本地复审的四个主文件 SHA256 与本目录 README 的冻结值相符：`audited_observer.py` 为 `b87308a636201eebd97b48b605382409a05d8e661eea471bd811d819ab710d48`；bootstrap `1f6cbd25368a4ed779d2235cd1542987246185682fdd5d09a8b739b19e68afa1`；sitecustomize `eb121091a58dba6cfed99c2db542436ff07fe3e0a548bd5ca358c2ec87377342`；audit_window `c4a76a42980ef1689f92bf8ccfd227e92d55956fb2f11619a0c64822b0236c5b`。结果记录 `stage-cpu-v4-result.json` 的 SHA256 为 `3c4356139eea1000611a05a30a818f2375252c944789f6b4e686922766bd4857`。该结果文件自身不含被测源文件的散列，不能仅凭这个文件独立完成测试产物与源码的身份绑定；需另核对 169 隔离测试目录的对应文件散列。

1. **原 P1 的常见初始化和启动失败路径已修正。** `audited_observer.py:291–303` 在替换类方法前初始化，之后启动失败则在 `305–314` 尝试恢复；`test_install_failure.py:11–24` 和已有 v4 CPU 结果记录了初始化返回空、抛异常以及替换后启动异常的模拟验证。恢复操作本身异常会被吞掉，且独立替换的方法不会被覆盖，所以不是所有并发或异常路径都得到全有或全无的证明。
2. **原 P2 的路径和文件权限缺口已收窄。** `audited_observer.py:228–252,159–168` 检查目录规范真实路径、所有者、权限，并检查既有目标为同 UID、单硬链的私有普通文件；对应测试覆盖宽权限和符号链接拒绝。这里并不证明路径位于用户授权的指定根目录；对目录树被并发重命名也不是原子防护。因此仍需控制器验证授权根并持续独占目录树。`checkpoint:83–96` 遇到同名临时文件时异常清理可能 unlink 它；应使用新建、独占的诊断目录，不能在其他任务的目录里运行。
3. **更正前文对 v3 优化模式的表述。** v3 没有禁用或记录解释器优化，部分 assert 有被移除风险；但其中部分 assert 还执行初始化等必要副作用，不能断言整个 v3 四项测试在 `-O` 下仍会 PASS。v4 测试启动器清除 `PYTHONOPTIMIZE`，5 个测试各自拒绝 `sys.flags.optimize != 0`，对这项风险作了明确防护。
4. **条件快照边界依旧。** `begin:184–190` 的第一次 gate 检查和 `watch:104–111` 的关窗之间仍存在并发窗口；`audit_window:52` 的结果明确是条件快照，不能声称完整覆盖。真实 worker 的调度屏障、fork、NPU graph replay、异步存储生命周期以及对性能的扰动均未获证明，继续标记 UNKNOWN。每 worker 最多一个 watcher，64 限制针对状态文件数量而非全系统线程数。

既有五项 CPU 测试记录 PASS，其中真实 torch CPU 切片仅使用合成目标方法，不证明真实 Engram 或 NPU 行为；此次复审未重跑测试，未修改目标实现或启动设备。

**补充当前文件身份核对：**只读连接 169 自有 stage，执行 `sha256sum` 对 `engram-audited-bootstrap-cpu-v4` 中 4 个主文件和 5 个测试逐一求散列，均与本地相同文件散列相符；远端 `cpu-results.json` 散列为 `cd37c7e9eead5cc6b1c4e6daff8cb7625db33c1efe677dc997ec7b709d814385`。`run_stage_cpu.py:8–18` 在文件已存在且内容不同时拒绝覆盖后运行测试；现时文件一致结合该脚本和既有运行记录加强了对应关系，但结果本身不包含运行时文件散列，不能独立排除测试之后被替换后再恢复等情况。此次只读取当前散列，未运行测试或操作 NPU。
