# Q27 gen8 阶段结果：请求身份补证

状态：阶段完成，standalone executor 验证，待 root 验收；整体未完成，未发布、未采用优化。

本轮仅做只读源码/证据提取和本地 CPU 分析，没有启动模型、服务或 NPU 作业，也未借 SDK slot、终止外部作业或释放卡组。

## 可核验进展

- 客户端 manifest 的完整请求 ID 与 API header observer 相等；API entry step_id 与所有 SSE observer 的 api_call_id 精确关联。
- 从冻结 commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607` 的真实 AST 执行完整 `request_id` / `request_id_item` 赋值规则，结果与已观察到的 SSE response ID、frontend ID 全值相等。**API→frontend 是源码规则佐证，不冒充同记录直接观测。** 两份补取基类与该 commit 的 git blob 完全一致；另外四份安装源码与已有冻结文件完全一致。
- 32 条 `token_ready_enter` 每条同时保存 engine ID 与 frontend ID，并有32条一致的 exit；与 enqueue、已有124回放的完整 engine ID相等。该结果独立保存于 `request-identity-audit-v2.json`；原 gen6/gen7 输入不变。
- 客户端 wire 34帧（33 JSON + DONE），完整 response ID一致；CPU output_processing 32次各报告1个新 token。**不按帧/调用顺序建立逐 token因果关系。** worker-step→输出token、SSE-frame→输出token仍unknown。
- 11个CPU fixture通过，包括错误api_call_id/PID、仅前缀相似、缺失/歧义双ID、重复frontend identity、错误结束状态等拒绝路径。
- 新 `map-preview/index.html` 显式加入请求身份表与证据类型，保留原7936层区间、124个vLLM回放和历史gen64缺口；冻结 baseline/mechanisms/design语义SHA不变。Chromium file:// QA验证4身份节点、124回放/7936行、128个SG选项、15source links、8FX sidecars、41CPU shards，无JS/console错误和网络请求。桌面/移动截图已查看；该QA只适用于本派生预览，非final204。

## 证据与范围

- `request-identity-audit-v2.json`：身份、edge类型、原始observer event index、输入SHA、部署范围与限制。
- `request-evidence/download.json` / `wire-download.json`：task-private remote读取后逐文件SHA验证；未包含transport或凭据。
- `installed-base-source-commit.json`：两份新增基类精确commit-byte核对。
- `map-preview/gen8-build-receipt.json` / `browser-QA.json`：实际产物SHA、保留语义SHA与浏览器检查。
- `q27-gen8-portable-review.tar.gz` / `package-receipt.json`：allowlist审阅包，成员重新读回核对，无transport/key/原生profiles。
- `remote-observation.json` / `external-jobs-receipt.json`：当前外部作业精确PID、start_ticks、command、host boot ID、日志与终止条件。

范围沿用已冻结gen81诊断：EXPERIMENT_HOST、aarch64、910B1、vLLM/CANN9.2、TP4、bucket1、512/C1/output32，带插桩短诊断。不能外推为未插桩矩阵时间；四rank不相加为墙钟；端点不等于独占kernel或全stream关键路径；跨CANN/卡组比较不是纯框架因果。

## 失败与差异

首次读取基类时误用旧目录 `openai/engine/serving.py`，文件不存在，未作修改；依据冻结completion import定位到真实generate/base与serve/engine路径后读取成功。属于证据提取路径错误，不是技术方案失败。

本地所有权核对发现Codex当前argv hash与driver记录的启动argv hash不同；runner精确PID/start/argv匹配，Codex精确start_ticks及直接父runner匹配，worker flock仍有唯一持有者，另一路径锁未占用。差异原样保存在 `local-owner-observation.json`；不据此声称独立核实了服务端模型身份。未发现第二资源写者，也未抢锁。

## 接续

原native204 pipeline仍运行，全部下游仍等待。下一轮先核对精确现场身份并读取最新状态，不重跑基线/消融/gen81；若未204完成，可独立完成现有证据的机制/源码/优化候选覆盖审查或发布材料校验。若204就绪，核对完整input/export/audit/archive gate，再将gen8 overlay/template/guide显式应用到实际final204资产、重新打包SHA并对该实际产物做浏览器QA。worker-step/token因果边若无新显式证据保持unknown，不为追求闭合重新采集同因诊断。

质量/greedy等价、优化采用、原gen64缺口、profiler RECORD-stop警告、final204 QA、私有main/index归档与root最终接受均保留待办。发布前刷新AGENTS/README/R_TEMPLATE/分支头/leafsys1权限；未分配R、未发布。
