# PR #12 与固定 0.30 镜像的源码对照

参考用户提供的 [PR #12](https://github.com/wio1997/Inference-Performance-Results/pull/12)，head 为 `eac303abd3eac260c1c4e6eaff4641c2986f13cf`，merge 为 `18489df47f217e3996e87b6f406c7deb27ecd572`。参考归档包含官方内置适配源码、注册入口及来源信息；它不是独立可安装的增量补丁。参考 vLLM-Ascend 上游 commit 为 `24487f4f48ee99cf45d86d1c504e645549b8c1ef`。

已取回并校验参考归档 113 个源文件的 SHA256 / Git blob，以及归档原 P/D 对照哈希。这是归档完整性检查。参考采用 0.32.0rc1、910B3、CANN 9.2.1、16K/C4/eager draft；本任务仍采用固定 0.30.0rc2 镜像、910B1、四档精确输入/C16/1024和完整 target/draft 图。

对固定镜像实际取出的源码逐路径比较，113 个文件中有 39 个字节一致、44 个有差异、30 个在相同相对路径下不存在；其中 6 个在迁移后的同名路径中字节一致。路径不存在不能推断功能缺失：固定镜像将部分 V2 补丁置于 `patch/worker/patch_v2/`，参考版本的布局不同。

当前固定版本已注册 same-checkpoint DSPARK 继承 target quant 的原生包装。G372 在 CPU 执行两版实际包装函数：PP1 下同 checkpoint/独立 draft × loader 正常/抛错，共 8 用例，均选择预期 quant 对象并在 finally 恢复原 hooks。这里用接口夹具隔离装载器；没有实际解析 ModelSlim 描述、载入权重、执行 NPU、PP>1、图、PD、质量或性能验证。本机制无需新增适配器。

参考中新增加的 bounded SWA replay、cache planning 和共享 draft WKV 投影融合值得进一步审查。两版依赖的 cache spec、planner 路径和 API 有差异；这些差异尚未完成回迁验证。没有将整份源码快照覆盖运行包。

- [113 文件逐项比较、SHA、迁移候选及范围](../evidence/PR12_source_comparison_G370.json)
- [8 个 CPU 原生包装控制用例](../evidence/PR12_quant_contract_G372.json)
- [固定镜像实际包装源文件](../code/pr12_quant_contract/fixed/patch_dspark.py)
- [PR 固定 head 的包装源文件](../code/pr12_quant_contract/reference/patch_dspark.py)
- [CPU 复现脚本](../code/pr12_quant_contract/verify.py)

```bash
python3 code/pr12_quant_contract/verify.py
```

输出 `code/pr12_quant_contract/result.local.json`，仅表示 CPU 控制分支检查。
