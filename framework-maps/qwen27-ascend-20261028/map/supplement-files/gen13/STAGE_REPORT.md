Q27-FULL-MAP-20261025 / standalone-codex / generation 13 / 阶段完成待 root 验收。

交付：四模块 SHA 固定导入候选、保留旧 loader 的派生 manifest 方案、16 个 CPU 导入 fixture、原 aarch64 容器四个真实 torch CPU pool 方法投影。

主要证据：import-cpu-results.json、torch-cpu-runtime-receipt.json、import-plan.json、external-jobs-receipt.json、package-manifest.json 和 package-receipt.json。机制与限制见 Q06_IMPORT_LOADER.md。

没有 live 安装、NPU/服务/SDK 作业、调度改动或性能收益。完整模块导入/真实 TP rank 命中/输出与状态安全未验证。原 native204 与后级仍运行，未终止或释放；gen9 预览不是最终204。

下一步：原 native204 单次现场核对；未完成则准备真实路径/SHA/导入顺序固定的独立 shadow bootstrap、精确 CPU tensor 类型绑定、循环外记录导出与清理，再按非阻塞 workflow lock 做 shadow-only correctness canary。GDN/cache/lifetime/TP 和输出等价未知时保持 fallback。最终204 ready 时优先档案闭合、gen8–13 overlay、重包 SHA 与实际地图 QA。
