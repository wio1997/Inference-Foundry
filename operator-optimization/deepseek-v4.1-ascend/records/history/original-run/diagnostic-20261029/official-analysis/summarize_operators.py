#!/usr/bin/env python3
"""Summarize a completed per-rank official run without merging rank clocks."""
import json, math, pathlib, re, statistics

p=pathlib.Path(__file__).resolve().parent
x=json.loads((p/"operators.json").read_text())
rows=x["rows"]
def med(xs): return statistics.median(xs)
def fmt(v): return f"{v:,.3f}"
out=["# 官方采集的算子与通信轨迹初析", "",
     f"Run: `{x['run_id']}`。所选文件共 {x['export_source_count']}，成功 {x['parsed_count']}，rank 数 {x['rank_count']}；失败 {len(x['failures'])}。每 rank 的每次记录单独计时，按该 rank 的 profiler 开始时间排序为窗口 1 和窗口 2。CSV 原始 SHA-256、每卡各类型记录数、时间和/并集、模型 ID 分布及 step_trace 行见 [operators.json](operators.json)。", "",
     "口径：统计全部正时长 CSV 记录，不选定旧 model ID；重复、跨流或 MIX 复合记录未作物理去重。每个类型分别取区间并集及时长和；所报告的 16 卡中位数是卡内指标再取中位数，不是跨卡汇总或端到端时长。分母为各卡已导出 kernel 最早开始至最晚结束，非整个请求、整个 profiler 捕获区间。", ""]
for window in (1,2):
    a=[r for r in rows if r["window_ordinal"]==window]
    if not a: continue
    span=[r["kernel_observed_span_ns"]/1000 for r in a]
    capture=[(r["record_end_monotonic_ns"]-r["record_start_monotonic_ns"])/1e6 for r in a]
    out += [f"## 窗口 {window}：{len(a)} 个 rank", "",
        f"导出 kernel 范围中位 {fmt(med(span))} µs（最小 {fmt(min(span))}，最大 {fmt(max(span))}）；profiler 元数据开始至结束中位 {fmt(med(capture))} ms（最小 {fmt(min(capture))}，最大 {fmt(max(capture))}）。", "",
        "| CSV 类型 | 每卡记录数中位 | 每卡时长和中位 µs | 每卡区间并集中位 µs | 并集/各卡导出范围中位 |",
        "|---|---:|---:|---:|---:|"]
    types=set().union(*(r["types"] for r in a))
    ordered=sorted(types,key=lambda t:med([r["types"].get(t,{}).get("union_ns",0) for r in a]),reverse=True)
    for t in ordered[:12]:
        q=[r["types"].get(t,{}) for r in a]
        count=med([z.get("count",0) for z in q])
        s=med([z.get("sum_ns",0)/1000 for z in q])
        u=med([z.get("union_ns",0)/1000 for z in q])
        fraction=med([z.get("union_ns",0)/r["kernel_observed_span_ns"] for z,r in zip(q,a)])
        out.append(f"| `{t}` | {count:g} | {fmt(s)} | {fmt(u)} | {fraction*100:.2f}% |")
    cc=med([r["compute_label_union_ns"]/1000 for r in a])
    comm=med([r["communication_label_union_ns"]/1000 for r in a])
    cross=med([r["compute_comm_intersection_ns"]/1000 for r in a])
    out += ["",f"按 CSV `Accelerator Core` 标签统计，计算类（AI_CORE、AI_VECTOR_CORE、MIX_AIC、MIX_AIV）并集中位 {fmt(cc)} µs；显式 COMMUNICATION 类并集中位 {fmt(comm)} µs；两类区间相交中位 {fmt(cross)} µs。这些是导出记录区间，不代表有效通信被隐藏，也不证明空白区间设备空闲；MC2 的 MIX 任务内通信可能没有单独 COMMUNICATION 记录。不同指标的中位数不能直接相加。",""]
out += ["## 范围与复现", "",
        "客户端的初始窗口覆盖首次非空文本出现的时期；后窗口标为 later_decode_candidate，客户端上它位于所有请求首次非空文本之后、结束之前。该标签只表示客户端观测时段，不证明每条设备任务的 prefill/decode 阶段、全请求对应或精确设备切换时间。需要先核对客户端与 profiler 的单调时钟域，再结合模型调用和 shape 判定；trace 存在和导出成功不等于完整捕获。", "",
        "形状佐证：按各卡和窗口，从所选 CSV 类型的 Input Shapes 读取首个输入的物理首维，统计每卡首维大于 3 的记录。数值是记录数，不是请求数或有效 token 数：", "",
        "| CSV 类型 | 窗口 1：16 卡每卡大于 3 的数量范围 | 窗口 2：16 卡每卡大于 3 的数量范围 |", "|---|---:|---:|"]
for typ in ("DispatchFFNCombine","SparseFlashMla","QuantBatchMatmulV3"):
    groups=[]
    for window in (1,2):
        a=[r for r in rows if r["window_ordinal"]==window]
        values=[]
        for r in a:
            count=0
            for shape,n in r["selected_shape_counts"].get(typ,{}).items():
                match=re.match(r'"?(\d+),',shape)
                if match and int(match.group(1))>3:count+=n
            values.append(count)
        groups.append(f"{min(values)}–{max(values)}" if values else "UNKNOWN")
    out.append(f"| `{typ}` | {groups[0]} | {groups[1]} |")
out += ["", "第 1 窗既有多行任务，也有首维不大于 3 的记录；第 2 窗这三个类型未见首维大于 3 的记录。配合客户端时序，这与第 1 窗含多行预填充/混合工作、第 2 窗偏后续 decode 相容，但单凭 shape 无法证明实际阶段、同一请求或全部任务均为 decode。", "",
        "默认记录包含多个 Model ID、多条 stream 和类别，算子 Type 不一定等于模型模块，不能据此给出真实端到端占比或根因。算子 duration 和、MIX 计数、不同 rank 时间、step_trace 字段及共享资源活动不得混加；没有有效专家行数、真实 HBM bytes 与通信依赖时不能给出它们的性能边界或优化收益。", "",
        "离线复现：先在指定 stage 容器中用 [export_one.py](export_one.py) 将每个原始 profiler 根复制到本 run 的 analysis 目录并以官方 `torch_npu.profiler.profiler.analyse(..., max_process_number=1, export_type='text')` 导出；本机 [run_export_batch.py](run_export_batch.py) 限制并发 2，原始采集不变。然后 [run_analyze_remote.py](run_analyze_remote.py) 调用 [analyze_remote.py](analyze_remote.py) 从导出 CSV 生成 [operators.json](operators.json)，运行本脚本生成报告。每个 rank 的 stdout/stderr 文件保存在本目录；版本为采集元数据所报 CANN 9.2.0 / torch_npu 2.10.0.post7。",""]
shape_path=p/"operator-shapes.json"
if shape_path.exists():
    y=json.loads(shape_path.read_text())
    assert y["files"]==32 and y["rank_count"]==16 and not y["invalid_or_nonpositive"]
    assert all(y["kernel_sha256_by_dir"][r["source_dir"]]==r["sha256"]["kernel_details.csv"] for r in rows)
    def metrics(z):
        hist=sorted((int(k),v) for k,v in z["histogram_ns"].items())
        n=sum(v for k,v in hist)
        assert n==z["count"] and n==sum(z["rank_counts"].values())
        def at(k):
            total=0
            for value,count in hist:
                total+=count
                if total>=k:return value
            raise AssertionError("rank out of range")
        middle=(at((n+1)//2)+at(n//2+1))/2
        return n, sum(k*v for k,v in hist),hist[0][0],middle,at(math.ceil(n*.95)),hist[-1][0]
    out += ["## 同次类型和完整输入形状：单条任务时长", "",
        "八类算子以窗口、CSV Type 和完整 Input Shapes 分组；两个窗口分别按组内所有记录时长和排序取前 10，并补列指定形状。时长和跨 16 rank 用于本样本的形状筛选，不能代表全机关键路径；下表单条时长单位 µs，median 为 pooled 中位，p95 为升序第 ceil(0.95N) 项，rank 列为出现该组的 rank 数。相同形状不证明同一调用栈、有效数据或工作量；DSpark/target 混在同一标签中。全部组和整数纳秒频数见 [operator-shapes.json](operator-shapes.json)，与前述 32 份 CSV 的 SHA-256 全部一致。", ""]
    for window in (1,2):
        groups=[z for z in y["groups"] if z["window"]==window]
        groups.sort(key=lambda z:metrics(z)[1],reverse=True)
        chosen=groups[:10]
        def target(z):
            t,s=z["type"],z["input_shapes"].strip('"')
            return ((t=="QuantBatchMatmulV3" and s.startswith("3,1280;1024,80,16,32;32768;")) or
                    (t=="TransposeBatchMatMul" and s.startswith("24,1,4096;1,4096,1024")) or
                    (t=="MatMulV2" and s.startswith("3,6144;25600,6144")) or
                    (t=="VllmAscendQuantLightningIndexerV2" and s.startswith("3,32,128;")) or
                    t in ("_prepare_indexer_indices_kernel","_engram_host_uva_gather_dequant_kernel"))
        ids={(z["type"],z["input_shapes"]) for z in chosen}
        for z in groups:
            if target(z) and (z["type"],z["input_shapes"]) not in ids:
                chosen.append(z);ids.add((z["type"],z["input_shapes"]))
        out += [f"### 窗口 {window}","",
          "| 类型 | 完整 Input Shapes | 总记录数 | rank | 跨 rank 时长和 ms | 最小 µs | 中位 µs | p95 µs | 最大 µs |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for z in chosen:
            n,total,lo,median,p95,hi=metrics(z)
            out.append(f'| `{z["type"]}` | `{z["input_shapes"]}` | {n} | {len(z["rank_counts"])} | {total/1000000:.3f} | {lo/1000:.3f} | {median/1000:.3f} | {p95/1000:.3f} | {hi/1000:.3f} |')
        out.append("")
    out += ["候选排序含义：第 1 窗较大的 MC2 物理行形状可支持将 MoE 预填充方向列为优先调查，但窗口 1 同时有小行任务，无法从 shape 唯一标记 prefill。第 2 窗多个 QLI 及小 M Q_b 组大量出现，支持研究 decode 侧投影/Indexer，但单次 QLI 时长受动态可见长度影响，不能将任意一次排序推广到端到端。MatMulV2 的 `[3,6144]×[25600,6144]` 与 Engram 投影相容，权重身份仍未绑定；`[3,1280]` NZ 记录与 Q_b 相容，不能仅靠形状断言每条是 Q_b。通信类型、AI_CPU 任务和 hcom 可能是同一通信的不同视图，不能相加。", "",
        "复算：`python3 diagnostic-20261029/official-analysis/run_analyze_shapes.py` 对已导出的 32 个 CSV 各读一遍，生成全部分组与时长整数纳秒频数，再运行本报告生成脚本。",""]
(p/"OPERATORS.md").write_text("\n".join(out))
print("wrote",p/"OPERATORS.md","parsed",len(rows),"failures",len(x["failures"]))
