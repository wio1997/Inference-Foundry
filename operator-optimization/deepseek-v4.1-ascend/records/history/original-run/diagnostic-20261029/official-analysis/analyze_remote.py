#!/usr/bin/env python3
"""Summarize exported CPU trace CSVs; no device access, all times rank-local."""
import collections, csv, decimal, hashlib, json, pathlib, re, sys

root=pathlib.Path("/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/official-20261030T084726Z-cb252d1d/analysis/operator-export/copies")
dirs=sorted(root.glob("*_ascend_pt"))
rows=[]
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()
def intervals_union(intervals):
    intervals.sort()
    merged=[]
    for a,b in intervals:
        if merged and a<=merged[-1][1]:
            merged[-1]=(merged[-1][0],max(merged[-1][1],b))
        else: merged.append((a,b))
    return merged
def length(intervals):
    return sum(b-a for a,b in intervals_union(intervals))
def intersection(left,right):
    a,b=intervals_union(left),intervals_union(right)
    i=j=total=0
    while i<len(a) and j<len(b):
        total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return total
bad=[]
for d in dirs:
    try:
        match=re.search(r"_ep(\d+)_rank",d.name); assert match
        rank=int(match.group(1))
        infos=list(d.glob("profiler_info_*.json")); assert len(infos)==1
        info=json.loads(infos[0].read_text()); assert info["rank_id"]==rank
        files=d/"ASCEND_PROFILER_OUTPUT"
        kernel=files/"kernel_details.csv"; step=files/"step_trace_time.csv"; stats=files/"op_statistic.csv"
        assert all(p.is_file() for p in (kernel,step,stats))
        bytype=collections.defaultdict(list); compute=[]; comm=[]; other=[]
        core_counts=collections.Counter(); model_counts=collections.Counter(); bad_rows=collections.Counter()
        shape_counts=collections.defaultdict(collections.Counter)
        shape_types={"QuantBatchMatmulV3","SparseFlashMla","DispatchFFNCombine",
                     "VllmAscendQuantLightningIndexerV2","HcPre","HcPost","MatMulV2"}
        with kernel.open(newline="") as f:
            reader=csv.DictReader(f)
            for row in reader:
                core=row["Accelerator Core"]; typ=row["Type"] or "<blank>"
                try:
                    start=decimal.Decimal(row["Start Time(us)"].strip())*1000
                    dur=decimal.Decimal(row["Duration(us)"].strip())*1000
                    if not start.is_finite() or not dur.is_finite() or start!=int(start) or dur!=int(dur) or dur<=0:raise ValueError()
                    start=int(start); end=start+int(dur)
                except (decimal.InvalidOperation,ValueError,TypeError):
                    bad_rows["invalid_or_nonpositive"]+=1; continue
                iv=(start,end);bytype[typ].append(iv)
                core_counts[core]+=1; model_counts[row["Model ID"]]+=1
                if typ in shape_types:
                    shape_counts[typ][row.get("Input Shapes","N/A")]+=1
                if core in ("AI_CORE","AI_VECTOR_CORE","MIX_AIC","MIX_AIV"):compute.append(iv)
                elif core=="COMMUNICATION":comm.append(iv)
                else:other.append(iv)
        assert bytype
        all_intervals=[iv for ivs in bytype.values() for iv in ivs]
        start=min(a for a,b in all_intervals);end=max(b for a,b in all_intervals)
        type_summary={t:{"count":len(v),"sum_ns":sum(b-a for a,b in v),"union_ns":length(v)}
                      for t,v in sorted(bytype.items())}
        with step.open(newline="") as f: step_rows=list(csv.DictReader(f))
        with stats.open(newline="") as f: stat_rows=list(csv.DictReader(f))
        rows.append({"rank":rank,"source_dir":d.name,"record_start_monotonic_ns":info["start_info"]["start_monotonic"],
          "record_end_monotonic_ns":info["end_info"]["MonotonicTimeEnd"],
          "kernel_observed_start_epoch_ns":start,"kernel_observed_end_epoch_ns":end,
          "kernel_observed_span_ns":end-start,"kernel_record_union_ns":length(all_intervals),
          "types":type_summary,"selected_shape_counts":{k:dict(v) for k,v in sorted(shape_counts.items())},
          "compute_label_union_ns":length(compute),
          "communication_label_union_ns":length(comm),"compute_comm_intersection_ns":intersection(compute,comm),
          "other_label_union_ns":length(other),"core_counts":dict(core_counts),"model_counts":dict(model_counts),
          "invalid_or_nonpositive_rows":dict(bad_rows),"step_rows":step_rows,
          "op_stat_row_count":len(stat_rows),"sha256":{"kernel_details.csv":sha(kernel),"step_trace_time.csv":sha(step),"op_statistic.csv":sha(stats)}})
    except Exception as e:
        bad.append({"dir":d.name,"error":type(e).__name__+": "+str(e)})
groups=collections.defaultdict(list)
for row in rows:groups[row["rank"]].append(row)
for rank,rr in groups.items():
    rr.sort(key=lambda x:x["record_start_monotonic_ns"])
    for i,row in enumerate(rr):row["window_ordinal"]=i+1
rows.sort(key=lambda x:(x["window_ordinal"],x["rank"]))
print(json.dumps({"run_id":"official-20261030T084726Z-cb252d1d",
    "export_source_count":len(dirs),"parsed_count":len(rows),"rank_count":len(groups),
    "rank_window_counts":{str(k):len(v) for k,v in sorted(groups.items())},"failures":bad,
    "notes":["Only exported CSV labels; rank clocks and windows remain separate.",
      "MIX_AIC may include MC2 communication; COMMUNICATION is a label, not all physical traffic.",
      "CSV durations are not critical-path time, effective overlap, HBM traffic or request attribution.",
      "Kernel span is earliest to latest selected exported record, not full profiling capture duration."],
    "rows":rows},ensure_ascii=False,indent=2))
