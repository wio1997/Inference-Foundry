#!/usr/bin/env python3
"""Group selected types by window and full input shape, preserving duration histograms."""
import collections, csv, decimal, hashlib, json, pathlib, re

root=pathlib.Path("/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/official-20261030T084726Z-cb252d1d/analysis/operator-export/copies")
types={"DispatchFFNCombine","VllmAscendQuantLightningIndexerV2","QuantBatchMatmulV3",
       "TransposeBatchMatMul","MatMulV2","HcPre","_prepare_indexer_indices_kernel",
       "_engram_host_uva_gather_dequant_kernel"}
paths=sorted(root.glob("*_ascend_pt"));assert len(paths)==32
groups=collections.defaultdict(list)
for d in paths:
    m=re.search(r"_ep(\d+)_rank",d.name);assert m
    rank=int(m.group(1)); info=json.loads(next(d.glob("profiler_info_*.json")).read_text())
    assert rank==info["rank_id"]
    groups[rank].append((info["start_info"]["start_monotonic"],d))
assert set(groups)==set(range(16)) and all(len(v)==2 for v in groups.values())
hists=collections.defaultdict(collections.Counter); counts=collections.defaultdict(collections.Counter)
hashes={}; invalid=collections.Counter()
for rank,ts in sorted(groups.items()):
    for window,(_,d) in enumerate(sorted(ts),1):
        p=d/"ASCEND_PROFILER_OUTPUT"/"kernel_details.csv";h=hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
        hashes[d.name]=h.hexdigest()
        with p.open(newline="") as f:
            for row in csv.DictReader(f):
                typ=row["Type"]
                if typ not in types:continue
                try:
                    ns=decimal.Decimal(row["Duration(us)"].strip())*1000
                    if not ns.is_finite() or ns<=0 or ns!=int(ns):raise ValueError()
                    ns=int(ns)
                except (decimal.InvalidOperation,ValueError,TypeError):
                    invalid[(window,typ)]+=1;continue
                key=(window,typ,row.get("Input Shapes",""))
                hists[key][ns]+=1;counts[key][rank]+=1
output=[]
for (w,t,s),hist in sorted(hists.items()):
    output.append({"window":w,"type":t,"input_shapes":s,
                   "histogram_ns":dict(sorted(hist.items())),
                   "count":sum(hist.values()),"rank_counts":dict(sorted(counts[w,t,s].items()))})
print(json.dumps({"run_id":"official-20261030T084726Z-cb252d1d","files":32,
 "rank_count":len(groups),"kernel_sha256_by_dir":hashes,
 "invalid_or_nonpositive":{str(k):v for k,v in invalid.items()},"groups":output},ensure_ascii=False,indent=2))
